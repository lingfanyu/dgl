import argparse
import os
import logging
import copy
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
from dataset import load_dataset


class Model(nn.Module):
    def __init__(self, in_dim, num_hidden, num_classes, num_layers, num_heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        in_feat = (in_dim, in_dim)  # GATConv needs to know it's bipartite
        for i in range(num_layers - 1):
            self.layers.append(
                GATConv(in_feat, num_hidden, num_heads, feat_drop,
                        attn_drop, negative_slope, residual, activation))
            in_feat = (num_hidden * num_heads, num_hidden * num_heads)
        # output layer
        self.layers.append(
            GATConv(in_feat, num_classes, num_heads, feat_drop, attn_drop,
                    negative_slope, residual, None))

    def forward(self, blocks, h):
        for layer_id, (layer, block) in enumerate(zip(self.layers, blocks)):
            # dst nodes are guaranteed to be at the beginning at src nodes
            h_dst = h[:block.number_of_dst_nodes()]
            h = h, h_dst
            h = layer(block, h)
            if layer_id < self.num_layers - 1:
                # flatten head and feat dimension
                h = h.flatten(1)
            else:
                # output layer, mean reduce over head dimension
                h = h.mean(1)
        return h

    def inference(self, eval_blocks, h):
        """
        Evaluate model layer by layer
        """
        h = h.clone()
        for layer_id, layer in enumerate(self.layers):
            new_h = []
            for block in tqdm.tqdm(eval_blocks):
                h_src = h[block.srcdata[dgl.NID]]
                h_dst = h[block.dstdata[dgl.NID]]
                with block.local_scope():
                    h_dst = layer(block, (h_src, h_dst))
                if layer_id < self.num_layers - 1:
                    # flatten head and feat dimension
                    h_dst = h_dst.flatten(1)
                else:
                    # output layer, mean reduce over head dimension
                    h_dst = h_dst.mean(1)
                new_h.append(h_dst)
            del h  # release memory because inference graph is large
            h = torch.cat(new_h, dim=0)
        return h


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def __call__(self, seeds):
        seeds = torch.LongTensor(seeds)
        blocks = []
        for fanout in self.fanouts:
            # For each layer, sample ``fanout`` neighbors
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            # Compact the frontier into a bipartite graph for message passing
            # For gat model, feature of dst nodes are needed for calculating
            # attention, so the need to include dst in src (True for third
            # argument)
            block = dgl.to_block(frontier, seeds, True)
            blocks.insert(0, block)
            # Obtain the seed nodes for next layer
            seeds = block.srcdata[dgl.NID]
        return blocks


def full_train(model, blocks, features, labels, loss_fcn, optimizer):
    model.train()
    # forward
    for batch_blocks, batch_nodes in tqdm.tqdm(blocks):
        input_nodes = batch_blocks[0].srcdata[dgl.NID]
        pred = model(batch_blocks, features[input_nodes])
        loss = loss_fcn(pred, labels[batch_nodes])
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def sample_train(model, g, features, labels, nodes, batch_size, fan_out,
                 loss_fcn, optimizer):
    # Create sampler
    sampler = NeighborSampler(g, fan_out)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=nodes,
        batch_size=batch_size,
        collate_fn=sampler,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    model.train()
    num_batch = (len(nodes) + batch_size - 1) // batch_size
    for blocks in tqdm.tqdm(dataloader, total=num_batch):
        # load data
        h = features[blocks[0].srcdata[dgl.NID]]
        # forward
        pred = model(blocks, h)
        # compute loss
        batch_labels = labels[blocks[-1].dstdata[dgl.NID]]
        loss = loss_fcn(pred, batch_labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def compute_acc(correct, nid):
    return correct[nid].sum().item() / float(len(nid))


def eval(model, eval_blocks, features, labels, train_nid, val_nid, test_nid):
    model.eval()
    pred = model.inference(eval_blocks, features)
    correct = torch.argmax(pred, dim=1) == labels
    train_acc = compute_acc(correct, train_nid)
    val_acc = compute_acc(correct, val_nid)
    test_acc = compute_acc(correct, test_nid)
    return train_acc, val_acc, test_acc


def get_log_folder(args):
    # create a folder to store training parameters, log, and model state
    log_dir = "gat-ns-{}-bz{}lr{}h{}fan{}drop{},{}wd{}".format(
        args.dataset, args.batch_size, args.lr, args.num_hidden, args.fan_out,
        args.feat_dropout, args.attn_dropout, args.weight_decay)
    if args.log_prefix:
        log_dir = args.log_prefix + "-" + log_dir
    if args.use_sage:
        log_dir += '-sage'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    return log_dir


def setup_logging(log_dir):
    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.INFO)
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler("{}/log.txt".format(log_dir))
    root_logger.addHandler(file_handler)


def build_eval_blocks(g, batch_size, nodes=None):
    if nodes is None:
        nodes = g.nodes()
    blocks = []
    num_nodes = len(nodes)
    n_batch = (num_nodes + batch_size - 1) // batch_size
    for i in range(n_batch):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_nodes)
        seeds = nodes[batch_start: batch_end]
        seeds = torch.LongTensor(seeds)
        block = dgl.to_block(dgl.in_subgraph(g, seeds), seeds)
        blocks.append(block)
    return blocks


def build_train_blocks(g, nodes, num_layers):
    blocks = []
    for _ in range(num_layers):
        block = dgl.to_block(dgl.in_subgraph(g, nodes), nodes, True)
        blocks.insert(0, block)
        nodes = block.srcdata[dgl.NID]
    return blocks


def build_minibatch_train_blocks(g, batch_size, nodes, num_layers):
    train_blocks = []
    num_nodes = len(nodes)
    n_batch = (num_nodes + batch_size - 1) // batch_size
    print("Building training graphs")
    for i in tqdm.trange(n_batch):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_nodes)
        batch_nodes = nodes[batch_start: batch_end]
        blocks = build_train_blocks(g, batch_nodes, num_layers)
        train_blocks.append((blocks, batch_nodes))
    return train_blocks


def main(args):
    # prepare data
    data = load_dataset(args.dataset)
    g, features, labels, num_classes, train_nid, val_nid, test_nid = data

    in_feats = features.shape[1]

    batch_size = args.batch_size
    fan_out = [int(fan_out) for fan_out in args.fan_out.split(',')]
    num_layers = len(fan_out)

    # Define model
    model = Model(in_feats, args.num_hidden, num_classes, num_layers,
                  args.num_heads, F.elu, args.feat_dropout, args.attn_dropout,
                  args.negative_slope, args.use_sage)

    loss_fcn = nn.CrossEntropyLoss()

    # prepare for eval
    eval_blocks = build_eval_blocks(g, args.eval_batch_size)

    # move to device
    if args.gpu < 0:
        train_dev = 'cpu'
    else:
        train_dev = 'cuda:{}'.format(args.gpu)
    features_train = features.to(train_dev)
    labels_train = labels.to(train_dev)
    model.to(train_dev)

    if args.eval_gpu is None:
        eval_dev = train_dev
    else:
        if args.eval_gpu < 0:
            eval_dev = 'cpu'
        else:
            eval_dev = 'cuda:{}'.format(args.eval_gpu)
    features_eval = features.to(eval_dev)
    labels_eval = labels.to(eval_dev)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    log_dir = get_log_folder(args)
    setup_logging(log_dir)
    logging.info(str(args))

    if args.full_neighbor_train:
        train_blocks = build_minibatch_train_blocks(g, args.batch_size,
                                                    train_nid, num_layers)

    best_val = 0
    best_epoch = 0
    print("Start training...")
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        if args.full_neighbor_train:
            full_train(model, train_blocks, features_train, labels_train,
                       loss_fcn, optimizer)
        else:
            sample_train(model, g, features_train, labels_train, train_nid,
                         batch_size, fan_out, loss_fcn, optimizer)

        if epoch % args.evaluate_every == 0:
            model.eval()
            with torch.no_grad():
                model.to(eval_dev)
                train_acc, val_acc, test_acc = eval(
                    model, eval_blocks, features_eval, labels_eval, train_nid,
                    val_nid, test_nid)
                model.to(train_dev)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(log_dir, "model_state"))
            log = "Epoch {}, Time: {:.4f}s".format(epoch, time.time() - start)
            log += ", Accuracy: Train {:.4f} Val {:.4f} Test {:.4f}".format(
                train_acc, val_acc, test_acc)
            logging.info(log)

    logging.info("Best epoch was Epoch {}, Val acc {:.4f}".format(best_epoch,
                                                                  best_val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GraphSage with Neighbor Sampling')
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-hidden", type=int, default=128)
    parser.add_argument("--fan-out", type=str, default="10,10,10")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--feat-dropout", type=float, default=0.5)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-gpu", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--use-sage", action="store_true")
    parser.add_argument("--log-prefix", type=str, default="")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--negative-slope", type=float, default=0.2)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--full-neighbor-train", action="store_true")
    args = parser.parse_args()

    print(args)
    main(args)

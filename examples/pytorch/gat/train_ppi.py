"""
Graph Attention Networks (PPI Dataset) in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code implements
early stopping.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import numpy as np
import time
import torch
import dgl
import torch.nn.functional as F
import argparse
from sklearn.metrics import f1_score
from gat import GAT
from dgl.data.ppi import PPIDataset
from torch.utils.data import DataLoader

def collate(sample):
    start = time.time()
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    end = time.time()
    print("batch graph: {:.6f}".format(end - start), end=' ,')
    return graph, feats, labels

def evaluate(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return score, loss_data.item()

def main(args):
    if args.gpu<0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    batch_size = args.batch_size
    cur_step = 0
    patience = args.patience
    best_score = -1
    best_loss = 10000
    # define loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    # create the dataset
    train_dataset = PPIDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=collate)
    n_classes = train_dataset.labels.shape[1]
    print(n_classes)
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    # define the model
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)
    dur = []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            torch.cuda.synchronize()
            t1 = time.time()
            subgraph, feats, labels = data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph
            print(subgraph.number_of_nodes(), subgraph.number_of_edges())
            exit()
            for layer in model.gat_layers:
                layer.g = subgraph
            torch.cuda.synchronize()
            t2 = time.time()
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            loss_list.append(loss.item())
            t3 = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t4 = time.time()
            print("prep: {:.6f}, forward: {:.6f}, backward: {:6f}".format(t2 - t1, t3 - t2, t4 - t3))
        end = time.time()
        if epoch > 5:
            dur.append(end - start)
            print("{:.5f}".format(np.mean(dur)))
        loss_data = np.array(loss_list).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    args = parser.parse_args()
    print(args)

    main(args)

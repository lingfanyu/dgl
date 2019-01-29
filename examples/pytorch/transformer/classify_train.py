from modules import *
from loss import *
from optims import *
from dataset import *
from modules.config import *
import numpy as np
import argparse
import torch 
import torch.nn.functional as f
from functools import partial

class Generator(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return th.log_softmax(
            self.proj(x), dim=-1
        )

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size, eps=1e-6)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            x = x + layer.sublayer[0].dropout(nodes.data['a'].view(x.shape[0], -1))
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Transformer(nn.Module):
    def __init__(self, encoder, src_embed, pos_enc, generator, h, d_k):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.pos_enc = pos_enc
        self.generator = generator
        self.h, self.d_k = h, d_k

    def propagate_attention(self, g, mat, eids):
        "Workaround solution, this would be replaced by calling built-in function"
        # Compute attention score
        # g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        edata = MaskedMMCSR.apply(mat['ptr_r'], mat['eid_r'], mat['nid_r'], mat['ptr_c'], mat['eid_c'], mat['nid_c'], g.ndata['k'], g.ndata['q'])
        edata = SparseSoftmax.apply(mat['ptr_c'], mat['eid_c'], edata / np.sqrt(self.d_k)) 
        # Send weighted values to target nodes
        # g.send_and_recv(eids,
        #                [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score')],
        #                [fn.sum('v', 'wv'), fn.sum('score', 'z')])
        g.ndata['a'] = VectorSPMM.apply(mat['ptr_c'], mat['eid_c'], mat['nid_c'], mat['ptr_r'], mat['eid_r'], mat['nid_r'], edata, g.ndata['v'])

    def update_graph(self, g, mat, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."
        # Pre-compute queries and key-value pairs.
        for pre_func in pre_pairs:
            g.apply_nodes(pre_func)
        
        self.propagate_attention(g, mat, eids)
        # Further calculation after attention mechanism
        for post_func in post_pairs:
            g.apply_nodes(post_func)

    def forward(self, graph):
        g = graph.g
        mat = graph.mat
        nids, eids = graph.nids, graph.eids

        # embed
        src_embed, src_pos = self.src_embed(graph.src[0]), self.pos_enc(graph.src[1])
        g.ndata['x'] = self.pos_enc.dropout(src_embed + src_pos)

        for i in range(self.encoder.N):
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            self.update_graph(g, mat, eids, [pre_func], [post_func])

        return self.generator(g.ndata['x'][nids])

def make_model(src_vocab, N=8, dim_model=512, dim_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    pos_enc = PositionalEncoding(dim_model, dropout)

    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    src_embed = Embeddings(src_vocab, dim_model)
    generator = Generator(dim_model, src_vocab)
    model = Transformer(
        encoder, src_embed, pos_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model

device = th.device('cuda:0')

dataset = get_dataset('long')
V = dataset.vocab_size
template = EncGraph(mode='sparse')
model = make_model(V).to(device)
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(10):
    train_iter = dataset(template, mode='train', batch_size=8192,
                         device=device)
    for g in train_iter:
        output = model(g)
        loss = F.nll_loss(output, g.tgt_y)
        acc = 1. * (output.max(dim=-1)[1] == g.tgt_y).sum().item() / len(g.tgt_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss: ', loss.item(), '\tacc: ', acc)

"""
Using scipy to produce csr representation is a workaround solution.
"""
import dgl
import torch as th
import numpy as np
import scipy.sparse as sparse
import itertools
import time
from collections import *

Graph = namedtuple('Graph',
                   ['g', 'src', 'tgt', 'tgt_y', 'nids', 'eids', 'mat', 'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])

_relative = {
    'sparse': (
        (np.array([-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]), np.arange(0, 15)), 
        (np.array([0, 1, 2, 4, 8, 16, 32, 64]), np.arange(7, 15))
    ),
    'neigh': (
        (np.arange(-7, 8), np.arange(0, 15)), 
        (np.arange(0, 8), np.arange(7, 15)),
    )
}

_adj_cache = {}

def _create_adj_mat(shift_row, shift_col, n_row, n_col, mode='fully', triu=False):
    global _adj_cache
    if shift_row != shift_col:
        mode = 'fully'
    if (n_row, n_col, mode, triu) in _adj_cache:
        rows, cols, etype = _adj_cache[(n_row, n_col, mode, triu)]
        return rows + shift_row, cols + shift_col, etype
    if mode == 'fully':
        us = np.arange(n_row)
        vs = np.arange(n_col)
        rows = np.repeat(us, n_col)
        cols = np.tile(vs, n_row) 
        if triu:
            cond = rows <= cols 
            rows = np.extract(cond, rows)
            cols = np.extract(cond, cols)
        _adj_cache[(n_row, n_col, mode, triu)] = rows, cols, np.zeros_like(cols) 
        return rows + shift_row, cols + shift_col, np.zeros_like(cols)
    elif mode == 'sparse' or mode == 'neigh':
        us = np.arange(n_row)
        vs = _relative[mode][triu][0]
        etype = _relative[mode][triu][1]
        rows = np.repeat(us, len(vs))
        cols = (np.tile(vs, n_row).reshape(n_row, -1) + np.arange(n_row).reshape(-1, 1)).reshape(-1)
        etype = np.tile(etype, n_row)
        cond = (cols >= 0) & (cols < n_col)
        rows = np.extract(cond, rows)
        cols = np.extract(cond, cols)
        etype = np.extract(cond, etype)
        _adj_cache[(n_row, n_col, mode, triu)] = rows, cols, etype
        return rows + shift_row, cols + shift_col, etype 
    else:
        raise NotImplementedError

class EncGraph:
    def __init__(self, mode='neigh'):
        """
        mode: neigh/sparse
        """
        self.mode = mode

    def create_adj_mat(self, shift_row, shift_col, n_row, n_col, triu=False):
        return [th.from_numpy(arr) for arr in _create_adj_mat(shift_row, shift_col, n_row, n_col, mode=self.mode, triu=triu)]

    def __call__(self, text_buf, label_buf, device='cpu'):
        '''
        Return a batched graph for Encoder side of transformer. 
        args:
            text: a set of text.
            device: 'cpu' or 'cuda:*'
        '''
        text_lens = [len(_) for _ in text_buf]

        src = []
        src_pos = []
        n_nodes, n_edges = 0, 0
        eids = []
        nids = []
        edata = []
        spmat = []
         
        for text_sample, n in zip(text_buf, text_lens):
            src.append(th.tensor(text_sample, dtype=th.long, device=device))
            src_pos.append(th.arange(n, dtype=th.long, device=device))
            nids.append(n_nodes) 
            spmat.append(self.create_adj_mat(n_nodes, n_nodes, n, n))
            row, col, _ = spmat[-1] 
            n_ee = len(row)
            n_nodes += n
            eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
            n_edges += n_ee

        row, col, data = (th.cat(_) for _ in zip(*spmat))
        g = dgl.DGLGraph(sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)), readonly=True)

        eids = th.cat(eids)
        edata = eids.cpu() 
        csr_mat = sparse.csr_matrix((edata, (row, col)), shape=(n_nodes, n_nodes))
        csc_mat = sparse.csc_matrix((edata, (row, col)), shape=(n_nodes, n_nodes))
        mat = {
            'ptr_r': th.tensor(csr_mat.indptr, dtype=th.long, device=device),
            'nid_r': th.tensor(csr_mat.indices, dtype=th.long, device=device),
            'eid_r': th.tensor(csr_mat.data, dtype=th.long, device=device),
            'ptr_c': th.tensor(csc_mat.indptr, dtype=th.long, device=device),
            'nid_c': th.tensor(csc_mat.indices, dtype=th.long, device=device),
            'eid_c': th.tensor(csc_mat.data, dtype=th.long, device=device),
        }

        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     eids=eids, 
                     mat=mat,
                     nids=th.tensor(nids, device=device),
                     tgt_y=th.tensor(label_buf, device=device),
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     nid_arr=None, tgt=None, n_tokens=None)   

class EncDecGraph:
    def __init__(self, mode='fully'):
        """
        mode: fully/sparse/neigh
        """
        self.mode = mode

    def create_adj_mat(self, shift_row, shift_col, n_row, n_col, triu=False):
        return [th.from_numpy(arr) for arr in _create_adj_mat(shift_row, shift_col, n_row, n_col, mode=self.mode, triu=triu)]

    def beam(self, src_buf, start_sym, max_len, k, device='cpu'):
        '''
        Return a Graph class for beam search during inference of Transformer.
        args:
            src_buf: a list of input sequence
            start_sym: the index of start-of-sequence symbol
            max_len: maximum length for decoding
            k: beam size
            device: 'cpu' or 'cuda:*' 
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [max_len] * len(src_buf)

        src, tgt = [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        eids = {'ee': [], 'ed': [], 'dd': []} 
        edata = {'ee': [], 'ed': [], 'dd': []}
        ecnt = {'ee': 0, 'ed': 0, 'dd': 0}
        rows = {'ee': [], 'ed': [], 'dd': []}
        cols = {'ee': [], 'ed': [], 'dd': []} 
        n_nodes, n_edges, n_tokens = 0, 0, 0
        
        spmat = []
        for src_sample, n in zip(src_buf, src_lens):
            for _ in range(k):
                src.append(th.tensor(src_sample, dtype=th.long, device=device))
                src_pos.append(th.arange(n, dtype=th.long, device=device))
                tgt_seq = th.zeros(max_len, dtype=th.long, device=device)
                tgt_seq[0] = start_sym
                tgt.append(tgt_seq)
                tgt_pos.append(th.arange(max_len, dtype=th.long, device=device))
                enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
                spmat.append(self.create_adj_mat(n_nodes, n_nodes, n, n))
                row, col, _ = spmat[-1] 
                rows['ee'].append(row)
                cols['ee'].append(col)
                n_ee = len(row)
                spmat.append(self.create_adj_mat(n_nodes, n_nodes + n, n, max_len))
                row, col, _ = spmat[-1] 
                rows['ed'].append(row)
                cols['ed'].append(col)
                n_ed = len(row)
                spmat.append(self.create_adj_mat(n_nodes + n, n_nodes + n, max_len, max_len, triu=True))
                row, col, _ = spmat[-1] 
                rows['dd'].append(row)
                cols['dd'].append(col)
                n_dd = len(row)
                n_nodes += n
                dec_ids.append(th.arange(n_nodes, n_nodes + max_len, dtype=th.long, device=device))
                n_nodes += max_len
                eids['ee'].append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
                edata['ee'].append(th.arange(ecnt['ee'], ecnt['ee'] + n_ee, dtype=th.long))
                ecnt['ee'] += n_ee
                n_edges += n_ee
                eids['ed'].append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
                edata['ed'].append(th.arange(ecnt['ed'], ecnt['ed'] + n_ed, dtype=th.long))
                ecnt['ed'] += n_ed
                n_edges += n_ed
                eids['dd'].append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
                edata['dd'].append(th.arange(ecnt['dd'], ecnt['dd'] + n_dd, dtype=th.long))
                ecnt['dd'] += n_dd
                n_edges += n_dd
        
        row, col, data = (th.cat(_) for _ in zip(*spmat))
        g = dgl.DGLGraph(sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)), readonly=True)

        mat = {}
        for key in ['ee', 'ed', 'dd']:
            rows[key] = th.cat(rows[key])
            cols[key] = th.cat(cols[key])
            eids[key] = th.cat(eids[key])
            edata[key] = th.cat(edata[key])
            csr_mat = sparse.csr_matrix((edata[key], (rows[key], cols[key])), shape=(n_nodes, n_nodes))
            csc_mat = sparse.csc_matrix((edata[key], (rows[key], cols[key])), shape=(n_nodes, n_nodes))
            mat[key] = {
                'ptr_r': th.tensor(csr_mat.indptr, dtype=th.long, device=device),
                'nid_r': th.tensor(csr_mat.indices, dtype=th.long, device=device),
                'eid_r': th.tensor(csr_mat.data, dtype=th.long, device=device),
                'ptr_c': th.tensor(csc_mat.indptr, dtype=th.long, device=device),
                'nid_c': th.tensor(csc_mat.indices, dtype=th.long, device=device),
                'eid_c': th.tensor(csc_mat.data, dtype=th.long, device=device),
            }

        for key in ['ee', 'ed', 'dd']:
            eids[key] = eids[key].to(device)

        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=None,
                     nids={'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids=eids, 
                     mat=mat,
                     nid_arr={'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)

    def __call__(self, src_buf, tgt_buf, device='cpu'):
        '''
        Return a Graph class for the training phase of Transformer.
        args:
            src_buf: a set of input sequence arrays.
            tgt_buf: a set of output sequence arrays.
            device: 'cpu' or 'cuda:*'
        '''
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [len(_) - 1 for _ in tgt_buf]

        src, tgt, tgt_y = [], [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        eids = {'ee': [], 'ed': [], 'dd': []} 
        edata = {'ee': [], 'ed': [], 'dd': []}
        ecnt = {'ee': 0, 'ed': 0, 'dd': 0}
        rows = {'ee': [], 'ed': [], 'dd': []}
        cols = {'ee': [], 'ed': [], 'dd': []} 
        spmat = []
         
        for src_sample, tgt_sample, n, m in zip(src_buf, tgt_buf, src_lens, tgt_lens):
            src.append(th.tensor(src_sample, dtype=th.long, device=device))
            tgt.append(th.tensor(tgt_sample[:-1], dtype=th.long, device=device))
            tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
            src_pos.append(th.arange(n, dtype=th.long, device=device))
            tgt_pos.append(th.arange(m, dtype=th.long, device=device))
            enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
            spmat.append(self.create_adj_mat(n_nodes, n_nodes, n, n))
            row, col, _ = spmat[-1] 
            rows['ee'].append(row)
            cols['ee'].append(col)
            n_ee = len(row)
            spmat.append(self.create_adj_mat(n_nodes, n_nodes + n, n, m))
            row, col, _ = spmat[-1] 
            rows['ed'].append(row)
            cols['ed'].append(col)
            n_ed = len(row)
            spmat.append(self.create_adj_mat(n_nodes + n, n_nodes + n, m, m, triu=True))
            row, col, _ = spmat[-1] 
            rows['dd'].append(row)
            cols['dd'].append(col)
            n_dd = len(row)
            n_nodes += n
            dec_ids.append(th.arange(n_nodes, n_nodes + m, dtype=th.long, device=device))
            n_nodes += m
            eids['ee'].append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
            edata['ee'].append(th.arange(ecnt['ee'], ecnt['ee'] + n_ee, dtype=th.long))
            ecnt['ee'] += n_ee
            n_edges += n_ee
            eids['ed'].append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
            edata['ed'].append(th.arange(ecnt['ed'], ecnt['ed'] + n_ed, dtype=th.long))
            ecnt['ed'] += n_ed
            n_edges += n_ed
            eids['dd'].append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
            edata['dd'].append(th.arange(ecnt['dd'], ecnt['dd'] + n_dd, dtype=th.long))
            ecnt['dd'] += n_dd
            n_edges += n_dd
            n_tokens += m

        row, col, data = (th.cat(_) for _ in zip(*spmat))
        g = dgl.DGLGraph(sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)), readonly=True)

        mat = {}
        for key in ['ee', 'ed', 'dd']:
            rows[key] = th.cat(rows[key])
            cols[key] = th.cat(cols[key])
            eids[key] = th.cat(eids[key])
            edata[key] = th.cat(edata[key])
            csr_mat = sparse.csr_matrix((edata[key], (rows[key], cols[key])), shape=(n_nodes, n_nodes))
            csc_mat = sparse.csc_matrix((edata[key], (rows[key], cols[key])), shape=(n_nodes, n_nodes))
            mat[key] = {
                'ptr_r': th.tensor(csr_mat.indptr, dtype=th.long, device=device),
                'nid_r': th.tensor(csr_mat.indices, dtype=th.long, device=device),
                'eid_r': th.tensor(csr_mat.data, dtype=th.long, device=device),
                'ptr_c': th.tensor(csc_mat.indptr, dtype=th.long, device=device),
                'nid_c': th.tensor(csc_mat.indices, dtype=th.long, device=device),
                'eid_c': th.tensor(csc_mat.data, dtype=th.long, device=device),
            }

        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=th.cat(tgt_y),
                     nids={'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids=eids, 
                     mat=mat,
                     nid_arr={'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)

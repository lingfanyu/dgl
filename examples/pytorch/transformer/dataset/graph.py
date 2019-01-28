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

relative = {
    'sparse': (np.array([-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]), np.array([0, 1, 2, 4, 8, 16, 32, 64])),
    'neigh': (np.array(list(range(-9, 10))), np.array(list(range(0, 10))))
}

def create_adj_mat(shift_row, shift_col, n_row, n_col, mode='fully', triu=False):
    if mode == 'fully':
        us = np.arange(shift_row, shift_row + n_row)
        vs = np.arange(shift_col, shift_col + n_col)
        rows = np.repeat(us, n_col)
        cols = np.tile(vs, n_row) 
        if triu:
            assert shift_row == shift_col
            assert n_row == n_col
            cond = rows <= cols 
            return np.extract(cond, rows), np.extract(cond, cols) 
        return rows, cols
    elif mode == 'sparse' or mode == 'neigh':
        us = np.arange(shift_row, shift_row + n_row)
        vs = relative[mode][triu] + shift_col
        rows = np.repeat(us, len(vs))
        cols = (np.tile(vs, n_row).reshape(n_row, -1) + np.arange(n_row).reshape(-1, 1)).reshape(-1)
        cond = (cols >= shift_col) & (cols < shift_col + n_col)
        return np.extract(cond, rows), np.extract(cond, cols)
    else:
        raise NotImplementedError

class EncGraph:
    def __init__(self, mode='fully'):
        pass

class EncDecGraph:
    def __init__(self, mode='fully'):
        """
        mode: fully/sparse/neigh
        """
        self.mode = mode

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
        num_edges = {k: [] for k in ['ee', 'ed', 'dd']}
        row_list = {k: [] for k in ['ee', 'ed', 'dd']}
        col_list = {k: [] for k in ['ee', 'ed', 'dd']}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            for _ in range(k):
                g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))
                row_list[key].append(self.us_pool[key][i][j])
                col_list[key].append(self.vs_pool[key][i][j])

        g = dgl.batch(g_list)
        src, tgt = [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        eids = {'ee': [], 'ed': [], 'dd': []} 
        edata = {'ee': [], 'ed': [], 'dd': []}
        ecnt = {'ee': 0, 'ed': 0, 'dd': 0}
        rows = {'ee': [], 'ed': [], 'dd': []}
        cols = {'ee': [], 'ed': [], 'dd': []} 
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, n, n_ee, n_ed, n_dd, row_ee, col_ee, row_ed, col_ed, row_dd, col_dd in zip(src_buf, src_lens, num_edges['ee'], num_edges['ed'], num_edges['dd'], row_list['ee'], col_list['ee'], row_list['ed'], col_list['ed'], row_list['dd'], col_list['dd']):
            for _ in range(k):
                src.append(th.tensor(src_sample, dtype=th.long, device=device))
                src_pos.append(th.arange(n, dtype=th.long, device=device))
                tgt_seq = th.zeros(max_len, dtype=th.long, device=device)
                tgt_seq[0] = start_sym
                tgt.append(tgt_seq)
                tgt_pos.append(th.arange(max_len, dtype=th.long, device=device))
                enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
                rows['ee'].append(row_ee + n_nodes)
                cols['ee'].append(col_ee + n_nodes)
                rows['ed'].append(row_ed + n_nodes)
                cols['ed'].append(col_ed + n_nodes)
                rows['dd'].append(row_dd + n_nodes)
                cols['dd'].append(col_dd + n_nodes)
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
        Return a batched graph for the training phase of Transformer.
        args:
            src_buf: a set of input sequence arrays.
            tgt_buf: a set of output sequence arrays.
            device: 'cpu' or 'cuda:*'
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [len(_) - 1 for _ in tgt_buf]
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        row_list = {k: [] for k in ['ee', 'ed', 'dd']}
        col_list = {k: [] for k in ['ee', 'ed', 'dd']}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))
                row_list[key].append(self.us_pool[key][i][j])
                col_list[key].append(self.vs_pool[key][i][j])

        g = dgl.batch(g_list)
        src, tgt, tgt_y = [], [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        eids = {'ee': [], 'ed': [], 'dd': []} 
        edata = {'ee': [], 'ed': [], 'dd': []}
        ecnt = {'ee': 0, 'ed': 0, 'dd': 0}
        rows = {'ee': [], 'ed': [], 'dd': []}
        cols = {'ee': [], 'ed': [], 'dd': []} 
         
        for src_sample, tgt_sample, n, m, n_ee, n_ed, n_dd, row_ee, col_ee, row_ed, col_ed, row_dd, col_dd in zip(src_buf, tgt_buf, src_lens, tgt_lens, num_edges['ee'], num_edges['ed'], num_edges['dd'], row_list['ee'], col_list['ee'], row_list['ed'], col_list['ed'], row_list['dd'], col_list['dd']):
            src.append(th.tensor(src_sample, dtype=th.long, device=device))
            tgt.append(th.tensor(tgt_sample[:-1], dtype=th.long, device=device))
            tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
            src_pos.append(th.arange(n, dtype=th.long, device=device))
            tgt_pos.append(th.arange(m, dtype=th.long, device=device))
            enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
            rows['ee'].append(row_ee + n_nodes)
            cols['ee'].append(col_ee + n_nodes)
            rows['ed'].append(row_ed + n_nodes)
            cols['ed'].append(col_ed + n_nodes)
            rows['dd'].append(row_dd + n_nodes)
            cols['dd'].append(col_dd + n_nodes)
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

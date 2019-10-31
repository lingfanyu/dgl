import torch
import os
import sys
import pyigl as igl
import geom_utils
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import torch.nn.functional as F
import random
import re
import glob
from iglhelpers import e2p, p2e
from sklearn.externals import joblib
import dgl


def read_npz(seq_names, args):
    with open(seq_names) as fp:
        Xd, Xi = igl.eigen.MatrixXd, igl.eigen.MatrixXi
        eV, eF, eVN = Xd(), Xi(), Xd()
        igl.readOBJ(seq_names, eV, Xd(), eVN, eF, Xi(), Xi())

        new_frame = {}
        npfloat = np.float32
        V, F = e2p(eV), e2p(eF)
        # Fix Degen,
        VN = e2p(eVN).astype(npfloat)
        vdist = VN

        L, weight = None, None

        if not np.isfinite(vdist).all():
            print(f'warning: {seq_names} nan vdist')
            return None

        if 'hack1' in args.additional_opt:
            hack = 1
        elif 'hack0' in args.additional_opt:
            hack = 0

        if 'intrinsic' in args.additional_opt:
            hack = None

        def hackit(Op, h):
            Op.data[np.where(np.logical_not(np.isfinite(Op.data)))[0]] = h
            Op.data[Op.data > 1e10] = h
            Op.data[Op.data < -1e10] = h
            return Op

        if args.uniform_mesh:
            V -= np.min(V, axis=0)
            V /= np.max(V)  # isotropic scaling

        if L is None:
            if hack is None:
                assert 0, "Use hack1"
                import ipdb
                ipdb.set_trace()
                import utils.mesh as mesh
                L = mesh.intrinsic_laplacian(V, F)
            else:
                L = geom_utils.hacky_compute_laplacian(V, F, hack)

        if L is None:
            print("warning: {} no L".format(seq_names))
            return None
        if np.any(np.isnan(L.data)):
            print(f"warning: {seq_names} nan L")
            return None
        new_frame['L'] = L.astype(np.float32)

        input_tensors = {}
        input_tensors['V'] = V

        new_frame['input'] = torch.cat(
            [torch.from_numpy(input_tensors[t].astype(np.float32))
                for t in input_tensors], dim=1)

        # save data to new frame
        new_frame['V'] = V
        new_frame['F'] = F
        new_frame['target_dist'] = torch.from_numpy(vdist).view(-1, 3)
        new_frame['name'] = seq_names
        return new_frame


def sample_batch(seq_names, args, num_batch):
    device = torch.device('cuda' if args.cuda else 'cpu')

    total_nodes = 0
    sample_id = 0
    nsample = 0
    graph_batch = []
    sample_names = []
    batch_count = 0

    if args.shuffle:
        random.shuffle(seq_names)

    while True:
        new_sample = None
        seq_choice = seq_names[sample_id]
        sample_id += 1
        if sample_id >= len(seq_names):
            sample_id = 0
        if isinstance(seq_choice, str) and os.path.isfile(seq_choice):
            new_sample = read_npz(seq_choice, args)
        else:
            assert args.pre_load
            new_sample = seq_choice
        assert(new_sample is not None)

        num_vertices, input_channel = new_sample['input'].shape

        if args.var_size and total_nodes + num_vertices >= args.threshold:
            if num_vertices > args.threshold:
                print("Sample ignored"
                      f"has {num_vertices} > {args.threshold} nodes")
                continue
            yield dgl.batch(graph_batch), sample_names
            batch_count += 1
            if batch_count == num_batch:
                return
            graph_batch = []
            sample_names = []
            nsample = 0
            total_nodes = 0

        L = new_sample['L']
        graph = dgl.DGLGraph(L, readonly=True)
        graph.edata['L'] = torch.from_numpy(L.data).to(device).view(-1, 1)
        graph.ndata['input'] = new_sample['input'].to(device)
        graph.ndata['mask'] = torch.ones(num_vertices, 1).to(device)
        graph.ndata['target'] = new_sample['target_dist'].to(device)
        graph_batch.append(graph)
        sample_names.append(new_sample['name'])
        nsample += 1
        total_nodes += num_vertices

        if not args.var_size and nsample == args.batch_size:
            yield dgl.batch(graph_batch), sample_names
            batch_count += 1
            if batch_count == args.num_updates:
                return
            graph_batch = []
            sample_names = []
            nsample = 0
            total_nodes = 0

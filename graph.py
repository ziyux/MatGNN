import pickle
import numpy as np
import scipy.sparse as sp

import torch as th
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI

from dgl.convert import graph as dgl_graph
from dgl.transform import to_bidirected
from dgl import backend as F


def save_pickle(filename, obj):
    with open(filename, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


device = th.device('cuda' if th.cuda.is_available() else 'cpu')


class GraphConstructor(object):
    def __init__(self,
                 connect_method='CWC',
                 cutoff=np.inf,
                 k=None,
                 gaussian_step=None):
        self.connect_method = connect_method
        self.cutoff = cutoff
        self.k = k
        self.gaussian_step = gaussian_step
        if self.gaussian_step is not None:
            self.ge = GaussianExpansion(0, cutoff, step=gaussian_step)

    def construct_graph(self, R, node_fea=None, edge_fea=None, node_fea_dtypes=None, edge_fea_dtypes=None):
        # if self.data is None:
        #     self.data = Dataset(self.dataset_name, self.feature_name, self.label_name, self.dataset_dir)
        # if self.connect_method == 'CWC':
        #     g, dist, adj = self.connect_within_cutoff(R)
        if self.connect_method == 'CWC':
            graph, dist, adj = self.connect_within_cutoff(R, self.cutoff, self.k)
        graph = self.feature_assignment(graph, dist, adj, node_fea, edge_fea, node_fea_dtypes, edge_fea_dtypes)
        return graph

    # def connect_within_cutoff(self, R):
    #     dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    #     adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(len(R), dtype=np.bool)
    #     adj = adj.tocoo()
    #     u, v = F.tensor(adj.row), F.tensor(adj.col)
    #     g = dgl_graph((u, v))
    #     g = to_bidirected(g)
    #     return g, dist, adj

    def connect_within_cutoff(self, R, cutoff=np.inf, k=None):
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        adj = np.array(dist <= cutoff) - np.identity(len(R))
        if k is not None:
            adj = self.choose_k_nearest(adj, dist, k)
        adj = adj + adj.T
        adj = sp.coo_matrix(adj, dtype=np.bool)
        u, v = F.tensor(adj.row), F.tensor(adj.col)
        graph = dgl_graph((u, v))
        graph = to_bidirected(graph)
        return graph, dist, adj

    def choose_k_nearest(self, adj, dist, k):
        for row in range(len(adj)):
            if len(dist[row, :]) > k:
                threshold = dist[row, dist[row, :].argsort()[k - 1]]
                adj[row, dist[row, :] > threshold] = 0
        return adj

    def feature_assignment(self, graph, dist, adj, node_fea, edge_fea, node_fea_dtypes, edge_fea_dtypes):
        for key, fea in node_fea.items():
            fea = F.tensor(fea, dtype=node_fea_dtypes[key])
            # if len(fea.shape) < 2:
            #     fea = fea[:, None]
            graph.ndata[key] = fea
        for key, fea in edge_fea.items():
            if key == 'D':
                fea = np.array(sp.csr_matrix(dist)[adj]).flatten()
            elif key == 'GD':
                D = np.array(sp.csr_matrix(dist)[adj]).flatten()
                fea = self.ge.expand(D)
            fea = F.tensor(fea, dtype=edge_fea_dtypes[key])
            # if len(fea.shape) < 2:
            #     fea = fea[:, None]
            graph.edata[key] = fea

        return graph


class GaussianExpansion(object):

    def __init__(self,
                 dmin,
                 dmax,
                 step,
                 var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        assert dmax < np.inf
        self.center = np.arange(dmin, dmax + step, step)
        self.var = step if var is None else var

    def expand(self, dist):
        return np.exp(-(dist[..., None] - self.center) ** 2 / self.var ** 2)


def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False):

    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = th.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    print(adj_t_row.set_value(None))
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(th.long)
    print(num_triplets)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    print(mask)
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = th.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = th.atan2(b, a)

    idx_batch = th.arange(len(idx_i), device=device)
    idx_k_n = adj_t[idx_j].storage.col()
    repeat = num_triplets - 1
    num_triplets_t = num_triplets.repeat_interleave(repeat)
    idx_i_t = idx_i.repeat_interleave(num_triplets_t)
    idx_j_t = idx_j.repeat_interleave(num_triplets_t)
    idx_k_t = idx_k.repeat_interleave(num_triplets_t)
    idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
    mask = idx_i_t != idx_k_n
    idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], \
                                                      idx_batch_t[mask]

    # Calculate torsions.
    if use_torsion:
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = th.cross(pos_ji, pos_j0)
        plane2 = th.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (th.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        torsion1 = th.atan2(b, a)  # -pi to pi
        torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
        torsion = scatter(torsion1, idx_batch_t, reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji

    else:
        return dist, angle, i, j, idx_kj, idx_ji
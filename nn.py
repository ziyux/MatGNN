import torch as th
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from dgl.nn.pytorch.utils import Identity
from dgl.nn.pytorch import Set2Set


gain_default = 1

# class Embedding(nn.Module):
#
#     def __init__(self,
#                  node_type_num,
#                  embedding_dim):
#         super(Embedding, self).__init__()
#         self.node_type_num = node_type_num
#         self.embedding_dim = embedding_dim
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.embedding_matrix = th.rand(self.node_type_num, self.embedding_dim)
#
#     def forward(self, feat):
#         # feat = th.tensor(feat, dtype=th.int64)
#         embeddings = F.embedding(feat, self.embedding_matrix)
#         return embeddings


# class GaussianExpansion(nn.Module):
#
#     def __init__(self,
#                  dmin,
#                  dmax,
#                  step,
#                  var=None):
#         super(GaussianExpansion, self).__init__()
#         assert dmin < dmax
#         assert dmax - dmin > step
#         self.center = th.arange(dmin, dmax + step, step)
#         self.var = step if var is None else var
#
#     def forward(self, feat):
#         return th.exp(-(feat[..., None] - self.center) ** 2 / self.var ** 2)


class NNBase(nn.Module):

    def __init__(self,
                 out_feats,
                 mpnn,
                 mpnn_feats,
                 hid_feats=(32, 16),
                 act='relu',
                 node_embedding=None,
                 aggregator_type='mean'):
        super(NNBase, self).__init__()
        self._out_feats = out_feats
        self.node_embeddings = node_embedding
        self.mpnn = mpnn
        self.act = act
        self.set2set = nn.ModuleDict({'node': Set2Set(mpnn_feats, n_iters=3, n_layers=1),
                                     'edge': Set2Set(mpnn_feats, n_iters=3, n_layers=1)})
        self.dense_layers = nn.ModuleList()
        hid_feats = [4*mpnn_feats] + list(hid_feats) + [out_feats]
        for i in range(len(hid_feats) - 1):
            self.dense_layers.append(nn.Linear(hid_feats[i], hid_feats[i+1]))

        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type

        self.reset_parameters()

    def reset_parameters(self):

        try:
            gain = nn.init.calculate_gain(self.act)
        except:
            gain = gain_default
        for i in range(len(self.dense_layers)):
            nn.init.xavier_normal_(self.dense_layers[i].weight, gain=gain)

    def forward(self, graph, feat, efeat):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            node_emb = self.node_embeddings(feat_src) if self.node_embeddings is not None else feat_src
            edge_emb = efeat

            node_emb, edge_emb = self.mpnn(graph, node_emb, edge_emb)

            node_s2s = self.set2set['node'](graph, node_emb)
            graph.edata['emb'] = edge_emb
            graph.update_all(fn.copy_e('emb', 'm'), self.reducer('m', 'agg'))
            edge_agg = graph.ndata['agg']
            edge_s2s = self.set2set['edge'](graph, edge_agg)
            out = th.cat((node_s2s, edge_s2s), 1)

            for i in range(len(self.dense_layers)):
                out = self.dense_layers[i](out)
                if i < len(self.dense_layers) - 1:
                    out = getattr(F, self.act)(out)
            return out



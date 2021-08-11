import torch as th
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from dgl.nn.pytorch.utils import Identity

gain_default = 1


class MEGNet(nn.Module):

    def __init__(self,
                 in_feats,
                 in_efeats,
                 megnet_num=2,
                 dim1=64,
                 dim2=32,
                 act='relu',
                 aggregator_type='mean',
                 dense_layer_num=2,
                 mlp_layer=3):
        super(MEGNet, self).__init__()
        self.megnet_blocks = nn.ModuleList()
        for i in range(megnet_num):
            if i == 0:
                self.megnet_blocks.append(
                    MEGNetBlock(in_feats, in_efeats, dim1, dim2, act,
                                aggregator_type, dense_layer_num, mlp_layer))
            else:
                self.megnet_blocks.append(
                    MEGNetBlock(dim2, dim2, dim1, dim2, act,
                                aggregator_type, dense_layer_num, mlp_layer))

    def forward(self, graph, feat, efeat):

        with graph.local_scope():
            vout, eout = feat, efeat
            for n in range(len(self.megnet_blocks)):
                vout, eout = self.megnet_blocks[n](graph, vout, eout, n)
            return vout, eout


class MEGNetBlock(nn.Module):

    def __init__(self,
                 in_feats,
                 in_efeats,
                 dim1,
                 dim2,
                 act,
                 aggregator_type,
                 dense_layer_num,
                 mlp_layer):
        super(MEGNetBlock, self).__init__()
        self.act = act
        self.dense_layer_num = dense_layer_num
        self.dense_layers = nn.ModuleDict({'node': nn.ModuleList(),
                                           'edge': nn.ModuleList()})
        self.edge_update = EdgeUpdate(dim2, act, dim1, dim2, mlp_layer)
        self.node_update = NodeUpdate(dim2, dim2, act, dim1, dim2, aggregator_type, mlp_layer)

        for i in range(dense_layer_num):
            if i == 0:
                self.dense_layers['node'].append(nn.Linear(in_feats, dim1))
                self.dense_layers['edge'].append(nn.Linear(in_efeats, dim1))
            elif i == (dense_layer_num - 1):
                self.dense_layers['node'].append(nn.Linear(dim1, dim2))
                self.dense_layers['edge'].append(nn.Linear(dim1, dim2))
            else:
                self.dense_layers['node'].append(nn.Linear(dim1, dim1))
                self.dense_layers['edge'].append(nn.Linear(dim1, dim1))

        self.reset_parameters()

    def reset_parameters(self):
        try:
            gain = nn.init.calculate_gain(self.act)
        except:
            gain = gain_default
        for i in range(self.dense_layer_num):
            nn.init.xavier_normal_(self.dense_layers['node'][i].weight, gain=gain)
            nn.init.xavier_normal_(self.dense_layers['edge'][i].weight, gain=gain)

    def forward(self, graph, feat, efeat, n):

        with graph.local_scope():
            hx = feat
            he = efeat
            for i in range(self.dense_layer_num):
                hx = self.dense_layers['node'][i](hx)
                hx = getattr(F, self.act)(hx)
                he = self.dense_layers['edge'][i](he)
                he = getattr(F, self.act)(he)
            eout = self.edge_update(graph, hx, he)
            vout = self.node_update(graph, hx, eout)
            # residual connection
            if n == 0:
                eout = th.add(eout, he)
                vout = th.add(vout, hx)
            else:
                eout = th.add(eout, efeat)
                vout = th.add(vout, feat)
            return vout, eout


class EdgeUpdate(nn.Module):

    def __init__(self,
                 in_feats,
                 act,
                 hid_feats=64,
                 out_feats=32,
                 mlp_layer=3):
        super(EdgeUpdate, self).__init__()
        self.act = act
        self.mlp_layer = mlp_layer
        self.mlp = nn.ModuleList()

        if mlp_layer == 1:
            mlp = nn.ModuleDict({'vsk': nn.Linear(in_feats, out_feats),
                                 'vrk': nn.Linear(in_feats, out_feats),
                                 'ek': nn.Linear(in_feats, out_feats)})
            self.mlp.append(mlp)
        else:
            for i in range(mlp_layer):
                if i == 0:
                    mlp = nn.ModuleDict({'vsk': nn.Linear(in_feats, hid_feats),
                                         'vrk': nn.Linear(in_feats, hid_feats),
                                         'ek': nn.Linear(in_feats, hid_feats)})
                    self.mlp.append(mlp)
                elif i == mlp_layer - 1:
                    self.mlp.append(nn.Linear(hid_feats, out_feats))
                else:
                    self.mlp.append(nn.Linear(hid_feats, hid_feats))
        self.reset_parameters()

    def reset_parameters(self):

        try:
            gain = nn.init.calculate_gain(self.act)
        except:
            gain = gain_default
        for i in range(self.mlp_layer):
            if i == 0:
                nn.init.xavier_normal_(self.mlp[i]['vsk'].weight, gain=gain)
                nn.init.xavier_normal_(self.mlp[i]['vrk'].weight, gain=gain)
                nn.init.xavier_normal_(self.mlp[i]['ek'].weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.mlp[i].weight, gain=gain)

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['vsk'] = feat_src
            graph.dstdata['vrk'] = feat_dst
            graph.edata['ek'] = efeat
            for i in range(self.mlp_layer):
                if i == 0:
                    graph.srcdata['vsk'] = self.mlp[i]['vsk'](graph.srcdata['vsk'])
                    graph.dstdata['vrk'] = self.mlp[i]['vrk'](graph.dstdata['vrk'])
                    graph.edata['ek'] = self.mlp[i]['ek'](graph.edata['ek'])
                    graph.apply_edges(fn.u_add_v('vsk', 'vrk', 'vk'))
                    out = graph.edata['vk'] + graph.edata['ek']
                    out = getattr(F, self.act)(out)
                else:
                    out = self.mlp[i](out)
                    out = getattr(F, self.act)(out)
            return out


class NodeUpdate(nn.Module):

    def __init__(self,
                 in_feats,
                 in_efeats,
                 act,
                 hid_feats=64,
                 out_feats=32,
                 aggregator_type='mean',
                 mlp_layer=3):
        super(NodeUpdate, self).__init__()
        self.act = act
        self.mlp_layer = mlp_layer
        self.mlp = nn.ModuleList()

        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type

        if mlp_layer == 1:
            mlp = nn.ModuleDict({'vi': nn.Linear(in_feats, out_feats),
                                 'vie': nn.Linear(in_efeats, out_feats)})
            self.mlp.append(mlp)
        else:
            for i in range(mlp_layer):
                if i == 0:
                    mlp = nn.ModuleDict({'vi': nn.Linear(in_feats, hid_feats),
                                         'vie': nn.Linear(in_efeats, hid_feats)})
                    self.mlp.append(mlp)
                elif i == mlp_layer - 1:
                    self.mlp.append(nn.Linear(hid_feats, out_feats))
                else:
                    self.mlp.append(nn.Linear(hid_feats, hid_feats))
        self.reset_parameters()

    def reset_parameters(self):

        try:
            gain = nn.init.calculate_gain(self.act)
        except:
            gain = gain_default
        for i in range(self.mlp_layer):
            if i == 0:
                nn.init.xavier_normal_(self.mlp[i]['vi'].weight, gain=gain)
                nn.init.xavier_normal_(self.mlp[i]['vie'].weight, gain=gain)
            else:
                nn.init.xavier_normal_(self.mlp[i].weight, gain=gain)

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['vi'] = feat_src
            graph.edata['ei'] = efeat
            for i in range(self.mlp_layer):
                if i == 0:
                    graph.srcdata['vi'] = self.mlp[i]['vi'](graph.srcdata['vi'])
                    graph.update_all(fn.copy_e('ei', 'neigh'), self.reducer('neigh', 'vie'))
                    graph.srcdata['vie'] = self.mlp[i]['vie'](graph.srcdata['vie'])
                    out = graph.ndata['vi'] + graph.ndata['vie']
                    out = getattr(F, self.act)(out)
                else:
                    out = self.mlp[i](out)
                    out = getattr(F, self.act)(out)
            return out

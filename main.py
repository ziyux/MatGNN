from MatGNN import MatGNN
from nn import NNBase
from nn_util.MEGNet import MEGNet
import torch.nn as F
from graph import xyz_to_dat
import numpy as np
import torch

dataset_name = 'QM9'
node_fea_sel = ['atomic_number', 'R']
edge_fea_sel = ['GD']
label_name = ['U0_atom']
cutoff = 5
k = 12
connect_method = 'CWC'
train_rate = 0.90
val_rate = 0.05
test_rate = 0.05
act = 'relu'
resume = True

mpnn = lambda feat, efeat: MEGNet(feat, efeat,
                                  megnet_num=3,
                                  dim1=64, dim2=32,
                                  act=act,
                                  aggregator_type='mean',
                                  dense_layer_num=2,
                                  mlp_layer=3)

node_embedding = F.Embedding(10, 16)
model = lambda feat, efeat: NNBase(out_feats=1,
                                   mpnn=mpnn(feat, efeat),
                                   mpnn_feats=32,
                                   act=act,
                                   node_embedding=node_embedding)

nn = MatGNN(dataset_name=dataset_name,
            model=model,
            label_sel=label_name,
            node_fea_sel=node_fea_sel,
            edge_fea_sel=edge_fea_sel,
            connect_method=connect_method,
            cutoff=cutoff,
            k=k,
            train_rate=train_rate,
            val_rate=val_rate,
            test_rate=test_rate,
            resume=resume,
            resume_model='bestmodel.tar')

# nn.train()
# num_nodes = nn.dataset[0][0].num_nodes()
# print(num_nodes)
# print(nn.dataset[0][0].num_edges())
# edge_index = [torch.tensor(i, dtype=torch.long) for i in nn.dataset[0][0].edges()]

# dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(nn.dataset[0][0].ndata['R'], edge_index, num_nodes, use_torsion=True)
# # print(len(dist))
# # print(len(angle))
# # print(len(torsion))
# # print(len(i))
# print(len(idx_kj))
# print(len(idx_ji))

nn.score()
import os
import pickle
import numpy as np
import torch as th

from dgl.data import DGLDataset
from dgl.data.utils import download, _get_dgl_url
from dgl import backend as F


def get_dataset(dataset_name,
                graph_constructor,
                label_sel,
                node_fea_sel=None,
                edge_fea_sel=None,
                url=None,
                root_dir=None,
                force_reload=True,
                verbose=False):
    if dataset_name == 'QM9':
        return QM9(dataset_name, graph_constructor, label_sel, node_fea_sel, edge_fea_sel, url, root_dir, force_reload,
                   verbose)
    else:
        return QM9(dataset_name, graph_constructor, label_sel, node_fea_sel, edge_fea_sel, url, root_dir, force_reload,
                   verbose)


class DatasetPartition(DGLDataset):

    def __init__(self,
                 set_name,
                 dataset,
                 mask,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.dataset = dataset
        self.mask = mask
        super(DatasetPartition, self).__init__(name=set_name,
                                               url=url,
                                               raw_dir=raw_dir,
                                               save_dir=save_dir,
                                               force_reload=force_reload,
                                               verbose=verbose)

    def __getitem__(self, idx):
        # get one example by index
        g = self.dataset[self.mask[idx]][0]
        label = self.dataset[self.mask[idx]][1]
        return g, label

    def __len__(self):
        # number of data examples
        return len(self.mask)


class QM9(DGLDataset):

    def __init__(self,
                 dataset_name,
                 graph_constructor,
                 label_sel,
                 node_fea_sel=None,
                 edge_fea_sel=None,
                 url=None,
                 root_dir=None,
                 force_reload=True,
                 verbose=False):
        self.label_name = label_sel
        self.gc = graph_constructor
        self.node_fea_sel = node_fea_sel
        self.edge_fea_sel = edge_fea_sel
        raw_dir = save_dir = os.path.join(root_dir, 'dataset')
        if url is None:
            url = self.get_url(dataset_name)
        # ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom',
        #         'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
        self.label_keys = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'r2': 5, 'zpve': 6, 'U0': 7, 'U': 8,
                           'H': 9, 'G': 10, 'Cv': 11, 'U0_atom': 12, 'U_atom': 13, 'H_atom': 14, 'G_atom': 15,
                           'A': 16, 'B': 17, 'C': 18}
        # ['atom_type', 'atomic_number', 'aromatic', 'sp', 'sp2', 'sp3', 'num_hs']
        # atom_type = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.node_fea_keys = {'atom_type': range(0, 4), 'atomic_number': 5, 'aromatic': 6, 'sp': 7, 'sp2': 8,
                              'sp3': 9, 'num_hs': 10}
        self.node_fea_dtypes = {'atom_type': th.int64, 'atomic_number': th.int64, 'aromatic': th.int64,
                                'sp': th.int64, 'sp2': th.int64, 'sp3': th.int64, 'num_hs': th.int64, 'R': th.float32}
        # bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.edge_fea_keys = {'bonds': range(0, 3)}
        self.edge_fea_dtypes = {'bonds': th.int64, 'D': th.float32, 'GD': th.float32}
        super(QM9, self).__init__(name=dataset_name,
                                  url=url,
                                  raw_dir=raw_dir,
                                  save_dir=save_dir,
                                  force_reload=force_reload,
                                  verbose=verbose)

    def get_url(self, dataset_name):
        url = _get_dgl_url('dataset/qm9_edge.npz')
        return url

    def download(self):
        file_path = f'{self.raw_dir}/qm9_edge.npz'
        if not os.path.exists(file_path):
            download(self._url, path=file_path)

    def process(self):
        npz_path = f'{self.raw_dir}/qm9_edge.npz'
        data_dict = np.load(npz_path, allow_pickle=True)
        # data_dict['N'] contains the number of atoms in each molecule.
        # Atomic properties (Z and R) of all molecules are concatenated as single tensors,
        # so you need this value to select the correct atoms for each molecule.
        self.N = data_dict['n_node']
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        self.NE = data_dict['n_edge']
        self.NE_cumsum = np.concatenate([[0], np.cumsum(self.NE)])
        self.node_fea = {}
        self.edge_fea = {}
        self.node_fea['R'] = data_dict['node_pos']
        for key, idx in self.node_fea_keys.items():
            self.node_fea[key] = data_dict['node_attr'][:, idx]
        for key, idx in self.edge_fea_keys.items():
            self.edge_fea[key] = data_dict['edge_attr'][:, idx]
        self.label = np.stack([data_dict['targets'][:, self.label_keys[key]] for key in self.label_name], axis=1)

    def __getitem__(self, idx):
        # get one example by index
        label = F.tensor(self.label[idx], dtype=F.data_type_dict['float32'])
        node_fea, edge_fea = {}, {}
        for key in self.node_fea_sel:
            node_fea[key] = self.node_fea[key][self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        for key in self.edge_fea_sel:
            if key == 'D' or key == 'GD':
                edge_fea[key] = None
            else:
                edge_fea[key] = self.edge_fea[key][self.NE_cumsum[idx]:self.NE_cumsum[idx + 1]]
        R = self.node_fea['R'][self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        graph = self.gc.construct_graph(R, node_fea, edge_fea, self.node_fea_dtypes, self.edge_fea_dtypes)
        return graph, label

    def __len__(self):
        r"""Number of graphs in the dataset.

              Return
              -------
              int
              """
        return self.label.shape[0]

    def save_pickle(self, filename, obj):
        with open(filename, 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def read_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self):
        # save processed data to directory `self.save_path`
        # filename = os.path.join(self.raw_dir, self.name+'_data.pkl')
        # self.save_pickle(filename, self.data)
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        # filename = os.path.join(self.raw_dir, self.name+'_data.pkl')
        # self.data = self.read_pickle(filename)
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        # filename = os.path.join(self.raw_dir, self.name+'_data.pkl')
        # return os.path.exists(filename)
        pass

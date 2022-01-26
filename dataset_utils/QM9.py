import torch
import numpy as np
from tqdm import tqdm


from dataset_utils.matdataset import MatDataset
from graph import GraphConstructor


class QM9(MatDataset):

    def __init__(self,
                 dataset_name,
                 label_name,
                 connect_method=None,
                 cutoff=5,
                 save_graphs=True,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 save_name='',
                 force_reload=False,
                 verbose=False,
                 download_name='qm9_edge.npz',
                 k=12,
                 gaussian_step=None,
                 node_fea_sel=(),
                 edge_fea_sel=(),
                 **kwargs):

        self.gc = GraphConstructor(connect_method='CWC' if connect_method is None else connect_method,
                                   cutoff=cutoff,
                                   k=k,
                                   gaussian_step=gaussian_step,
                                   **kwargs)
        self.save_graphs = save_graphs
        self.node_fea_sel = node_fea_sel
        self.edge_fea_sel = edge_fea_sel

        if url is None:
            url = self.get_url('dataset/qm9_edge.npz')
        # ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom',
        #         'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
        self.label_keys = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'r2': 5, 'zpve': 6, 'U0': 7, 'U': 8,
                           'H': 9, 'G': 10, 'Cv': 11, 'U0_atom': 12, 'U_atom': 13, 'H_atom': 14, 'G_atom': 15,
                           'A': 16, 'B': 17, 'C': 18}
        # ['atom_type', 'atomic_number', 'aromatic', 'sp', 'sp2', 'sp3', 'num_hs']
        # atom_type = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.node_fea_keys = {'atom_type': range(0, 4), 'atomic_number': 5, 'aromatic': 6, 'sp': 7, 'sp2': 8,
                              'sp3': 9, 'num_hs': 10}
        self.node_fea_dtypes = {'atom_type': torch.int64, 'atomic_number': torch.int64, 'aromatic': torch.int64,
                                'sp': torch.int64, 'sp2': torch.int64, 'sp3': torch.int64, 'num_hs': torch.int64}
        # bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.edge_fea_keys = {'bonds': range(0, 3)}
        self.edge_fea_dtypes = {'bonds': torch.int64}
        super(QM9, self).__init__(dataset_name=dataset_name,
                                  label_name=label_name,
                                  url=url,
                                  raw_dir=raw_dir,
                                  save_dir=save_dir,
                                  save_name=save_name,
                                  force_reload=force_reload,
                                  verbose=verbose,
                                  download_name=download_name)

    def process(self):
        npz_path = f'{self.save_path}/{self.download_name}'
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
        self.node_fea['coords'] = torch.tensor(data_dict['node_pos'], dtype=torch.float32)
        for key, idx in self.node_fea_keys.items():
            self.node_fea[key] = torch.tensor(data_dict['node_attr'][:, idx], dtype=self.node_fea_dtypes[key])
        for key, idx in self.edge_fea_keys.items():
            self.edge_fea[key] = torch.tensor(data_dict['edge_attr'][:, idx], dtype=self.edge_fea_dtypes[key])
        for key, idx in self.label_keys.items():
            self.label_dict[key] = torch.tensor(data_dict['targets'][:, idx], dtype=torch.float32)
        if self.save_graphs:
            idxs = tqdm(range(len(self.N)), desc='Construct Graphs') if self.verbose else range(len(self.N))
            for idx in idxs:
                self.graphs_saved.append(self.construct_graph(idx))

    def getitem(self, idx):
        # get one example by index
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        graph = self.construct_graph(idx)
        return graph, label

    def construct_graph(self, idx):
        node_fea, edge_fea = {}, {}
        for key in self.node_fea_sel:
            node_fea[key] = self.node_fea[key][self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        for key in self.edge_fea_sel:
            edge_fea[key] = self.edge_fea[key][self.NE_cumsum[idx]:self.NE_cumsum[idx + 1]]
        coords = self.node_fea['coords'][self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        if self.gc.connect_method == 'CWC':
            graph = self.gc.connect_graph(coords)
        else:
            raise Exception('Unsupported connection method: ', self.gc.connect_method)
        graph = self.gc.feature_assignment(graph, node_fea, edge_fea)
        return graph

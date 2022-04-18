import os
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from dgl.data.utils import save_info, load_info, save_graphs, load_graphs

from dataset_utils.matdataset import MatDataset
from graph import GraphConstructor, SphericalFeatures
from fe_utils.sph_fea import xyz_to_dat


class MaterialsProject(MatDataset):

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
                 download_name='lookup_table.csv',
                 api_key=None,
                 criteria=None,
                 properties=None,
                 step=1000,
                 node_fea_sel=None,
                 edge_fea_sel=None,
                 **kwargs):
        self.gc = GraphConstructor(connect_method='PBC' if connect_method is None else connect_method,
                                   cutoff=cutoff,
                                   **kwargs)
        self.save_graphs = save_graphs
        self.api_key = api_key
        self.step = step
        self.criteria = criteria
        self.properties = properties if properties is not None \
            else ['uncorrected_energy_per_atom', 'formation_energy_per_atom']
        self.lookup = None
        super(MaterialsProject, self).__init__(dataset_name=dataset_name,
                                               label_name=label_name,
                                               url=url,
                                               raw_dir=raw_dir,
                                               save_dir=save_dir,
                                               save_name=save_name,
                                               force_reload=force_reload,
                                               verbose=verbose,
                                               download_name=download_name)
        # load saved data
        if 'cells' in self.data_saved:
            self.cells = self.data_saved['cells']

    def download(self):
        pass

    def process(self):
        df = self.criteria
        structures = df['structure'].values
        cells = []
        for structure in structures:
            cells.append(Structure.from_dict(structure))
        label_dict = {key: [] for key in self.properties}
        for label_name in self.properties:
            label_dict[label_name] = torch.tensor(df[label_name].values)
        self.label_dict = label_dict
        self.data_saved = {'cells': cells}

        if self.save_graphs:
            for cell in tqdm(cells, desc='Construct Graphs'):
                self.graphs_saved.append(self.construct_graph(cell))

    def construct_graph(self, cell):
        if self.gc.connect_method == 'PBC' or self.gc.connect_method == 'CGC':
            graph = self.gc.connect_graph(cell)
            image = graph.ndata['image']
            atomic_number = torch.tensor([element.data['Atomic no']
                                          for element in cell.species], dtype=torch.int64)[image]
        elif self.gc.connect_method == 'CWC':
            graph = self.gc.connect_graph(cell.cart_coords)
            atomic_number = torch.tensor([element.data['Atomic no']
                                          for element in cell.species], dtype=torch.int64)
        else:
            raise Exception('Unsupported connection method: ', self.gc.connect_method)
        node_fea = {'atomic_number': atomic_number}
        graph = self.gc.feature_assignment(graph, node_fea)
        return graph

    def getitem(self, idx):
        # get one example by index
        if self.save_graphs:
            if not self.graphs_saved:
                print('Reprocess to save graphs')
                self.process()
            try:
                graph = load_graphs(f'{self.raw_path}/graphs/graphs.' + str(idx) + '.bin')[0][0]
            except:
                graph = self.construct_graph(self.cells[idx])
        else:
            graph = self.construct_graph(self.cells[idx])
        label = self.label[idx]
        return graph, label

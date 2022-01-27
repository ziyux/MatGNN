import torch
from pymatgen.ext.matproj import MPRester
import pandas as pd
from tqdm import tqdm
import os
from dgl.data.utils import save_info, load_info

from dataset_utils.matdataset import MatDataset
from graph import GraphConstructor


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
                 node_fea_sel=None,
                 edge_fea_sel=None,
                 **kwargs):
        self.gc = GraphConstructor(connect_method='PBC' if connect_method is None else connect_method,
                                   cutoff=cutoff,
                                   **kwargs)
        self.save_graphs = save_graphs
        self.mpr = MPRester(api_key=api_key)
        self.criteria = criteria if criteria is not None \
            else {"elements": {"$in": ["Li", "Na", "K"], "$all": ["O"]}, "nelements": 2}
        self.properties = ['task_id',
                           'pretty_formula', 'nsites', 'nelements',
                           'band_gap', 'formation_energy_per_atom', 'energy_per_atom', 'e_above_hull']
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
        if not self.save_graphs:
            self.cells = self.data_saved['cells']

    def download(self):
        # Query the Materials Project Database
        # criteria = {'nsites':{'$gte':2,'$lt':20}}

        if type(self.criteria) is list:
            start = self.read_temp(f'{self.raw_path}/temp', '.csv')
            step = 1000
            j = 0
            res = []
            ran = tqdm(range(start, len(self.criteria)), desc='Query information') \
                if self.verbose else range(len(self.criteria))
            for i in ran:
                info = pd.DataFrame(self.query(criteria=self.criteria[i], properties=self.properties)
                                    , columns=self.properties)
                res.append(info)
                if i % step == 0 and i > 0:
                    j = int(i / step)
                    lookup = pd.concat(res)
                    lookup.to_csv(f'{self.raw_path}/temp/{self.download_name[:-4]}' + '.' + str(j) + '.csv')
                    res = []
            if res:
                lookup = pd.concat(res)
                lookup.to_csv(f'{self.raw_path}/temp/{self.download_name[:-4]}' + '.' + str(j + 1) + '.csv')
            lookup = []
            idx = [int(file.split('.')[-2]) for file in os.listdir(f'{self.raw_path}/temp')]
            file_lists = [file for _, file in sorted(zip(idx, os.listdir(f'{self.raw_path}/temp')))]
            for file in file_lists:
                if file.endswith('.csv'):
                    lookup.append(pd.read_csv(f'{self.raw_path}/temp/{file}'))
            lookup = pd.concat(lookup)
        else:
            res = self.query(criteria=self.criteria, properties=self.properties)
            # convert to dataframe
            lookup = pd.DataFrame(res, columns=self.properties)
        lookup.to_csv(f'{self.save_path}/{self.download_name}')
        self.clean_temp(f'{self.raw_path}/temp')

    def process(self):
        if self.has_cache() and 'cells' in self.data_saved:
            cells = self.data_saved['cells']
        else:
            lookup = pd.read_csv(f'{self.save_path}/{self.download_name}')
            # download structures

            start = self.read_temp(f'{self.raw_path}/temp', '.pkl')
            step = 1000
            j = 0
            cells = []
            label_dict = {key: [] for key in self.properties}
            if self.verbose:
                print('Total structures: ', lookup.shape[0])
                rows = tqdm(lookup.itertuples(), desc='Get structure')
            else:
                rows = lookup.itertuples()

            for i, row in enumerate(rows):
                if i < start:
                    continue
                cell = self.get_structure_by_material_id(row.task_id, conventional_unit_cell=True)
                cells.append(cell)
                for label_name in self.properties:
                    if label_name == 'task_id' or label_name == 'pretty_formula':
                        label_dict[label_name].append(getattr(row, label_name))
                    else:
                        label = getattr(row, label_name)
                        label_dict[label_name].append(label)

                if i % step == 0 and i > 0:
                    j = int(i / step)
                    temp_data = {'cells': cells, 'label_dict': label_dict}
                    save_info(f'{self.raw_path}/temp/temp_data' + '.' + str(j) + '.pkl', temp_data)
                    cells = []
                    label_dict = {key: [] for key in self.properties}
            if cells:
                temp_data = {'cells': cells, 'label_dict': label_dict}
                save_info(f'{self.raw_path}/temp/temp_data' + '.' + str(j + 1) + '.pkl', temp_data)
            cells = []
            label_dict = {key: [] for key in self.properties}
            idx = [int(file.split('.')[-2]) for file in os.listdir(f'{self.raw_path}/temp')]
            file_lists = [file for _, file in sorted(zip(idx, os.listdir(f'{self.raw_path}/temp')))]
            for file in file_lists:
                if file.endswith('.pkl'):
                    cells += load_info(f'{self.raw_path}/temp/{file}')['cells']
                    for label_name in self.properties:
                        new_label = load_info(f'{self.raw_path}/temp/{file}')['label_dict'][label_name]
                        label_dict[label_name] += new_label

            # save processed data
            for label_name in self.properties:
                if label_name != 'task_id' and label_name != 'pretty_formula':
                    label_dict[label_name] = torch.tensor(label_dict[label_name], dtype=torch.float32)
            self.label_dict = label_dict
        if self.save_graphs:
            self.graphs_saved = []
            cells = tqdm(cells, desc='Construct graph') if self.verbose else cells
            for cell in cells:
                self.graphs_saved.append(self.construct_graph(cell))
        else:
            self.data_saved = {'cells': cells}
        # self.clean_temp(f'{self.raw_path}/temp')

    def construct_graph(self, cell):
        if self.gc.connect_method == 'PBC':
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
        # get one example by index\
        if self.save_graphs:
            raise Exception('Reprocess to save graphs.')
        label = self.label[idx]
        graph = self.gc.connect_graph(self.cells[idx])
        return graph, label

    def query(self, criteria, properties, i=0):
        with self.mpr as mpr:
            try:
                return mpr.query(criteria=criteria, properties=properties)
            except:
                if i < 10:
                    i += 1
                    return self.query(criteria=criteria, properties=properties, i=i)
                else:
                    raise Exception('MPRester query error.')

    def get_structure_by_material_id(self, task_id, conventional_unit_cell=True, i=0):
        with self.mpr as mpr:
            try:
                return mpr.get_structure_by_material_id(task_id, conventional_unit_cell=conventional_unit_cell)
            except:
                if i < 10:
                    i += 1
                    return self.get_structure_by_material_id(task_id, conventional_unit_cell, i=i)
                else:
                    raise Exception('MPRester get structure error.')

    def clean_temp(self, path):
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(f'{path}/{file}')
            os.rmdir(path)

    def read_temp(self, path, ext):
        if not os.path.exists(path):
            os.mkdir(path)
        s = []
        step = 10
        for file in os.listdir(path):
            if file.endswith(ext):
                s.append(int(file.split('.')[-2]))
        if s:
            s = len(s) * step + 1 if len(s) == max(s) else 0
        else:
            s = 0
        return s

import os
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from pymatgen.ext.matproj import MPRester
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
        self.criteria = criteria if criteria is not None \
            else {"elements": {"$in": ["Li", "Na", "K"], "$all": ["O"]}, "nelements": 2}
        self.properties = properties if properties is not None \
            else ['task_id', 'pretty_formula', 'nsites', 'nelements',
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
        if 'cells' in self.data_saved:
            self.cells = self.data_saved['cells']

    def download(self):
        # Query the Materials Project Database
        # criteria = {'nsites':{'$gte':2,'$lt':20}}

        if type(self.criteria) is list:
            step = self.step
            start = self.read_temp(f'{self.raw_path}/temp', '.csv', step)
            runs = []
            runs_num = int((len(self.criteria) - start) / step) + \
                       int((len(self.criteria) - start) % step != 0) if (len(self.criteria) - start) >= 0 else 0
            for i in range(runs_num):
                if i < runs_num - 1:
                    runs.append(range(start + i * step, start + (i + 1) * step))
                else:
                    runs.append(range(start + i * step, len(self.criteria)))

            # multiprocessing query
            pool = Pool()
            for i in range(int(start/step), int(start/step + runs_num)):
                pool.apply_async(self.sub_download, args=(runs[i], i))
            pool.close()
            pool.join()

            lookup = []
            idx = [int(file.split('.')[-2]) for file in os.listdir(f'{self.raw_path}/temp')]
            file_lists = [file for _, file in sorted(zip(idx, os.listdir(f'{self.raw_path}/temp')))]
            for file in file_lists:
                if file.endswith('.csv'):
                    lookup.append(pd.read_csv(f'{self.raw_path}/temp/{file}'))
            lookup = pd.concat(lookup)
        else:
            info = self.query(criteria=self.criteria, properties=self.properties)
            lookup = pd.DataFrame(info, columns=self.properties)
        lookup.to_csv(f'{self.save_path}/{self.download_name}')
        self.clean_temp(f'{self.raw_path}/temp')

    def sub_download(self, run, i):
        info = []
        run = tqdm(run, desc='Query Information ' + str(i)) if self.verbose else run
        for j in run:
            info.append(pd.DataFrame(self.query(criteria=self.criteria[j], properties=self.properties)
                                     , columns=self.properties))
        lookup = pd.concat(info)
        lookup.to_csv(f'{self.raw_path}/temp/{self.download_name[:-4]}' + '.' + str(i) + '.csv')

    def process(self):
        if self.has_cache() and 'cells' in self.data_saved:
            cells = self.data_saved['cells']
        else:
            lookup = pd.read_csv(f'{self.save_path}/{self.download_name}')
            step = self.step
            start = self.read_temp(f'{self.raw_path}/temp', '.pkl', step)
            runs = []
            runs_num = int((lookup.shape[0] - start) / step) + int((lookup.shape[0] - start) % step != 0) \
                if (lookup.shape[0] - start) >= 0 else 0
            for i in range(runs_num):
                if i < runs_num - 1:
                    runs.append((start + i * step, start + (i + 1) * step))
                else:
                    runs.append((start + i * step, lookup.shape[0]))

            # multiprocessing download structures
            pool = Pool()
            for i in range(int(start/step), int(start/step + runs_num)):
                pool.apply_async(self.sub_process, args=(lookup, runs[i], i))
            pool.close()
            pool.join()

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

        self.data_saved = {'cells': cells}

        if self.save_graphs:

            # cells = tqdm(cells, desc='Construct graph', total=len(cells)) if self.verbose else cells
            # for cell in cells:
            #     self.graphs_saved.append(self.construct_graph(cell))
            step = 1
            # start = self.read_temp(f'{self.raw_path}/info', '.pkl', step)
            start = self.read_temp(f'{self.raw_path}/graphs', '.bin', step)
            runs = []
            runs_num = int((len(cells) - start) / step) + int((len(cells) - start) % step != 0) \
                if len(cells) - start >= 0 else 0
            for i in range(runs_num):
                if i < runs_num - 1:
                    runs.append((start + i * step, start + (i + 1) * step))
                else:
                    runs.append((start + i * step, len(cells)))
            for i in tqdm(range(int(start/step), int(start/step + runs_num)), desc='Construct graphs'):
                self.sub_construct_graph(cells, runs[i], i)

            idx = [int(file.split('.')[-2]) for file in os.listdir(f'{self.raw_path}/graphs')]
            file_lists = [file for _, file in sorted(zip(idx, os.listdir(f'{self.raw_path}/graphs')))]
            for file in tqdm(file_lists, desc='Save graphs'):
                if file.endswith('.bin'):
                    self.graphs_saved += load_graphs(f'{self.raw_path}/graphs/{file}')[0]

            # self.graphs_saved = [[] for _ in range(runs_num)]
            # pool = Pool()
            # for i in range(runs_num):
            #     pool.apply_async(self.sub_construct_graph, args=(cells, runs[i], i))
            # pool.close()
            # pool.join()
            # graphs_saved = self.graphs_saved[0]
            # for i in range(1, runs_num):
            #     graphs_saved += self.graphs_saved[i]
            # self.graphs_saved = graphs_saved
            # print(graphs_saved)

        # self.clean_temp(f'{self.raw_path}/temp')

    def sub_process(self, lookup, run, i):
        start, end = run
        cells = []
        label_dict = {key: [] for key in self.properties}
        rows = lookup.itertuples()
        pbar = tqdm(desc='Get structure ' + str(i), total=end - start) if self.verbose else None
        for j, row in enumerate(rows):
            if j < start:
                continue
            elif j >= end:
                break
            cell = self.get_structure_by_material_id(row.task_id, conventional_unit_cell=True)
            cells.append(cell)
            for label_name in self.properties:
                if label_name == 'task_id' or label_name == 'pretty_formula':
                    label_dict[label_name].append(getattr(row, label_name))
                else:
                    label = getattr(row, label_name)
                    label_dict[label_name].append(label)
            if self.verbose:
                pbar.update(1)
        temp_data = {'cells': cells, 'label_dict': label_dict}
        save_info(f'{self.raw_path}/temp/temp_data' + '.' + str(i) + '.pkl', temp_data)

    def sub_construct_graph(self, cells, run, idx):
        cells_sub = cells[run[0]:run[1]]
        # cells_sub = tqdm(cells_sub, desc='Construct graph ' + str(i), total=len(cells_sub))\
        #     if self.verbose else cells_sub
        graphs_saved = []
        fea_saved = []
        for cell in cells_sub:
            graph = self.construct_graph(cell)
            graphs_saved.append(graph)

            # pos = graph.ndata['coords']
            # edge_index = graph.edges()
            # num_nodes = graph.num_nodes()
            # dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)
            # rbf, sbf, tbf = SphericalFeatures(num_spherical=3, num_radial=6, cutoff=5)(dist, angle, torsion, idx_kj)
            # fea_saved.append({'idx_kj': idx_kj, 'idx_ji': idx_ji, 'rbf': rbf, 'sbf': sbf, 'tbf': tbf})

        # save_info(f'{self.raw_path}/info/info.' + str(idx) + '.pkl', {'fea': fea_saved})
        save_graphs(f'{self.raw_path}/graphs/graphs.' + str(idx) + '.bin', graphs_saved)



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
        # get one example by index\
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

    def query(self, criteria, properties, i=0):
        with MPRester(api_key=self.api_key) as mpr:
            try:
                info = mpr.query(criteria=criteria, properties=properties)
                if not info:
                    print('Nan: ', criteria, info)
                return info
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                if i < 10:
                    i += 1
                    return self.query(criteria=criteria, properties=properties, i=i)
                else:
                    raise Exception('MPRester query error.')

    def get_structure_by_material_id(self, task_id, conventional_unit_cell=True, i=0):
        with MPRester(api_key=self.api_key) as mpr:
            try:
                return mpr.get_structure_by_material_id(task_id, conventional_unit_cell=conventional_unit_cell)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
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

    def read_temp(self, path, ext, step=None):
        step = self.step if step is None else step
        if not os.path.exists(path):
            os.mkdir(path)
        s = []
        for file in os.listdir(path):
            if file.endswith(ext):
                s.append(int(file.split('.')[-2]))
        if s:
            s = len(s) * step if (len(s) - 1) == max(s) else 0
        else:
            s = 0
        return s

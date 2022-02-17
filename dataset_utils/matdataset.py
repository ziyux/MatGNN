import os
import torch

from dgl.data import DGLDataset
from dgl.data.utils import download, _get_dgl_url, save_info, load_info, makedirs
from dgl import save_graphs, load_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MatDataset(DGLDataset):
    def __init__(self,
                 dataset_name: str,
                 label_name: str,
                 url: str = None,
                 raw_dir: str = None,
                 save_dir: str = None,
                 save_name: str = '',
                 force_reload: bool = False,
                 verbose: bool = False,
                 download_name: str = None):
        """
        The basic DGL dataset for creating graph datasets.
        This class defines a basic template class for DGL Dataset.
        The following steps will be executed automatically:

            1. Check whether there is a dataset cache on disk
                (already processed and stored on the disk) if
                force_reload is False. If Cached, skip 2,3 and 4.
            2. Call ``download()`` to download the data.
            3. Call ``process()`` to process the data.
            4. Cached processed data in ``process()``.

        Usage:
            Call ``__len__`` to get the number of samples in the dataset.

            Call ``__getitem__(idx)`` to get the graph and label for the given idx.

        :param dataset_name: str, name for the dataset used.
        :param label_name:  str, name for the label used in training
        :param url: str, url of the downloaded dataset. Default None. If not None, files wil be automatically downloaded to self.save_path
        :param raw_dir: str, root directory to read processed data. If None, default at dataset_util/dataset_cache/dataset_name.
        :param save_dir: str, root directory to store processed data. If None, default at dataset_util/dataset_cache/dataset_name.
        :param save_name: str, sub-directory name to read/store data under raw_dir/save_dir. Default to have no sub-directory.
        :param force_reload: bool, False to load stored processed data each time and True to re-process data each time.
        :param verbose: bool, False to not print info and True to print info.
        :param download_name: str, name for files downloaded in self.download().
        """
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_cache', dataset_name) \
            if raw_dir is None else raw_dir
        save_dir = raw_dir if save_dir is None else save_dir
        self.label_name = label_name
        self.download_name = download_name
        self.graphs_saved = []
        self.data_saved = {}
        self.label_dict = {}
        self.label = None
        super(MatDataset, self).__init__(name=save_name,
                                         url=url,
                                         raw_dir=raw_dir,
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def get_url(self, name):
        url = _get_dgl_url(name)
        return url

    def _download(self):
        if self.download_name is None:
            raise ValueError('Downloaded file name needs to be specified.')
        else:
            if os.path.exists(os.path.join(self.raw_path, self.download_name)):
                return

        makedirs(self.save_path)

        if self.url is not None:
            file_path = os.path.join(self.save_path, self.download_name)
            if not os.path.exists(file_path):
                download(self.url, path=file_path)

        self.download()

    def download(self):
        # download raw data to local disk. Don't need to be rewritten if url is given
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        # use self.graphs to store graphs; self.label_dict to store lable and self.data to store extra info
        raise NotImplementedError

    def __getitem__(self, idx):
        # get one example by index
        self.label = self.label_dict[self.label_name]
        if self.graphs_saved and self.label is not None:
            return self.graphs_saved[idx].to(device), (self.label[idx].to(device), idx)
        else:
            # try:
            graph, label = self.getitem(idx)
            # except ValueError('Fail to get items. Try to process data again.'):
            #     try:
            #         self._force_reload = True
            #         self._load()
            #         graph, label = self.__getitem__(idx)
            #     except:
            #         raise Exception('Processed items are incomplete. Please check getitem() and process() functions.')
            return graph.to(device), (label.to(device), idx)

    def getitem(self, idx):
        raise NotImplementedError('getitem() function needs to be specified.')

    def __len__(self):
        # number of data examples
        return len(self.label_dict[self.label_name])

    def save(self):
        # save processed data to directory `self.save_path`
        data_file = os.path.join(self.save_path, self._name + '_data.pkl')
        graphs_file = os.path.join(self.save_path, self._name + '_graphs.bin')
        if self.label_dict is {}:
            raise Exception('self.lable_dict is empty.')
        self.data_saved['label_dict'] = self.label_dict
        if not self.has_cache() or self._force_reload:
            if self.graphs_saved:
                save_graphs(graphs_file, self.graphs_saved)
            save_info(data_file, self.data_saved)

    def load(self):
        # load processed data from directory `self.raw_dir`
        data_file = os.path.join(self.raw_path, self._name + '_data.pkl')
        graphs_file = os.path.join(self.raw_path, self._name + '_graphs.bin')
        if os.path.exists(data_file):
            self.data_saved = load_info(data_file)
            self.label_dict = self.data_saved['label_dict']
        if os.path.exists(graphs_file):
            self.graphs_saved, labels = load_graphs(graphs_file)
        if self.graphs_saved == [] and self.data_saved == {}:
            raise Exception('Failed to load cache to the dataset')

    def has_cache(self):
        data_file = os.path.join(self.raw_path, self._name + '_data.pkl')
        return os.path.exists(data_file)

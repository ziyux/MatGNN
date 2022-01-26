from dataset_utils.matdataset import MatDataset
from graph import GraphConstructor


class Template(MatDataset):
    def __init__(self,
                 dataset_name: str,
                 label_name: str,
                 connect_method: str = None,
                 cutoff: int or float = 5,
                 url: str = None,
                 raw_dir: str = None,
                 save_dir: str = None,
                 save_name: str = '',
                 force_reload: bool = False,
                 verbose: bool = False,
                 download_name: str = None,
                 **kwargs):
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
            Call ``__len__()`` to get the number of samples in the dataset.

            Call ``__getitem__(idx)`` to get the graph and label for the given idx.

        :param dataset_name: str, name for the dataset used.
        :param label_name:  str, name for the label used in training
        :param connect_method: str, method to connect nodes within graphs
        :param cutoff: float, cutoff to construct the graph.
        :param url: str, url of the downloaded dataset. Default None. If not None, files wil be automatically downloaded to self.save_path
        :param raw_dir: str, root directory to read processed data. If None, default at dataset_util/dataset_cache/dataset_name.
        :param save_dir: str, root directory to store processed data. If None, default at dataset_util/dataset_cache/dataset_name.
        :param save_name: str, sub-directory name to read/store data under raw_dir/save_dir. Default to have no sub-directory.
        :param force_reload: bool, False to load stored processed data each time and True to re-process data each time.
        :param verbose: bool, False to not print info and True to print info.
        :param download_name: str, name for files downloaded in self.download().
        :param kwargs: other args to construct graphs.
        """
        self.gc = GraphConstructor(  # Specify default connect method here
            connect_method='CWC' if connect_method is None else connect_method,
            cutoff=cutoff,
            # Add other args if needed
            **kwargs)

        super(Template, self).__init__(dataset_name=dataset_name,
                                       label_name=label_name,
                                       url=url,
                                       raw_dir=raw_dir,
                                       save_dir=save_dir,
                                       save_name=save_name,
                                       force_reload=force_reload,
                                       verbose=verbose,
                                       download_name=download_name)

    def download(self):
        # Download raw data to local disk. Don't need to be rewritten if url is given.
        pass

    def process(self):
        # Required to process raw data to labels and other data.
        # Optionally choose to generate graphs here if wish to preprocess all graphs before training.
        file_path = f'{self.save_path}/{self.download_name}'  # read downloaded data

        # Use self.graphs to store graphs; self.label_dict to store labels and self.data to store extra info
        # All three variables will be automatically cached and automatically read to skip re-processing.
        self.label_dict = {'label_name': []}  # required to store labels
        self.graphs_saved = []  # optionally, leave blank list if not to store graphs.
        self.data_saved = {}  # optionally, leave blank dict if not to store extra info

    def getitem(self, idx):
        # # Required if not save graphs in self.graphs_saved in self.process().
        # # Choose to generate graphs here if wish to process each graph during training.
        # label = self.label[idx]  # self.label = self.label_dict[self.label_name] is automatically loaded.
        # graph = self.gc.construct_graph(self.data_saved[idx])
        # return graph, label
        raise NotImplementedError('getitem() function needs to be specified.')

    def construct_graph(self, idx_data):
        self.gc.connect_graph(idx_data)
        node_fea = {}
        edge_fea = {}
        self.gc.feature_assignment(node_fea, edge_fea)


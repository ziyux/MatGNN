from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader


class Dataloader(object):

    def __init__(self,
                 dataset_name: str,
                 label_name: str,
                 connect_method: str = None,
                 cutoff: int or float = 5,
                 save_graphs: bool = True,
                 url: str = None,
                 raw_dir: str = None,
                 save_name: str = '',
                 force_reload: bool = False,
                 verbose: bool = True,
                 **kwargs):
        """
        The class to load dataset and get train test split loaders.

        Usage:
            Call ``get_split_loaders()`` to get split loaders.

        :param dataset_name: str, name for the dataset used.
        :param label_name: str, name for the label used in training.
        :param connect_method: str, method to connect nodes within graphs.
        :param cutoff: float, cutoff to construct the graph.
        :param save_graphs: str, root directory to store processed data. If None, default at dataset_util/dataset_cache/dataset_name.
        :param url: str, url of the downloaded dataset. Default None. If not None, files wil be automatically downloaded to self.save_path.
        :param raw_dir: str, root directory to read processed data. If None, default at dataset_util/dataset_cache/dataset_name.
        :param save_name: str, sub-directory name to read/store data under raw_dir/save_dir. Default to have no sub-directory.
        :param force_reload: bool, False to load stored processed data each time and True to re-process data each time.
        :param verbose: bool, False to not print info and True to print info.
        :param kwargs: other args to construct graphs.
        """
        self.dataset = self.get_dataset(dataset_name)(dataset_name=dataset_name, label_name=label_name,
                                                      connect_method=connect_method, cutoff=cutoff,
                                                      save_graphs=save_graphs, url=url, raw_dir=raw_dir,
                                                      save_name=save_name, force_reload=force_reload,
                                                      verbose=verbose, **kwargs)

    def get_split_loaders(self, train_rate=0.8, valid_rate=0.1, test_rate=0.1, batch_size=1, random_state=25,
                          shuffle=False, drop_last=False, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        train_set, valid_set, test_set = self.split_dataset(train_rate=train_rate, valid_rate=valid_rate, test_rate=test_rate,
                                                            random_state=random_state, dataset=dataset)
        train_loader, valid_loader, test_loader = map(
            lambda splitdataset: self.get_loader(batch_size, drop_last, shuffle, splitdataset),
            [train_set, valid_set, test_set])
        return train_loader, valid_loader, test_loader

    def split_dataset(self, train_rate=0.8, valid_rate=0.1, test_rate=0.1, shuffle=False, random_state=25, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        train_set, valid_set, test_set = split_dataset(dataset, [train_rate, valid_rate, test_rate], shuffle=shuffle,
                                                     random_state=random_state)
        return train_set, valid_set, test_set

    def get_loader(self, batch_size=1, drop_last=False, shuffle=False, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        return GraphDataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

    def get_dataset(self, dataset_name):
        if dataset_name == 'QM9':
            from dataset_utils.QM9 import QM9
            return QM9
        elif dataset_name == 'MaterialsProject':
            from dataset_utils.MaterialsProject import MaterialsProject
            return MaterialsProject
        else:
            raise Warning('No corresponded datasetname found.')

    # def split_dataset(self, train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=25, dataset=None):
    #     dataset = self.dataset if dataset is None else dataset
    #     train_mask, val_mask, test_mask = self.train_test_split(train_rate, val_rate, test_rate, random_state)
    #     # train_set, val_set, test_set = map(lambda set_name, mask: DatasetPartition(set_name, dataset, mask),
    #     #                                    ['train_set', 'val_set', 'test_set'], [train_mask, val_mask, test_mask])
    #     train_set = DatasetPartition('train_set', dataset, train_mask)
    #     val_set = DatasetPartition('val_set', dataset, val_mask)
    #     test_set = DatasetPartition('test_set', dataset, test_mask)
    #     return train_set, val_set, test_set, train_mask, val_mask, test_mask

    # def train_test_split(self, train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=25):
    #     samples = len(self.dataset)
    #     train_size = int(train_rate * samples)
    #     val_size = int(val_rate * samples)
    #     test_size = int(test_rate * samples)
    #     unused_size = samples - train_size - val_size - test_size
    #     train_mask, val_mask, test_mask, unused_mask = torch.utils.data.random_split(
    #         list(range(samples)), [train_size, val_size, test_size, unused_size],
    #         torch.Generator().manual_seed(random_state))
    #     return train_mask, val_mask, test_mask
    #
    # def get_split_loaders(self, train_mask, val_mask, test_mask, batch_size=1, drop_last=False, shuffle=False):
    #     train_set = DatasetPartition('train_set', self.dataset, train_mask)
    #     self.train_loader = GraphDataLoader(train_set, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    #     val_set = DatasetPartition('val_set', self.dataset, val_mask)
    #     self.val_loader = GraphDataLoader(val_set, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    #     test_set = DatasetPartition('test_set', self.dataset, test_mask)
    #     self.test_loader = GraphDataLoader(test_set, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    #     return self.train_loader, self.val_loader, self.test_loader, train_set, val_set, test_set


# from dgl.data import DGLDataset

# class DatasetPartition(DGLDataset):
#
#     def __init__(self,
#                  name,
#                  dataset,
#                  mask,
#                  url=None,
#                  raw_dir=None,
#                  save_dir=None,
#                  force_reload=False,
#                  verbose=False):
#         self.dataset = dataset
#         self.mask = mask
#         super(DatasetPartition, self).__init__(name=name,
#                                                url=url,
#                                                raw_dir=raw_dir,
#                                                save_dir=save_dir,
#                                                force_reload=force_reload,
#                                                verbose=verbose)
#
#     def process(self):
#         # process raw data to graphs, labels, splitting masks
#         pass
#
#     def __getitem__(self, idx):
#         # get one example by index
#         graph = self.dataset[self.mask[idx]][0]
#         label = self.dataset[self.mask[idx]][1]
#         return graph, label
#
#     def __len__(self):
#         # number of data examples
#         return len(self.mask)

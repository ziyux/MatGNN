
def dataloader(dataset_name: str,
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
    return get_dataset(dataset_name)(dataset_name=dataset_name, label_name=label_name,
                                                  connect_method=connect_method, cutoff=cutoff,
                                                  save_graphs=save_graphs, url=url, raw_dir=raw_dir,
                                                  save_name=save_name, force_reload=force_reload,
                                                  verbose=verbose, **kwargs)

def get_dataset(dataset_name):
    if dataset_name == 'QM9':
        from dataset_utils.QM9 import QM9
        return QM9
    elif dataset_name == 'MaterialsProject':
        from dataset_utils.MaterialsProject import MaterialsProject
        return MaterialsProject
    elif dataset_name == 'MaterialsProject_raw':
        from dataset_utils.MaterialsProject_raw import MaterialsProject
        return MaterialsProject
    else:
        raise Warning('No corresponded datasetname found.')

import os
import random
import time
import pickle
import torch as th
import torch.optim as opt
from torch.nn.functional import mse_loss, l1_loss

os.environ['DGLBACKEND'] = 'pytorch'
from dgl import backend as F
from dgl.dataloading import GraphDataLoader

from graph import GraphConstructor
from datasets import get_dataset, DatasetPartition


def rename_dir(directory, new_directory, i=1):
    try:
        new_directory = ''.join([new_directory[:-(2 + len(str(i - 1)))], '(', str(i), ')'])
        os.rename(directory, new_directory)
    except OSError:
        i += 1
        new_directory = rename_dir(directory, new_directory, i)
    return new_directory


def make_dir(directory, resume):
    try:
        os.mkdir(directory)
    except OSError:
        if not resume:
            new_directory = rename_dir(directory, ''.join([directory, '(1)']))
            print('Warning: Moving existing directory \"' + directory + '\" to \"' + new_directory + '\".')
            make_dir(directory, resume)
    return directory


def save_pickle(filename, obj):
    with open(filename, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def printf(string, filename='log.txt', is_new_file=False):
    mode = 'w+' if is_new_file else 'a+'
    with open(filename, mode) as f:
        print(string)
        print(string, file=f)
    f.close()


class MatGNN(object):
    def __init__(self,
                 dataset_name,
                 model,
                 label_sel,
                 node_fea_sel=None,
                 edge_fea_sel=None,
                 connect_method='NKA',
                 cutoff=float('inf'),
                 k=None,
                 train_rate=0.9,
                 val_rate=0.05,
                 test_rate=0.05,
                 batch_size=512,
                 project_name=None,
                 resume=False,
                 resume_model='checkpoint.tar',
                 use_cuda=False):
        self.root_dir = os.getcwd()
        project_name = dataset_name if project_name is None else project_name
        self.result_dir = make_dir(os.path.join(os.getcwd(), project_name + '_Results'), resume)
        self.local_time = lambda: time.asctime(time.localtime(time.time()))
        self.use_cuda = use_cuda

        self.node_fea_sel = node_fea_sel
        self.edge_fea_sel = edge_fea_sel
        self.graph_constructor = GraphConstructor(connect_method, cutoff, k, gaussian_step=0.5)
        self.dataset = get_dataset(dataset_name, self.graph_constructor, label_sel, node_fea_sel, edge_fea_sel,
                                   root_dir=self.root_dir)

        self.model = self.construct_model(model, 16, 11)
        if self.use_cuda:
            self.model = self.model.cuda()
        self.optimizer = th.optim.Adam(self.model.parameters())
        self.criterion = mse_loss
        self.batch_size = batch_size
        self.start_epoch = 0
        self.valid_loss = float('inf')
        if resume is True:
            self.load_model(resume_model)
        else:
            self.loss_list = []
            self.train_mask, self.val_mask, self.test_mask = self.train_test_split(train_rate, val_rate, test_rate)
        self.get_loader()

    def train(self, MAX_ITER=2000, model_name=None):
        if model_name is not None:
            self.load_model(model_name)
            self.get_loader()
        for epoch in range(self.start_epoch, MAX_ITER):
            self.loss_list.append([])
            self.model.train()
            for idx, (batched_graph, labels) in enumerate(self.train_loader):
                if self.use_cuda:
                    batched_graph, labels = batched_graph.cuda(), labels.cuda()
                # feat = th.cat([batched_graph.ndata[key] for key in self.node_fea_sel], 1)
                # efeat = th.cat([batched_graph.edata[key] for key in self.edge_fea_sel], 1)
                feat = batched_graph.ndata['atomic_number']
                efeat = batched_graph.edata['GD']
                logits = self.model(batched_graph, feat, efeat)
                loss = self.criterion(logits, labels)
                self.loss_list[-1].append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                printf('[' + str(self.local_time()) + ']' +
                       '    Epoch: ' + str(epoch + 1) + '/' + str(MAX_ITER) +
                       '    Step: ' + str(idx + 1) + '/' + str(len(self.train_loader)) +
                       '    Loss: ' + str(loss.item()),
                       filename=os.path.join(self.result_dir, 'log.txt'))

            valid_loss = self.evaluate(loader=self.val_loader)
            if valid_loss < self.valid_loss:
                self.valid_loss = valid_loss
                self.save_model('bestmodel.tar', epoch, self.valid_loss, self.loss_list)
            self.save_model('checkpoint.tar', epoch, self.valid_loss, self.loss_list)

            printf('[' + str(self.local_time()) + ']' +
                   '    Epoch: ' + str(epoch + 1) + '/' + str(MAX_ITER) +
                   '    Valid_Loss: ' + str(float(self.valid_loss)) + '\n',
                   filename=os.path.join(self.result_dir, 'log.txt'))

    def evaluate(self, loader=None, criterion=l1_loss):
        self.model.eval()
        loader = self.val_loader if loader is None else loader
        criterion = self.criterion if criterion is None else criterion
        valid_loss = 0
        for idx, (batched_graph, labels) in enumerate(loader):
            with th.no_grad():
                if self.use_cuda:
                    batched_graph, labels = batched_graph.cuda(), labels.cuda()
                # feat = th.cat([batched_graph.ndata[key] for key in self.node_fea_sel], 1)
                # efeat = th.cat([batched_graph.edata[key] for key in self.edge_fea_sel], 1)
                feat = batched_graph.ndata['atomic_number']
                efeat = batched_graph.edata['GD']
                logits = self.model(batched_graph, feat, efeat)
                print(th.cat((logits, labels), 1))
                loss = criterion(logits, labels)
                print(float(loss))
                valid_loss = valid_loss + ((1 / (idx + 1)) * (loss.data - valid_loss))
        return float(valid_loss)

    def score(self, model_name=None, loader=None):
        if model_name is not None:
            self.load_model(model_name)
            self.get_loader()
        loader = self.test_loader if loader is None else loader
        return self.evaluate(loader)

    def predict(self, model_name=None, loader=None):
        if model_name is not None:
            self.load_model(model_name)
            self.get_loader()
        loader = self.test_loader if loader is None else loader
        predicts = []
        self.model.eval()
        for idx, (batched_graph, labels) in enumerate(loader):
            with th.no_grad():
                if self.use_cuda:
                    batched_graph, labels = batched_graph.cuda(), labels.cuda()
                # feat = th.cat([batched_graph.ndata[key] for key in self.node_fea_sel], 1)
                # efeat = th.cat([batched_graph.edata[key] for key in self.edge_fea_sel], 1)
                feat = batched_graph.ndata['atomic_number']
                efeat = batched_graph.edata['GD']
                predicts.append(self.model(batched_graph, feat, efeat))
        return predicts

    def train_test_split(self, train_rate, val_rate, test_rate):
        random.seed(0)
        samples = len(self.dataset)
        train_size = int(train_rate * samples)
        val_size = int(val_rate * samples)
        test_size = int(test_rate * samples)
        unused_size = samples - train_size - val_size - test_size
        train_mask, val_mask, test_mask, unused_mask = th.utils.data.random_split(
            list(range(samples)), [train_size, val_size, test_size, unused_size])
        return train_mask, val_mask, test_mask

    def get_loader(self, drop_last=False, shuffle=False):
        train_set = DatasetPartition('train_set', self.dataset, self.train_mask)
        self.train_loader = GraphDataLoader(train_set, batch_size=self.batch_size, drop_last=drop_last, shuffle=shuffle)
        val_set = DatasetPartition('val_set', self.dataset, self.val_mask)
        self.val_loader = GraphDataLoader(val_set, batch_size=self.batch_size, drop_last=drop_last, shuffle=shuffle)
        test_set = DatasetPartition('test_set', self.dataset, self.test_mask)
        self.test_loader = GraphDataLoader(test_set, batch_size=self.batch_size, drop_last=drop_last, shuffle=shuffle)

    def construct_model(self, model, feat, efeat):
        return model(feat, efeat)

    def save_model(self, model_name, epoch, loss, loss_list):
        th.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_list': loss_list,
            'mask': (self.train_mask, self.val_mask, self.test_mask)
        }, os.path.join(self.result_dir, model_name))

    def load_model(self, model_name='checkpoint.tar'):
        checkpoint = th.load(os.path.join(self.result_dir, model_name))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.valid_loss = checkpoint['loss']
        self.loss_list = checkpoint['loss_list']
        self.train_mask, self.val_mask, self.test_mask = checkpoint['mask']

    def print_model(self, model_name='checkpoint.tar', print_out=False):
        checkpoint = th.load(os.path.join(self.result_dir, model_name))
        if print_out:
            print('model_state_dict: ', checkpoint['model_state_dict'])
            print('optimizer_state_dict: ', checkpoint['optimizer_state_dict'])
            print('epoch: ', checkpoint['epoch'])
            print('min valid_loss: ', checkpoint['loss'])
            print('loss_list: ', checkpoint['loss_list'])
            print('train_mask: ', list(checkpoint['mask'][0]), '\nval_mask: ', list(checkpoint['mask'][1]),
                  '\ntest_mask: ', list(checkpoint['mask'][2]))
        return checkpoint

import os
import time
import torch
from torch.nn.functional import mse_loss, l1_loss
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from tqdm import tqdm
import gc

try:
    from torch.utils.tensorboard import SummaryWriter
    use_tensorboard = True
except ImportError:
    use_tensorboard = False

os.environ['DGLBACKEND'] = 'pytorch'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MatGNN(object):
    def __init__(self,
                 project_name: str,
                 model,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 criterion=l1_loss,
                 lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=100, weight_decay=0,
                 resume: bool = False,
                 resume_model: str = 'checkpoint.tar'):
        """
        The class to train a GNN model and save the results.

        Usage:
            Call ``train()`` to train the GNN model

            Call ``score()`` to evaluate a loader

            Call ``predict()`` to predict a target of a given loader

            Call ``save()`` to save the current model

            Call ``load()`` to load a saved model

            Call ``print()`` to print info of a saved model

        :param project_name: str, directory name for saving the results in "Results" folder.
        :param model: nn.Module, GNN module to train.
        :param train_loader: Dataloader, pytorch dataloader for training.
        :param valid_loader: Dataloader, pytorch dataloader for validating.
        :param test_loader: Dataloader, pytorch dataloader for testing.
        :param criterion: pytorch criterion.
        :param resume: bool, True to load a saved model, False to start a new one.
        :param resume_model: str, name of the saved model to load.
        """
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.root_dir = os.getcwd()
        self.make_dir(os.path.join(os.getcwd(), 'Results'), resume=True)
        self.result_dir = self.make_dir(os.path.join(os.getcwd(), 'Results', project_name), resume)
        self.local_time = lambda: time.asctime(time.localtime(time.time()))
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
        self.criterion = criterion
        self.start_epoch = 0
        self.valid_loss = float('inf')
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.result_dir)
        if resume is True:
            self.load_model(resume_model)
        else:
            self.loss_list = {'train':[], 'valid': []}

    def train(self, MAX_ITER=1000, model_name=None, train_loader=None, valid_loader=None):
        train_loader = self.check_input(train_loader, 'train_loader')
        valid_loader = self.check_input(valid_loader, 'valid_loader')
        self.load_model(model_name)
        for epoch in range(self.start_epoch, MAX_ITER):
            loss_list = []
            self.model.train()
            for step, (batched_graph, (label, idx)) in enumerate(tqdm(train_loader, desc='Training')):
                self.optimizer.zero_grad()
                logits = self.model(batched_graph, idx)
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.criterion(logits.flatten(), label)
                loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()
            self.loss_list['train'] = train_loss = sum(loss_list)/(step+1)
            self.loss_list['valid'] = valid_loss = self.score(loader=valid_loader)

            if valid_loss < self.valid_loss:
                self.valid_loss = valid_loss
                self.save_model('bestmodel.tar', epoch, self.valid_loss, self.loss_list)
            self.save_model('checkpoint.tar', epoch, self.valid_loss, self.loss_list)

            if use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/valid', valid_loss, epoch)
            self.printf('[' + str(self.local_time()) + ']' +
                        '    Epoch: ' + str(epoch + 1) + '/' + str(MAX_ITER) +
                        '    Train_Loss: ' + str(float(train_loss)) +
                        '    Valid_Loss: ' + str(float(valid_loss)) +
                        '    Best_Loss: ' + str(float(self.valid_loss)) + '\n',
                        filename=os.path.join(self.result_dir, 'log.txt'))
            self.scheduler.step()

    def score(self, criterion=None, loader=None):
        self.model.eval()
        loader = self.check_input(loader, 'test_loader')
        criterion = self.check_input(criterion, 'criterion')
        valid_loss = []
        for step, (batched_graph, (label, idx)) in enumerate(tqdm(loader, desc='Evaluating')):
            with torch.no_grad():
                logits = self.model(batched_graph, idx)
                loss = criterion(logits.flatten(), label)
                valid_loss.append(loss)
        return sum(valid_loss)/(step + 1)

    def predict(self, model_name=None, loader=None):
        self.load_model(model_name)
        loader = self.check_input(loader, 'test_loader')
        predicts = []
        self.model.eval()
        for step, (batched_graph, (labels, idx)) in enumerate(tqdm(loader, desc='Predicting')):
            with torch.no_grad():
                predicts.append(self.model(batched_graph, idx))
        return predicts

    def save_model(self, model_name, epoch, loss, loss_list):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_list': loss_list,
            # 'mask': (self.train_mask, self.val_mask, self.test_mask)
        }, os.path.join(self.result_dir, model_name))

    def load_model(self, model_name=None):
        if model_name is not None:
            checkpoint = torch.load(os.path.join(self.result_dir, model_name))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.valid_loss = checkpoint['loss']
            self.loss_list = checkpoint['loss_list']

    def print_model(self, model_name='checkpoint.tar', print_out=False):
        checkpoint = torch.load(os.path.join(self.result_dir, model_name))
        if print_out:
            print('model_state_dict: ', checkpoint['model_state_dict'])
            print('optimizer_state_dict: ', checkpoint['optimizer_state_dict'])
            print('epoch: ', checkpoint['epoch'])
            print('min valid_loss: ', checkpoint['loss'])
            print('loss_list: ', checkpoint['loss_list'])
        return checkpoint

    def check_input(self, inp, input_name):
        if inp is None:
            if getattr(self, input_name) is None:
                raise Exception('No ' + input_name + ' has been input.')
            else:
                return getattr(self, input_name)
        else:
            return inp

    def rename_dir(self, directory, new_directory, i=1):
        try:
            new_directory = ''.join([new_directory[:-(2 + len(str(i - 1)))], '(', str(i), ')'])
            os.rename(directory, new_directory)
        except OSError:
            i += 1
            new_directory = self.rename_dir(directory, new_directory, i)
        return new_directory

    def make_dir(self, directory, resume):
        try:
            os.mkdir(directory)
        except OSError:
            if os.path.exists(directory):
                if not resume:
                    os.path.exists(directory)
                    new_directory = self.rename_dir(directory, ''.join([directory, '(1)']))
                    print('Warning: Moving existing directory \"' + directory + '\" to \"' + new_directory + '\".')
                    self.make_dir(directory, resume)
            else:
                raise OSError('Cannot create the result directory.')
        return directory

    def printf(self, string, filename='log.txt', is_new_file=False):
        mode = 'w+' if is_new_file else 'a+'
        with open(filename, mode) as f:
            print(string)
            print(string, file=f)
        f.close()

import torch
from torch import nn
import numpy as np


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def list_as_str_stream(data, extra_toks_to_remove=['[', ']']):
    '''Convert list elements as consecutive elements (like a stream).
        data: str list
        extra_toks_to_remove: any additional character(s) to remove.

        Return:
        string stream of list elements.
    '''
    # print(f"[list_as_str_stream] in:{data[:100]}")
    str_data = []
    for t in data:
        str_data.append(t)
    str_data = str(str_data)
    str_data = str_data.replace("\n", '').replace(' ', '').replace(',', '').replace('.', '').replace("'", '').replace('"', '').replace('[', '').replace(']', '').replace("\n", '')
    if extra_toks_to_remove:
        for tok_remove in extra_toks_to_remove:
            str_data = str_data.replace(tok_remove, '')
    return str_data



def find_conv_kr(hz=100):
    """
        Convolution kernel size is 44ms equivalent samples which consititutes
        average QRS complex length.
    """
    milli = 44
    conv_kr = int(hz / 1000 * milli)
    conv_kr += 1 if conv_kr % 2 == 0 else 0
    return int(conv_kr)
    # return 3


def padding_same(input,  kernel, stride=1, dilation=1):
    """
        Calculates padding for applied dilation.
    """
    return int(0.5 * (stride * (input - 1) - input + kernel + (dilation - 1) * (kernel - 1)))


def find_dilation_factor(channel=1, layer=1, channels_per_layer=2, dilation_limit=10):
    b"s_ij = ((iw + j) mod 10) + 1"
    return ((layer * channels_per_layer + channel) % dilation_limit) + 1


def find_layer_dilations(layer=1, channels_per_layer=2, dilation_limit=10):
    dilations = []
    for ch in range(channels_per_layer):
        dilations.append(
            find_dilation_factor(
                channel=ch+1, layer=layer,
                channels_per_layer=channels_per_layer,
                dilation_limit=dilation_limit))
        # dilations.append(1 if ch == 0 else 2 * ch)
    # log(f'[find_layer_dilations] channels_per_layer: {channels_per_layer}, dilation_calculated: {dilations}')
    return dilations


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def save_model(model, base_path='./', file_name=None, db_name=None, metric=None):
    '''Save a model to file.

        Returns:
            None.
    '''
    if file_name:
        file_path = f'{base_path}/{model.name()}_{file_name}_{metric:.04f}.pt'
    else:
        metric = "nomet" if metric is None else f'_{metric:.04f}'
        db_name = "nodb" if db_name is None else f'_{db_name}'
        filename_postfix = f'{db_name}_{metric}'
        # file_path = '../../../models/' + file_name + '_' + filename_postfix + '.pt'
        file_path_ = f'{file_path}/{model.name()}_{filename_postfix}.pt'
    torch.save(model.state_dict(), file_path_)
    print(f'[save_model] Model saved to {file_path_}')


def load_model(model, file_name, base_path='./', device='cpu'):
    '''Load a saved model.

        Args:
            model: The model for which weights to be loaded from file.
            file_name: Model file name
            base_path: Model location
            device: cpu or gpu

        Returns:
            weighted updated model.
    '''
    file_path = f"{base_path}/{file_name}.pt"
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    # Debug weights
    print(f"[load_model] checkpoint: {checkpoint.keys()}")
    for key in checkpoint.keys():
        print(f"[load_model] {key}: {checkpoint[key].size()}")

    model.load_state_dict(checkpoint)
    model.eval()
    print(f'[load_model] Model loaded from {file_path}')
    return model


class EarlyStopping():
    '''Early stops the training if validation loss doesn't improve after a
        given patience.'''

    def __init__(
            self, patience=7, verbose=False, delta=0,
            path='checkpoint.pt', log=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
            verbose (bool): If True, prints a message for each validation loss
                improvement.
            delta (float): Minimum change in the monitored quantity to qualify
                as an improvement.
            path (str): Path for the checkpoint to be saved to.
            log : log function (TAG, msg).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_extra = None  # Extra best other scores/info
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.log = log

        self.print_(f'[{self.__class__.__name__}] patience:{patience}, delta:{delta}, model-path:{path}')

    def print_(self, msg):
        if self.log is None:
            print(msg)
        else:
            self.log(f"[{EarlyStopping}] {msg}")

    def __call__(self, val_loss, model, extra=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.print_(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.print_(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

import torch.nn as nn
import numpy as np
np.random.seed(42)
import tensorflow as tf
import torch as pt
import torch.nn.functional as F
import torchvision as ptv
import numpy as np

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense

import numpy as np

MAX_INT = np.iinfo(np.int32).max
data_format = 0


class Cnn(pt.nn.Module):
    def __init__(self, input_shape, n_cls, device):
        super(Cnn, self).__init__()
        self.input_shape = input_shape
        self.n_cls = n_cls
        self.device = device

    def forward(self, data):
        return data


class CnnCls22(Cnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls22, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int((self.input_shape+1*2-3)/1 + 1)
        input_size = int((input_size + 1 * 2 - 3) / 1 + 1)
        self.fc1 = nn.Linear(input_size*6, int(input_size*6/2)).to(device)
        self.fc2 = nn.Linear(int(input_size*6/2), self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        out = self.bn2(out)
        return out


class CnnCls21(Cnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls21, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int((self.input_shape+1*2-3)/1 + 1)
        input_size = int((input_size + 1 * 2 - 3) / 1 + 1)
        self.fc1 = nn.Linear(input_size*6, self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        return out


class CnnCls11(Cnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls11, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int((self.input_shape+1*2-3)/1 + 1)
        self.fc1 = nn.Linear(input_size*3, self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        return out


class CnnCls12(Cnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls11, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int((self.input_shape+1*2-3)/1 + 1)
        self.fc1 = nn.Linear(input_size * 3, int(input_size* 3 / 2)).to(device)
        self.fc2 = nn.Linear(int(input_size* 3 / 2), self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        out = self.bn2(out)
        return out

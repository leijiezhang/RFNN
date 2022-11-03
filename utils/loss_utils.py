import torch.nn as nn
import torch
import abc


class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, yhat, y: torch.Tensor):
        loss = []
        return loss


class RMSELoss(LossFunc):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        loss = torch.sqrt(torch.sum((yhat - y).pow(2) / (y.shape[0])))
        return loss


class NRMSELoss(LossFunc):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        loss = torch.sqrt(torch.sum((yhat - y).pow(2)) / (y.shape[0]*torch.var(y)))
        return loss


class MSELoss(LossFunc):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        loss = torch.sum((yhat - y).pow(2)) / (y.shape[0])
        return loss


class Map(LossFunc):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        y_ones = torch.ones(y.shape).to(y.device)
        y_zeros = torch.zeros(y.shape).to(y.device)
        acc_num = torch.where(yhat == y, y_ones, y_zeros).sum()
        acc = acc_num / y.shape[0]
        return acc


class LikelyLoss(LossFunc):
    """
    todo: used for calculate the loss of classification task with one output node
    """
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        yhat = torch.round(yhat)
        yhat = torch.where(yhat > max(y), max(y), yhat)
        y_ones = torch.ones(y.shape).to(y.device)
        y_zeros = torch.zeros(y.shape).to(y.device)
        acc_num = torch.where(yhat == y, y_ones, y_zeros).sum()
        acc = (acc_num / y.shape[0])
        return acc


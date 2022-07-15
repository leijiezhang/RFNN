import warnings

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns, sns.set()
from scipy.cluster.vq import kmeans2

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform
from torchvision import transforms

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.contrib.examples.util import get_data_loader
from pyro.infer import MCMC, NUTS, Predictive, SVI, TraceMeanField_ELBO


class LinearT(nn.Module):
    """Linear transform and transpose"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x).t()


class DeepGP3(pyro.nn.PyroModule):
    def __init__(self, X, y, n_fea, n_cls, Xu, mean_fn):
        super(DeepGP3, self).__init__()
        self.n_fea = n_fea
        self.n_cls = n_cls
        self.layer1 = gp.models.VariationalSparseGP(
            X,
            None,
            gp.kernels.RBF(n_fea, variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=Xu,
            likelihood=None,
            mean_function=mean_fn,
            latent_shape=torch.Size([30]))
        # make sure that the input for next layer is batch_size x 30
        h = mean_fn(X).t()
        hu = mean_fn(Xu).t()
        self.layer2 = gp.models.VariationalSparseGP(
            h,
            y,
            gp.kernels.RBF(30, variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=hu,
            likelihood=gp.likelihoods.MultiClass(num_classes=n_cls),
            latent_shape=torch.Size([n_cls]))

    def model(self, X, y):
        self.layer1.set_data(X, None)
        h_loc, h_var = self.layer1.model()
        # approximate with a Monte Carlo sample (formula 15 of [1])
        h = dist.Normal(h_loc, h_var.sqrt())()
        self.layer2.set_data(h.t(), y)
        self.layer2.model()

    def guide(self, X, y):
        self.layer1.guide()
        self.layer2.guide()

    # make prediction
    def forward(self, X_new):
        # because prediction is stochastic (due to Monte Carlo sample of hidden layer),
        # we make 100 prediction and take the most common one (as in [4])
        pred = []
        for _ in range(100):
            h_loc, h_var = self.layer1(X_new)
            h = dist.Normal(h_loc, h_var.sqrt())()
            f_loc, f_var = self.layer2(h.t())
            pred.append(f_loc.argmax(dim=0))
        return torch.stack(pred).mode(dim=0)[0]
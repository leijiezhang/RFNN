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
    def __init__(self, dim_in, dim_out, device):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False).to(device)
        self.linear.requires_grad_(False)

    def forward(self, x):
        return self.linear(x).t()


class DeepGP(pyro.nn.PyroModule):
    def __init__(self, X, y, input_shape, n_cls, Xu, device):
        super(DeepGP, self).__init__()
        self.input_shape = input_shape
        self.n_cls = n_cls

    def model(self, X, y):
        return X

    def guide(self, X, y):
        return X

    # make prediction
    def forward(self, X_new):
        return X_new


class DeepGP21(DeepGP):
    def __init__(self, X, y, input_shape, n_cls, Xu, device):
        super(DeepGP21, self).__init__(X, y, input_shape, n_cls, Xu, device)
        self.input_shape = input_shape
        self.n_cls = n_cls
        mean_fn1 = LinearT(input_shape, int(input_shape/2), device)
        self.layer1 = gp.models.VariationalSparseGP(
            X,
            None,
            gp.kernels.RBF(input_shape, variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=Xu,
            likelihood=None,
            mean_function=mean_fn1,
            latent_shape=torch.Size([int(input_shape/2)])).to(device)
        self.layer1.u_scale_tril = self.layer1.u_scale_tril * 1e-5
        # make sure that the input for next layer is batch_size x 30
        h = mean_fn1(X).t()
        hu = mean_fn1(Xu).t()
        self.layer2 = gp.models.VariationalSparseGP(
            h,
            y,
            gp.kernels.RBF(int(input_shape/2), variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=hu,
            likelihood=gp.likelihoods.MultiClass(num_classes=n_cls),
            latent_shape=torch.Size([n_cls])).to(device)

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


class DeepGP121(DeepGP):
    def __init__(self, X, y, input_shape, n_cls, Xu, device):
        super(DeepGP121, self).__init__(X, y, input_shape, n_cls, Xu, device)
        self.input_shape = input_shape
        self.n_cls = n_cls
        mean_fn1 = LinearT(input_shape, 2*input_shape, device)
        self.layer1 = gp.models.VariationalSparseGP(
            X,
            None,
            gp.kernels.RBF(input_shape, variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=Xu,
            likelihood=None,
            mean_function=mean_fn1,
            latent_shape=torch.Size([2*input_shape])).to(device)
        self.layer1.u_scale_tril = self.layer1.u_scale_tril * 1e-5
        # make sure that the input for next layer is batch_size x 30
        h1 = mean_fn1(X).t()
        hu1 = mean_fn1(Xu).t()
        
        mean_fn2 = LinearT(2 * input_shape, input_shape, device)
        self.layer2 = gp.models.VariationalSparseGP(
            h1,
            None,
            gp.kernels.RBF(2 * input_shape, variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=hu1,
            likelihood=None,
            mean_function=mean_fn2,
            latent_shape=torch.Size([input_shape])).to(device)
        self.layer2.u_scale_tril = self.layer2.u_scale_tril * 1e-1
        h2 = mean_fn2(h1).t()
        hu2 = mean_fn2(hu1).t()
        self.layer3 = gp.models.VariationalSparseGP(
            h2,
            y,
            gp.kernels.RBF(input_shape, variance=torch.tensor(2.), lengthscale=torch.tensor(2.)),
            Xu=hu2,
            likelihood=gp.likelihoods.MultiClass(num_classes=n_cls),
            latent_shape=torch.Size([n_cls])).to(device)

    def model(self, X, y):
        self.layer1.set_data(X, None)
        h_loc1, h_var1 = self.layer1.model()
        # approximate with a Monte Carlo sample (formula 15 of [1])
        h1 = dist.Normal(h_loc1, h_var1.sqrt())()
        
        self.layer2.set_data(h1.t(), None)
        h_loc2, h_var2 = self.layer2.model()
        # approximate with a Monte Carlo sample (formula 15 of [1])
        h2 = dist.Normal(h_loc2, h_var2.sqrt())()
        # # approximate with a Monte Carlo sample (formula 15 of [1])
        # h3 = dist.Normal(h_loc2, h_var2.sqrt())()
        self.layer3.set_data(h2.t(), y)
        self.layer3.model()

    def guide(self, X, y):
        self.layer1.guide()
        self.layer2.guide()
        self.layer3.guide()

    # make prediction
    def forward(self, X_new):
        # because prediction is stochastic (due to Monte Carlo sample of hidden layer),
        # we make 100 prediction and take the most common one (as in [4])
        pred = []
        for _ in range(100):
            h_loc1, h_var1 = self.layer1(X_new)
            h1 = dist.Normal(h_loc1, h_var1.sqrt())()
            h_loc2, h_var2 = self.layer2(h1.t())
            h2 = dist.Normal(h_loc2, h_var2.sqrt())()
            f_loc, f_var = self.layer3(h2.t())
            pred.append(f_loc.argmax(dim=0))
        return torch.stack(pred).mode(dim=0)[0]
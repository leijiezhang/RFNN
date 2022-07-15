import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer, DeepLikelihood


class HiddenLayer(DeepGPLayer):
  def __init__(self, input_dims, output_dims, num_inducing=128):
    if output_dims is None:
      inducing_points = torch.randn(num_inducing, input_dims)
      if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()
      batch_shape = torch.Size([])
    else:
      inducing_points = torch.randn(output_dims, num_inducing, input_dims)
      if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()
      batch_shape = torch.Size([output_dims])

    variational_distribution = CholeskyVariationalDistribution(
        num_inducing_points=num_inducing,
        batch_shape=batch_shape
    )

    variational_strategy = VariationalStrategy(
        self,
        inducing_points,
        variational_distribution,
        learn_inducing_locations=True
    )

    super().__init__(variational_strategy, input_dims, output_dims)

    self.mean_module = ConstantMean(batch_shape=batch_shape)
    self.covar_module = ScaleKernel(
        RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
        batch_shape=batch_shape, ard_num_dims=None
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return MultivariateNormal(mean_x, covar_x)


class Normal_DeepGP(DeepGP):

    def __init__(self, input_dims, n_cls, n_layer=5, inter_dims=20):
        """All dimension are the same"""
        assert n_layer >= 2
        super().__init__()
        self.first_layer = HiddenLayer(input_dims=input_dims,
                               output_dims=inter_dims)
        if n_layer == 2:
          self.hiddens = []
        else:
          self.hiddens = torch.nn.ModuleList(
              [HiddenLayer(input_dims=inter_dims,output_dims=inter_dims) for _ in range(n_layer-2)])

        self.last_layer = HiddenLayer(input_dims=inter_dims,
                                      output_dims=n_cls)

        self.likelihood = SoftmaxLikelihood(num_features=n_cls, num_classes=n_cls)
        self.n_cls = n_cls

    def forward(self, x):
        x = self.first_layer(x, are_samples=True)
        for hidden in self.hiddens:
            x = hidden(x)
        output = self.last_layer(x)
        return output



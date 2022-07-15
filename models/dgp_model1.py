import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
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

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGP1(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=input_shape,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()
        self.n_cls = n_cls
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class DeepGP21(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        super().__init__()
        self.n_cls = n_cls
        self.likelihood = GaussianLikelihood()
        self.hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=int(input_shape/2),
            mean_type='linear',
        )
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape / 2),
            output_dims=None,
            mean_type='constant',
        )

    def forward(self, inputs):
        hidden_rep = self.hidden_layer1(inputs)
        output = self.last_layer(hidden_rep)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class DeepGP421(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        super().__init__()
        self.n_cls = n_cls
        self.likelihood = GaussianLikelihood()
        self.hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=int(input_shape/2),
            mean_type='linear',
        )
        self.hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape / 2),
            output_dims=int(input_shape / 4),
            mean_type='linear',
        )
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape / 4),
            output_dims=None,
            mean_type='constant',
        )

    def forward(self, inputs):
        hidden_rep = self.hidden_layer1(inputs)
        hidden_rep = self.hidden_layer2(hidden_rep)
        output = self.last_layer(hidden_rep)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class DeepGP121(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        super().__init__()
        self.n_cls = n_cls
        self.likelihood = GaussianLikelihood()
        self.hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=2*input_shape,
            mean_type='linear',
        )
        self.hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=2*input_shape,
            output_dims=input_shape,
            mean_type='linear',
        )
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=n_cls,
            mean_type='constant',
        )

    def forward(self, inputs):
        hidden_rep = self.hidden_layer1(inputs)
        hidden_rep = self.hidden_layer2(hidden_rep)
        output = self.last_layer(hidden_rep)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class DeepGP212(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        super().__init__()
        self.n_cls = n_cls
        self.likelihood = GaussianLikelihood()
        self.hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=int(input_shape/2),
            mean_type='linear',
        )
        self.hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape/2),
            output_dims=input_shape,
            mean_type='linear',
        )
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=None,
            mean_type='constant',
        )

    def forward(self, inputs):
        hidden_rep = self.hidden_layer1(inputs)
        hidden_rep = self.hidden_layer2(hidden_rep)
        output = self.last_layer(hidden_rep)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class DeepGP42124(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        super().__init__()
        self.n_cls = n_cls
        self.likelihood = GaussianLikelihood()
        self.hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=int(input_shape/2),
            mean_type='linear',
        )
        self.hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape/2),
            output_dims=int(input_shape/4),
            mean_type='linear',
        )
        self.hidden_layer3 = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape / 4),
            output_dims=int(input_shape / 2),
            mean_type='linear',
        )
        self.hidden_layer4 = ToyDeepGPHiddenLayer(
            input_dims=int(input_shape / 2),
            output_dims=input_shape,
            mean_type='linear',
        )
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=None,
            mean_type='constant',
        )

    def forward(self, inputs):
        hidden_rep = self.hidden_layer1(inputs)
        hidden_rep = self.hidden_layer2(hidden_rep)
        hidden_rep = self.hidden_layer3(hidden_rep)
        hidden_rep = self.hidden_layer4(hidden_rep)
        output = self.last_layer(hidden_rep)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class DeepGP12421(DeepGP):
    def __init__(self, input_shape, n_cls, device):
        super().__init__()
        self.n_cls = n_cls
        self.likelihood = GaussianLikelihood()
        self.hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=2*input_shape,
            mean_type='linear',
        )
        self.hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=2*input_shape,
            output_dims=4*input_shape,
            mean_type='linear',
        )
        self.hidden_layer3 = ToyDeepGPHiddenLayer(
            input_dims=4*input_shape,
            output_dims=2*input_shape,
            mean_type='linear',
        )
        self.hidden_layer4 = ToyDeepGPHiddenLayer(
            input_dims=2*input_shape,
            output_dims=input_shape,
            mean_type='linear',
        )
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=input_shape,
            output_dims=None,
            mean_type='constant',
        )

    def forward(self, inputs):
        hidden_rep = self.hidden_layer1(inputs)
        hidden_rep = self.hidden_layer2(hidden_rep)
        hidden_rep = self.hidden_layer3(hidden_rep)
        hidden_rep = self.hidden_layer4(hidden_rep)
        output = self.last_layer(hidden_rep)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

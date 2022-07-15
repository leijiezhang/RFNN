import torch
import torch.nn.functional as nnf
import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
import pyro.optim as pyroopt
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.distributions import Normal, Categorical, Uniform
from pyro.nn import PyroModule
from models.dnn_model import Dnn
softplus = torch.nn.Softplus()
import copy
from collections import OrderedDict


class BNN(PyroModule):

    def __init__(self, dataset_name, inference,
                 epochs, lr, n_samples, warmup,
                 dnn: Dnn, dnn_name,
                 step_size=0.005, num_steps=10):
        super(BNN, self).__init__()
        self.dataset_name = dataset_name
        self.inference = inference
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.warmup = warmup
        self.step_size = step_size
        self.num_steps = num_steps
        self.basenet = dnn
        self.basenet_name = dnn_name
        print(self.basenet)
        self.name = self.get_name()

    def get_name(self, n_inputs=None):

        name = str(self.dataset_name) + "_bnn_" + str(self.inference)

        if self.inference == "svi":
            return name + "_ep=" + str(self.epochs) + "_lr=" + str(self.lr)
        elif self.inference == "hmc":
            return name + "_samp=" + str(self.n_samples) + "_warm=" + str(self.warmup) + \
                   "_stepsize=" + str(self.step_size) + "_numsteps=" + str(self.num_steps)

    def model(self, x_data, y_data):

        priors = {}
        for key, value in self.basenet.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key): prior})

        lifted_module = pyro.random_module("module", self.basenet, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            lhat = nnf.log_softmax(logits, dim=-1)
            obs = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
        return obs

    def guide(self, x_data, y_data=None):

        dists = {}
        for key, value in self.basenet.state_dict().items():
            loc = pyro.param(str(f"{key}_{self.basenet_name}_loc"), torch.randn_like(value.float()))
            scale = pyro.param(str(f"{key}_{self.basenet_name}_scale"), torch.randn_like(value.float()))
            distr = Normal(loc=loc, scale=softplus(scale))
            dists.update({str(key): distr})

        lifted_module = pyro.random_module("module", self.basenet, dists)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            preds = nnf.softmax(logits, dim=-1)

        return preds

    def forward(self, inputs, n_samples=10, avg_posterior=False, seeds=None):

        if seeds:
            if len(seeds) != n_samples:
                raise ValueError("Number of seeds should match number of samples.")

        if self.inference == "svi":

            if avg_posterior is True:

                guide_trace = poutine.trace(self.guide).get_trace(inputs)

                avg_state_dict = {}
                for key in self.basenet.state_dict().keys():
                    avg_weights = guide_trace.nodes[str(key) + "_loc"]['value']
                    avg_state_dict.update({str(key): avg_weights})

                self.basenet.load_state_dict(avg_state_dict)
                preds = [self.basenet.model(inputs)]

            else:

                preds = []

                if seeds:
                    for seed in seeds:
                        pyro.set_rng_seed(seed)
                        guide_trace = poutine.trace(self.guide).get_trace(inputs)
                        preds.append(guide_trace.nodes['_RETURN']['value'])

                else:

                    for _ in range(n_samples):
                        guide_trace = poutine.trace(self.guide).get_trace(inputs)
                        preds.append(guide_trace.nodes['_RETURN']['value'])

                # print("\nlearned variational params:\n")
                # print(pyro.get_param_store().get_all_param_names())
                # print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))
                # print("\n", pyro.get_param_store()["model.0.weight_loc"][0][:5])
                # print(guide_trace.nodes['module$$$model.0.weight']["fn"].loc[0][:5])
                # print("posterior sample: ",
                #       guide_trace.nodes['module$$$model.0.weight']['value'][5][0][0])

        elif self.inference == "hmc":

            preds = []
            posterior_predictive = list(self.posterior_predictive.values())

            if seeds is None:
                seeds = range(n_samples)

            for seed in seeds:
                net = posterior_predictive[seed]
                preds.append(net.forward(inputs))
        elif self.inference == "nuts":

            preds = []
            posterior_predictive = list(self.posterior_predictive.values())

            # if seeds is None:
            #     seeds = range(n_samples)

            for seed in torch.arange(len(posterior_predictive)):
                net = posterior_predictive[seed]
                preds.append(net.forward(inputs))

        output_probs = torch.stack(preds).mean(0)
        return output_probs

    def _train_hmc(self, train_loader, n_samples, warmup, step_size, num_steps, device):

        print("\n == HMC training ==")
        pyro.clear_param_store()

        num_batches = int(len(train_loader.dataset) / train_loader.batch_size)
        batch_samples = int(n_samples / num_batches) + 1
        print("\nn_batches=", num_batches, "\tbatch_samples =", batch_samples)

        kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        mcmc = MCMC(kernel=kernel, num_samples=batch_samples, warmup_steps=warmup, num_chains=1)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            # labels = y_batch.to(device).argmax(-1)
            labels = y_batch.squeeze()
            # labels = y_batch
            mcmc.run(x_batch, labels)

        self.posterior_predictive = {}
        posterior_samples = mcmc.get_samples(n_samples)
        state_dict_keys = list(self.basenet.state_dict().keys())

        # print("\n", list(posterior_samples.values())[-1])

        for model_idx in range(n_samples):
            net_copy = copy.deepcopy(self.basenet)

            model_dict = OrderedDict({})
            for weight_idx, weights in enumerate(posterior_samples.values()):
                model_dict.update({state_dict_keys[weight_idx]: weights[model_idx]})

            net_copy.load_state_dict(model_dict)
            self.posterior_predictive.update({str(model_idx): net_copy})

        # print("\n", weights[model_idx])

        # self.save()

    def _train_nuts(self, train_loader, n_samples, warmup, step_size, num_steps, device):

        print("\n == NUTS training ==")
        pyro.clear_param_store()

        num_batches = int(len(train_loader.dataset) / train_loader.batch_size)
        batch_samples = int(n_samples / num_batches) + 1
        print("\nn_batches=", num_batches, "\tbatch_samples =", batch_samples)

        kernel = NUTS(self.model, adapt_step_size=True)
        mcmc = MCMC(kernel=kernel, num_samples=batch_samples, warmup_steps=warmup, num_chains=1)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            # labels = y_batch.to(device).argmax(-1)
            labels = y_batch.squeeze()
            # labels = y_batch
            mcmc.run(x_batch, labels)

        self.posterior_predictive = {}
        posterior_samples = mcmc.get_samples(n_samples)
        state_dict_keys = list(self.basenet.state_dict().keys())

        # print("\n", list(posterior_samples.values())[-1])

        for model_idx in range(n_samples):
            net_copy = copy.deepcopy(self.basenet)

            model_dict = self.basenet.state_dict()
            for posterior_sample_key, posterior_sample_value in zip(posterior_samples.keys(), posterior_samples.values()):
                model_dict.update({posterior_sample_key.split("$$$")[1]: posterior_sample_value[model_idx]})

            net_copy.load_state_dict(model_dict)
            self.posterior_predictive.update({str(model_idx): net_copy})

        # print("\n", weights[model_idx])

        # self.save()

    def _train_svi(self, train_loader, epochs, lr, device):
        self.device = device

        print("\n == SVI training ==")

        optimizer = pyro.optim.Adam({"lr": lr})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        loss_list = []
        accuracy_list = []

        for epoch in range(epochs):
            loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # labels = y_batch.argmax(-1)
                labels = y_batch.squeeze()
                loss += svi.step(x_data=x_batch, y_data=labels)

                outputs = self.forward(x_batch, n_samples=10)
                predictions = outputs.argmax(dim=-1)
                correct_predictions += (predictions == labels).sum().item()

            # print("\n", pyro.get_param_store()["model.0.weight_loc"][0][:5])
            # print("\n", predictions[:10], "\n", labels[:10])

            total_loss = loss / len(train_loader.dataset)
            accuracy = 100 * correct_predictions / len(train_loader.dataset)

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}",
                  end="\t")

            loss_list.append(loss)
            accuracy_list.append(accuracy)

        # self.save()

    def train(self, train_loader, device):
        self.device = device
        self.basenet.device = device

        self.to(device)
        self.basenet.to(device)

        # random.seed(0)
        pyro.set_rng_seed(0)

        if self.inference == "svi":
            self._train_svi(train_loader, self.epochs, self.lr, device)

        elif self.inference == "hmc":
            self._train_hmc(train_loader, self.n_samples, self.warmup,
                            self.step_size, self.num_steps, device)
        elif self.inference == "nuts":
            self._train_nuts(train_loader, self.n_samples, self.warmup,
                            self.step_size, self.num_steps, device)

    def evaluate(self, test_loader, device, n_samples=10, seeds_list=None):
        self.device = device
        self.basenet.device = device
        self.to(device)
        self.basenet.to(device)

        # random.seed(0)
        pyro.set_rng_seed(0)

        bnn_seeds = list(range(n_samples)) if seeds_list is None else seeds_list

        with torch.no_grad():
            correct_predictions = 0.0
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                outputs = self.forward(x_batch, n_samples=n_samples, seeds=bnn_seeds)
                predictions = outputs.argmax(-1)
                # labels = y_batch.to(device).argmax(-1)
                labels = y_batch.squeeze()
                correct_predictions += (predictions == labels).sum().item()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("Accuracy: %.2f%%" % (accuracy))
            return accuracy
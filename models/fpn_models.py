import torch
import torch.nn as nn
import torch.nn.functional as F


class FpnMlpFsCls_2(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    The fire strength is generated from a 3 layer mlp net structure
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, device):
        """

        :param prototypes:
        :param n_fea:
        :param device:
        """
        super(FpnMlpFsCls_2, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        # parameters in network
        # self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)
        self.var = nn.Parameter(variance, requires_grad=True).to(device)

        # self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        # self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        # if torch.cuda.is_available():
        #     self.proto = self.proto.cuda()
        #     self.proto_reform = self.proto_reform.cuda()
        self.fs_layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.n_fea, self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, 1),
            # torch.nn.Tanh()
            # torch.nn.Linear(self.n_fea, self.n_fea),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.n_fea, 1),
        ).to(device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [torch.nn.Sequential(
            torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.n_fea, self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, self.n_cls),
            # torch.nn.Tanh()
        ).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        var_expands = self.var.repeat_interleave(n_batch, dim=0)
        fuzzy_set = torch.exp(
            -(data_expands - proto_expands) ** 2 / (2 * var_expands ** 2))
        data_diff = fuzzy_set.view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        outputs = F.softmax(outputs, dim=1)

        return outputs


class ConsequentLayer(nn.Module):
    """
    This is the consequent layer of FPNN based on BP
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(ConsequentLayer, self).__init__()
        # parameters in network
        # self.consq_layers = nn.Linear(n_fea, num_class)
        hidden_dim = int(n_fea / 2)
        if hidden_dim < num_class:
            hidden_dim = num_class
        self.consq_layers = nn.Sequential(
            nn.Linear(n_fea, hidden_dim),
            # nn.BatchNorm1d(2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_class),
        )

    def forward(self, data: torch.Tensor):
        output = self.consq_layers(data)

        return output


class FpnMlpFsCls_1(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    The fire strength is generated from a 3 layer mlp net structure
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, device):
        """

        :param prototypes:
        :param n_fea:
        :param device:
        """
        super(FpnMlpFsCls_1, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        # parameters in network
        # self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)
        self.var = nn.Parameter(variance, requires_grad=True).to(device)

        # self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        # self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        # if torch.cuda.is_available():
        #     self.proto = self.proto.cuda()
        #     self.proto_reform = self.proto_reform.cuda()
        self.fs_layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            # nn.BatchNorm1d(2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.n_fea, self.n_fea),
            # nn.BatchNorm1d(self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, 1),
            # torch.nn.Tanh()
        ).to(device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [ConsequentLayer(self.n_fea, self.n_cls).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        var_expands = self.var.repeat_interleave(n_batch, dim=0)
        fuzzy_set = torch.exp(
            -(data_expands - proto_expands) ** 2 / (2 * var_expands ** 2))
        data_diff = fuzzy_set.view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        outputs = F.softmax(outputs, dim=1)

        return outputs


class FpnMlpFsCls(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    The fire strength is generated from a 3 layer mlp net structure
    """

    def __init__(self, prototypes: torch.Tensor, n_cls, device):
        """

        :param prototypes:
        :param n_fea:
        :param device:
        """
        super(FpnMlpFsCls, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        # parameters in network
        # self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)

        # self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        # self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        # if torch.cuda.is_available():
        #     self.proto = self.proto.cuda()
        #     self.proto_reform = self.proto_reform.cuda()
        self.fs_layers = torch.nn.Sequential(
            # torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            # torch.nn.ReLU(),
            # torch.nn.Linear(2 * self.n_fea, self.n_fea),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, 1),
            # torch.nn.Tanh()
        ).to(device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [nn.Linear(self.n_fea, self.n_cls).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        # outputs = F.softmax(outputs, dim=1)

        return outputs


class FpnMlpFsReg(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    The fire strength is generated from a 3 layer mlp net structure
    """

    def __init__(self, prototypes: torch.Tensor, device):
        """

        :param prototypes:
        :param n_fea:
        :param device:
        """
        super(FpnMlpFsReg, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = 1

        # parameters in network
        # self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)

        # self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        # self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        # if torch.cuda.is_available():
        #     self.proto = self.proto.cuda()
        #     self.proto_reform = self.proto_reform.cuda()
        self.fs_layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.n_fea, self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, 1),
            # torch.nn.Tanh()
        ).to(device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [nn.Linear(self.n_fea, self.n_cls).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        # outputs = F.softmax(outputs, dim=1)

        return outputs


class FpnCov1dFSCls(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    The fire strength is generated from a 3 layer mlp net structure
    """

    def __init__(self, prototypes: torch.Tensor, n_cls, device):
        """

        :param prototypes:
        :param n_fea:
        :param device:
        """
        super(FpnCov1dFSCls, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        # parameters in network
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)

        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        n_channel = 4
        n_padding = 1
        kernel_size = 5
        pooling_size = 2
        att_size = n_channel * torch.floor(
            (torch.floor(torch.tensor(self.n_fea + 2 * n_padding + 1 - kernel_size).float() / pooling_size) +
             2 * n_padding + 1 - kernel_size) / pooling_size)
        self.fs_layers = RelationNetwork(int(att_size), int(att_size / 2), n_channel, n_padding,
                                         kernel_size, pooling_size, device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [nn.Linear(self.n_fea, self.n_cls).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        # outputs = F.softmax(outputs, dim=1)

        return outputs


class FpnCov1dFSReg(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    The fire strength is generated from a 3 layer mlp net structure
    """

    def __init__(self, prototypes: torch.Tensor, device):
        """

        :param prototypes:
        :param n_fea:
        :param device:
        """
        super(FpnCov1dFSReg, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = 1

        # parameters in network
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)

        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(
            device)
        n_channel = 4
        n_padding = 1
        kernel_size = 3
        pooling_size = 2
        att_size = n_channel * torch.ceil(
            (torch.ceil(torch.tensor(self.n_fea + 2 * n_padding + 1 - kernel_size).float() / pooling_size) +
             2 * n_padding + 1 - kernel_size) / pooling_size)
        self.fs_layers = RelationNetwork(int(att_size), int(att_size / 2), n_channel, n_padding,
                                         kernel_size, pooling_size, device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [nn.Linear(self.n_fea, self.n_cls).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        # outputs = F.softmax(outputs, dim=1)

        return outputs


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size, n_channel, n_padding, kernel_size, pooling_size, device):
        super(RelationNetwork, self).__init__()
        self.conv1d1 = nn.Conv1d(1, n_channel, kernel_size=kernel_size, padding=n_padding).to(device)
        self.bn1 = nn.BatchNorm1d(n_channel, momentum=1, affine=True).to(device)
        self.rl1 = nn.ReLU().to(device)
        self.maxp1 = nn.AvgPool1d(pooling_size).to(device)

        self.conv1d2 = nn.Conv1d(n_channel, n_channel, kernel_size=kernel_size, padding=n_padding).to(device)
        self.bn2 = nn.BatchNorm1d(n_channel, momentum=1, affine=True).to(device)
        self.rl2 = nn.ReLU().to(device)
        self.maxp2 = nn.AvgPool1d(pooling_size).to(device)

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, n_channel, kernel_size=kernel_size, padding=n_padding),
            nn.BatchNorm1d(n_channel, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_channel, n_channel, kernel_size=kernel_size, padding=n_padding),
            nn.BatchNorm1d(n_channel, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size)).to(device)
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        # out = self.layer1(x.unsqueeze(1))
        # out = self.layer2(out)
        out = self.conv1d1(x.unsqueeze(1))
        # out = self.bn1(out)
        out = self.rl1(out)
        out = self.maxp1(out)

        out = self.conv1d2(out)
        # out = self.bn2(out)
        out = self.rl2(out)
        out = self.maxp2(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out


class AntecedentLayerRdm(nn.Module):
    """
    This is the antecedent layer of FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea):
        """
        :param n_fea: feature number of samples
        """
        super(AntecedentLayerRdm, self).__init__()
        # parameters in network
        prototype = torch.zeros(1, n_fea)
        variance = torch.ones(1, n_fea)
        self.proto = nn.Parameter(prototype, requires_grad=True)
        self.var = nn.Parameter(variance, requires_grad=True)

    def forward(self, data: torch.Tensor):
        # membership_values = torch.exp(-(data - self.proto) ** 2 * (2 * self.var_process(self.var) ** 2))
        # membership_values = torch.exp(-(data - torch.clamp(self.proto, -1, 1)) ** 2 / (2 * torch.clamp(
        #     self.var, 1e-4, 1) ** 2))
        membership_values = torch.exp(-(data - self.proto) ** 2 / (2 * torch.clamp(
            self.var, 1e-4, 1e-1) ** 2))

        return membership_values


class AntecedentLayerIni(nn.Module):
    """
    This is the antecedent layer of FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, prototype, variance):
        """
        :param n_fea: feature number of samples
        """
        super(AntecedentLayerIni, self).__init__()
        # parameters in network
        self.proto = nn.Parameter(prototype, requires_grad=True)
        self.var = nn.Parameter(variance, requires_grad=True)

    def forward(self, data: torch.Tensor):
        membership_values = torch.exp(-(data - self.proto) ** 2 / (2 * torch.clamp(
            self.var, 1e-4, 1e-1) ** 2))

        return membership_values


class ConsequentLayerFC(nn.Module):
    """
    This is the consequent layer of FPNN based on BP
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(ConsequentLayerFC, self).__init__()
        # parameters in network
        self.consq_layers = nn.Linear(n_fea, num_class)

    def forward(self, data: torch.Tensor):
        output = self.consq_layers(data)

        return output


class ConsequentLayerMLP(nn.Module):
    """
    This is the consequent layer of FPNN based on BP
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(ConsequentLayerMLP, self).__init__()
        # parameters in network
        # hidden_dim = int(n_fea/2)
        # if hidden_dim < num_class:
        #     hidden_dim = num_class
        # self.consq_layers = nn.Sequential(
        #     nn.Linear(n_fea, hidden_dim),
        #     # nn.BatchNorm1d(2 * self.n_fea),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, num_class),
        # )
        self.consq_layers = nn.Sequential(
            nn.Linear(n_fea, 2 * n_fea),
            nn.ELU(),
            nn.Linear(2 * n_fea, n_fea),
            nn.ELU(),
            nn.Linear(n_fea, num_class),
        )

    def forward(self, data: torch.Tensor):
        output = self.consq_layers(data)

        return output


class ConsequentLayerMLP1(nn.Module):
    """
    This is the consequent layer of FPNN based on BP
    """

    def __init__(self, n_fea, num_class, drop_rate=0.25):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(ConsequentLayerMLP1, self).__init__()
        # parameters in network
        self.consq_layers = nn.Sequential(
            nn.Linear(n_fea, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, num_class),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, data: torch.Tensor):
        output = self.consq_layers(data)

        return output


class RuleMLPIni(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, prototype, variance, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleMLPIni, self).__init__()
        self.antecedent_layer = AntecedentLayerIni(prototype, variance)
        self.consequent_layer = ConsequentLayerMLP(n_fea, num_class)


class RuleMLPIni1(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, prototype, variance, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleMLPIni1, self).__init__()
        self.antecedent_layer = AntecedentLayerIni(prototype, variance)
        self.consequent_layer = ConsequentLayerMLP1(n_fea, num_class)


class RuleFCIni(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, prototype, variance, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleFCIni, self).__init__()
        self.antecedent_layer = AntecedentLayerIni(prototype, variance)
        self.consequent_layer = ConsequentLayerFC(n_fea, num_class)


class RuleMLPRdm(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleMLPRdm, self).__init__()
        self.antecedent_layer = AntecedentLayerRdm(n_fea)
        self.consequent_layer = ConsequentLayerMLP(n_fea, num_class)


class RuleMLPRdm1(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, num_class, drop_rate):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleMLPRdm1, self).__init__()
        self.antecedent_layer = AntecedentLayerRdm(n_fea)
        self.consequent_layer = ConsequentLayerMLP1(n_fea, num_class, drop_rate)


class RuleFCRdm(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleFCRdm, self).__init__()
        self.antecedent_layer = AntecedentLayerRdm(n_fea)
        self.consequent_layer = ConsequentLayerFC(n_fea, num_class)


class RuleFC(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleFC, self).__init__()
        self.antecedent_layer = AntecedentLayerRdm(n_fea)
        self.consequent_layer = ConsequentLayerFC(n_fea, num_class)


class FSLayerMLP(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self, n_fea, dropout_rat=0.0):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerMLP, self).__init__()
        self.dr = dropout_rat
        # parameters in network
        # self.fs_layers = nn.Sequential(
        #     nn.Linear(n_fea, int(n_fea/2)),
        #     nn.ELU(),
        #     nn.Linear(int(n_fea/2), int(n_fea/4)),
        #     nn.ELU(),
        #     nn.Linear(int(n_fea/4), 1),
        #     # nn.Tanh()
        #     # nn.Dropout(p=self.dr)
        # )
        self.fs_layers = nn.Sequential(
            nn.Linear(n_fea, 2 * n_fea),
            nn.ELU(),
            nn.Linear(2 * n_fea, n_fea),
            nn.ELU(),
            nn.Linear(n_fea, 1),
            # nn.Tanh()
            # nn.Dropout(p=self.dr)
        )

    def forward(self, data: torch.Tensor):
        output = self.fs_layers(data)
        return output


class FSLayerMLP1(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self, n_fea, dropout_rat=0.0):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerMLP1, self).__init__()
        self.dr = dropout_rat
        # parameters in network
        self.fs_layers = nn.Sequential(
            nn.Linear(n_fea, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
            # nn.Tanh()
            nn.Dropout(p=self.dr)
        )

    def forward(self, data: torch.Tensor):
        output = self.fs_layers(data)
        return output


class FSLayerL2(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerL2, self).__init__()
        # parameters in network

    def forward(self, data: torch.Tensor):
        output = (data * data).sum(1).unsqueeze(1)
        return output


class FSLayerMLPFNN(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self, n_rules, n_fea, n_cls, device):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerMLPFNN, self).__init__()
        # parameters in network
        self.fs_layers = FnnNormMlpRdm(n_rules, n_fea, n_cls, device)

    def forward(self, data: torch.Tensor):
        output, _ = self.fs_layers(data)
        return output


class FSLayerFCFNN(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self, n_rules, n_fea, n_cls, device):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerFCFNN, self).__init__()
        # parameters in network
        self.fs_layers = FnnNormFCRdm(n_rules, n_fea, n_cls, device)

    def forward(self, data: torch.Tensor):
        output, _ = self.fs_layers(data)
        return output


class FnnMlpMlpIni(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnMlpMlpIni, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerMLP(self.n_fea, 0.25).to(device)
        self.rule_list = [RuleMLPIni(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnMlpMlpRdm(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, n_rules, n_fea, n_cls, drop_rate, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnMlpMlpRdm, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea
        self.n_cls = n_cls

        self.fs_layer = FSLayerMLP1(self.n_fea, drop_rate).to(device)
        self.rule_list = [RuleMLPRdm1(self.n_fea, self.n_cls, drop_rate).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnMlpMlpIni1(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnMlpMlpIni1, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerMLP1(self.n_fea, 0.25).to(device)
        self.rule_list = [RuleMLPIni(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnNormMlpIni(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnNormMlpIni, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerL2().to(device)
        self.rule_list = [RuleMLPIni1(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnNormFCIni(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnNormFCIni, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerL2().to(device)
        self.rule_list = [RuleFCIni(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnNormMlpRdm(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, n_rules, n_fea, n_cls, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnNormMlpRdm, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea
        self.n_cls = n_cls

        self.fs_layer = FSLayerL2().to(device)
        self.rule_list = [RuleMLPRdm(self.n_fea, self.n_cls).to(device)
                          for _ in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnNormFCRdm(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, n_rules, n_fea, n_cls, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnNormFCRdm, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea
        self.n_cls = n_cls

        self.fs_layer = FSLayerL2().to(device)
        self.rule_list = [RuleFCRdm(self.n_fea, self.n_cls).to(device)
                          for _ in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnMlpFnnMlpIni(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, n_rule_inf, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnMlpFnnMlpIni, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerMLPFNN(n_rule_inf, self.n_fea, 1, device).to(device)
        self.rule_list = [RuleMLPIni(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnFcFnnMlpIni(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, n_rule_inf, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnFcFnnMlpIni, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerFCFNN(n_rule_inf, self.n_fea, 1, device).to(device)
        self.rule_list = [RuleMLPIni(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FnnFcFnnFCIni(nn.Module):
    """
    This is the FNN based on BP, the firing strength is generated via 3-layer MLP
    """

    def __init__(self, prototypes: torch.Tensor, variance: torch.Tensor, n_cls, n_rule_inf, device):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FnnFcFnnFCIni, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        self.fs_layer = FSLayerFCFNN(n_rule_inf, self.n_fea, 1, device).to(device)
        self.rule_list = [RuleFCIni(self.n_fea, prototypes[i, :], variance[i, :], self.n_cls).to(device)
                          for i in range(self.n_rules)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini, dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed, fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength

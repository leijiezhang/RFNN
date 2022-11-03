from utils.param_config import ParamConfig
from utils.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# from gpytorch.models.deep_gps import DeepGP
from typing import Dict
import lightgbm as lgb
# from gpytorch.likelihoods import GaussianLikelihood
from models.dgp_model import *
from torchmetrics import F1Score
import torch.nn as nn
import pyro
from pyro import poutine
from pyro.infer import SVI
import pyro.optim as pyroopt
from pyro.infer.mcmc import MCMC, HMC, NUTS
import os
import torch
# from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from utils.loss_utils import NRMSELoss, LikelyLoss, LossFunc, MSELoss
from models.fpn_models import *

from utils.dataset import DatasetTorch, DatasetTorchB
# from models.dnn_model import MlpReg, Dnn, MlpCls21, MlpCls212, MlpCls121, MlpCls421, MlpCls12421, MlpCls42124
# from models.dnn_model import CnnCls11, CnnCls12, CnnCls21, CnnCls22
# from models.dnn_model import MlpCls21GNIA, MlpCls212GNIA, MlpCls121GNIA, MlpCls421GNIA, MlpCls12421GNIA, MlpCls42124GNIA
# from models.dnn_model import CnnCls11GNIA, CnnCls12GNIA, CnnCls21GNIA, CnnCls22GNIA
from models.dnn_model import *
import scipy.io as io
from models.h_utils import HNormal
from models.rules import RuleKmeans
from models.fnn_solver import FnnSolveReg
from models.bnn_model import *
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def svc(train_fea: torch.Tensor, test_fea: torch.Tensor, train_gnd: torch.Tensor,
        test_gnd: torch.Tensor, loss_fun: LossFunc, paras: Dict):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param train_fea: training data
    :param test_fea: test data
    :param train_gnd: training label
    :param test_gnd: test label
    :param loss_fun: the loss function that used to calculate the loss of regression or f1uracy of classification task
    :param paras: parameters that used for training SVC model

    :return:
    """
    """ codes for parameters 
        paras = dict()
        paras['kernel'] = 'rbf'
        paras['gamma'] = gamma
        paras['C'] = C
    """
    print("training the one-class SVM")
    train_gnd = train_gnd.squeeze()
    test_gnd = test_gnd.squeeze()
    if 'kernel' in paras:
        svm_kernel = paras['kernel']
    else:
        svm_kernel = 'rbf'
    if 'gamma' in paras:
        svm_gamma = paras['gamma']
    else:
        svm_gamma = 'scale'
    if 'C' in paras:
        svm_c = paras['C']
    else:
        svm_c = 1
    svm_train = SVC(kernel=svm_kernel, gamma=svm_gamma, C=svm_c)
    clf = make_pipeline(StandardScaler(), svm_train)
    clf.fit(train_fea.numpy(), train_gnd.numpy())
    train_gnd_hat = clf.predict(train_fea.numpy())
    test_gnd_hat = clf.predict(test_fea.numpy())

    train_f1 = loss_fun.forward(train_gnd.squeeze(), torch.tensor(train_gnd_hat))
    test_f1 = loss_fun.forward(test_gnd.squeeze(), torch.tensor(test_gnd_hat))

    """ following code is designed for those functions that need to output the svm results
    if test_data.task == 'C':
        param_config.log.info(f"f1uracy of training data using SVM: {train_f1:.2f}%")
        param_config.log.info(f"f1uracy of test data using SVM: {test_f1:.2f}%")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_f1:.4f}")
        param_config.log.info(f"loss of test data using SVM: {test_f1:.4f}")
    """

    return train_f1, test_f1


def svr(train_fea: torch.Tensor, test_fea: torch.Tensor, train_gnd: torch.Tensor,
        test_gnd: torch.Tensor, loss_fun: LossFunc, paras: Dict):
    """
    todo: this is the method for sSVR
    :param train_fea: training data
    :param test_fea: test data
    :param train_gnd: training label
    :param test_gnd: test label
    :param loss_fun: the loss function that used to calculate the loss of regression or f1uracy of classification task
    :param paras: parameters that used for training SVC model

    :return:
    """
    """ codes for parameters 
        paras = dict()
        paras['kernel'] = 'rbf'
        paras['gamma'] = gamma
        paras['C'] = C
    """
    print("training the one-class SVR")
    train_gnd = train_gnd.squeeze()
    test_gnd = test_gnd.squeeze()
    if 'kernel' in paras:
        svm_kernel = paras['kernel']
    else:
        svm_kernel = 'rbf'
    if 'gamma' in paras:
        svm_gamma = paras['gamma']
    else:
        svm_gamma = 'scale'
    if 'C' in paras:
        svm_c = paras['C']
    else:
        svm_c = 1
    svm_train = SVR(kernel=svm_kernel, gamma=svm_gamma, C=svm_c)
    clf = make_pipeline(StandardScaler(), svm_train)
    clf.fit(train_fea.numpy(), train_gnd.numpy())
    train_gnd_hat = clf.predict(train_fea.numpy())
    test_gnd_hat = clf.predict(test_fea.numpy())

    train_loss = loss_fun.forward(train_gnd.squeeze(), torch.tensor(train_gnd_hat))
    test_loss = loss_fun.forward(test_gnd.squeeze(), torch.tensor(test_gnd_hat))

    """ following code is designed for those functions that need to output the svr results
    if test_data.task == 'C':
        param_config.log.info(f"f1uracy of training data using SVR: {train_f1:.2f}%")
        param_config.log.info(f"f1uracy of test data using SVE: {test_f1:.2f}%")
    else:
        param_config.log.info(f"loss of training data using SVE: {train_loss:.4f}")
        param_config.log.info(f"loss of test data using SVE: {test_loss:.4f}")
    """

    return train_loss, test_loss


def fnn_cls(n_rules, train_data: Dataset, test_data: Dataset):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param n_rules:
        :param train_data: training dataset
        :param test_data: test dataset
        :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_f1 = LikelyLoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_f1 = LikelyLoss().forward(test_data.gnd, y_test_hat)

    # param_config.log.info(f"Training f1 of traditional FNN: {fnn_train_f1}")
    # param_config.log.info(f"Test f1 of test traditional FNN: {fnn_test_f1}")
    return fnn_train_f1, fnn_test_f1


def gp_cls(train_data: Dataset, test_data: Dataset, length_scale):
    """
        todo: this is the method for Gaussian process classification task
        :param param_config:
        :param train_data: training dataset
        :param test_data: test dataset
        :param length_scale:
        :return:
    """
    kernel = 1.0 * RBF(length_scale)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(train_data.fea.numpy(), train_data.gnd.numpy().ravel())
    n_cls = train_data.y.unique().shape[0]
    train_f1 = gpc.predict(train_data.fea.numpy())
    test_f1 = gpc.predict(test_data.fea.numpy())

    return train_f1, test_f1


def fnn_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param test_data: test dataset
        :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = NRMSELoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = NRMSELoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training me of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test mse of test traditional FNN: {fnn_test_mse}")
    return fnn_train_mse, fnn_test_mse


def dnn_cls(dnn_model: Dnn, param_config: ParamConfig, train_loader: DataLoader, valid_loader: DataLoader, model_name):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param dnn_model: mlp model
        :param param_config: config information
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :param model_name: the name of the model
        :return:
    """
    param_config.log.info(f"mlp epoch:======================={model_name} started===========================")
    n_cls = dnn_model.n_cls
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch
    n_para = sum(param.numel() for param in dnn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    mlp_train_f1 = torch.empty(0, 1).to(param_config.device)
    mlp_valid_f1 = torch.empty(0, 1).to(param_config.device)

    for epoch in range(epochs):
        dnn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs = dnn_model(data)
            loss = loss_fn(outputs, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dnn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = dnn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            f1 = F1Score(num_classes=n_cls).to(param_config.device)
            f1_train = f1(predicted_train, gnd_train.long().squeeze())
            mlp_train_f1 = torch.cat([mlp_train_f1, f1_train.unsqueeze(0).unsqueeze(1)], 0)

            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = dnn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            f1_val = f1(predicted_val, gnd_val.long().squeeze())
            mlp_valid_f1 = torch.cat([mlp_valid_f1, f1_val.unsqueeze(0).unsqueeze(1)], 0)

        param_config.log.info(
            f"{model_name} epoch : {epoch + 1}, train f1 : {mlp_train_f1[-1, 0]}, test f1 : {mlp_valid_f1[-1, 0]}")

    param_config.log.info(f":======================={model_name} finished===========================")
    return mlp_train_f1, mlp_valid_f1


def lightgbm_cls(param_config: ParamConfig, train_data, valid_data, model_name):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param dnn_model: mlp model
        :param param_config: config information
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :param model_name: the name of the model
        :return:
    """
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_data.fea.cpu().numpy(), train_data.gnd.cpu().numpy().squeeze())
    lgb_eval = lgb.Dataset(valid_data.fea.cpu().numpy(), valid_data.gnd.cpu().numpy().squeeze(), reference=lgb_train)

    # specify your configurations as a dict
    n_cls = train_data.gnd.unique().shape[0]
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_classes': n_cls,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.early_stopping(stopping_rounds=5)])
    print('Starting predicting...')
    # predict
    predicted_train = gbm.predict(train_data.fea.cpu().numpy(), num_iteration=gbm.best_iteration)
    _, predicted_train = torch.max(torch.tensor(predicted_train), 1)
    predicted_val = gbm.predict(valid_data.fea.cpu().numpy(), num_iteration=gbm.best_iteration)
    _, predicted_val = torch.max(torch.tensor(predicted_val), 1)
    # n_cls = train_data.gnd.unique().shape[0]
    f1 = F1Score(num_classes=n_cls).to(param_config.device)
    lgb_train_f1 = f1(predicted_train, train_data.gnd.long().squeeze().cpu())
    lgb_valid_f1 = f1(predicted_val, valid_data.gnd.long().squeeze().cpu())

    param_config.log.info(f":======================={model_name} finished===========================")
    return lgb_train_f1, lgb_valid_f1


def lgb_cmp_mthds(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for deep gaussian process
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    lgb_train_f1, lgb_valid_f1 = lightgbm_cls(param_config, train_data, test_data, "light_gbm")

    # save all the results
    save_dict = dict()
    save_dict["lgb_train_f1"] = lgb_train_f1.cpu().numpy()
    save_dict["lgb_valid_f1"] = lgb_valid_f1.cpu().numpy()

    data_save_file = f"{data_save_dir}/f1_lgb_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}" \
                     f"_k_{current_k + 1}.mat"
    io.savemat(data_save_file, save_dict)
    return lgb_train_f1, lgb_valid_f1


def dgp_cls(dgp_model: DeepGP, param_config: ParamConfig, train_loader: DataLoader, valid_loader: DataLoader, model_name):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param dnn_model: mlp model
        :param param_config: config information
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :param model_name: the name of the model
        :return:
    """
    param_config.log.info(f"dgp epoch:======================={model_name} started===========================")
    # if torch.cuda.is_available():
    #     dgp_model = dgp_model.cuda()

    # dgp_model.layer1.u_scale_tril = dgp_model.layer1.u_scale_tril * 1e-5
    # dgp_model.cuda()
    optimizer = torch.optim.Adam(dgp_model.parameters(), lr=param_config.lr_dgp)
    loss_fn = TraceMeanField_ELBO().differentiable_loss
    n_cls = train_loader.dataset.y.unique().shape[0]
    # loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch_dgp
    n_para = sum(param.numel() for param in dgp_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    dgp_train_f1 = torch.empty(0, 1).to(param_config.device)
    dgp_valid_f1 = torch.empty(0, 1).to(param_config.device)

    for epoch in range(epochs):
        dgp_model.train()

        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(dgp_model.model, dgp_model.guide, data, labels.squeeze())
            loss.backward()
            optimizer.step()

        dgp_model.eval()
        outputs_train = torch.empty(0, 1).to(param_config.device)
        outputs_val = torch.empty(0, 1).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = dgp_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp.unsqueeze(1)), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            f1 = F1Score(num_classes=n_cls).to(param_config.device)
            f1_train = f1(outputs_train.long(), gnd_train.long())
            dgp_train_f1 = torch.cat([dgp_train_f1, f1_train.unsqueeze(0).unsqueeze(1)], 0)

            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = dgp_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp.unsqueeze(1)), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            f1_val = f1(outputs_val.long(), gnd_val.long())
            dgp_valid_f1 = torch.cat([dgp_valid_f1, f1_val.unsqueeze(0).unsqueeze(1)], 0)

        param_config.log.info(
            f"{model_name} epoch : {epoch + 1}, loss: {loss}, train f1: {dgp_train_f1[-1, 0]}, test f1: "
            f"{dgp_valid_f1[-1, 0]}")

    param_config.log.info(f":======================={model_name} finished===========================")
    return dgp_train_f1, dgp_valid_f1


def bnn_cls(dnn_model: Dnn, param_config: ParamConfig, train_loader: DataLoader, valid_loader: DataLoader, model_name):
    """
        todo: this is the method for Baysian Neural Network
        :param dnn_model: mlp model
        :param param_config: config information
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :param model_name: the name of the model
        :return:
    """
    param_config.log.info(f"{param_config.inference}_bnn epoch:======================={model_name} started============="
                          f"==============")
    n_para = sum(param.numel() for param in dnn_model.parameters())
    bnn_model = BNN(param_config.dataset_folder, param_config.inference, param_config.n_epoch_svi, param_config.lr_svi,
                    n_samples=param_config.n_samples, warmup=param_config.warmup,
                    dnn=dnn_model, dnn_name=model_name,
                    step_size=param_config.step_size, num_steps=param_config.num_steps)
    bnn_model.train(train_loader, param_config.device)
    param_config.log.info(f'# generator parameters: {n_para}')
    bnn_train_f1 = torch.empty(0, 1).to(param_config.device)
    bnn_valid_f1 = torch.empty(0, 1).to(param_config.device)
    n_clas = train_loader.dataset.y.unique().shape[0]
    f1_train = torch.tensor(bnn_model.evaluate_f1(train_loader, param_config.device, n_clas, 100)).unsqueeze(0).unsqueeze(1).to(param_config.device)
    f1_val = torch.tensor(bnn_model.evaluate_f1(valid_loader, param_config.device, n_clas, 100)).unsqueeze(0).unsqueeze(1).to(param_config.device)
    bnn_train_f1 = torch.cat([bnn_train_f1, f1_train], 0)
    bnn_valid_f1 = torch.cat([bnn_valid_f1, f1_val], 0)
    param_config.log.info(
        f"{model_name} epoch : {param_config.n_epoch}, train f1 : {bnn_train_f1[-1, 0]}, test f1 : {bnn_valid_f1[-1, 0]}")

    param_config.log.info(f":======================={model_name} finished===========================")
    return bnn_train_f1, bnn_valid_f1


def fpn_cls(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    # prototype_ids, prototype_list = kmeans(
    #     X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
    #     device=torch.device(train_data.fea.device)
    # )
    kmeans = KMeans(n_clusters=param_config.n_rules, random_state=0).fit(train_data.fea.cpu())
    prototype_ids = torch.tensor(kmeans.labels_)
    prototype_list = torch.tensor(kmeans.cluster_centers_).float()
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std.float())
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FpnMlpFsCls_1(prototype_list, std, n_cls, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_f1 = torch.empty(0, 1).to(param_config.device)
    fpn_valid_f1 = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            f1 = F1Score(num_classes=n_cls).to(param_config.device)
            f1_train = f1(predicted_train, gnd_train.long().squeeze())
            fpn_train_f1 = torch.cat([fpn_train_f1, f1_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            f1_val = f1(predicted_val, gnd_val.long().squeeze())
            fpn_valid_f1 = torch.cat([fpn_valid_f1, f1_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        idx = fpn_model.fire_strength.max(1)[1]
        idx_unique = idx.unique(sorted=True)
        idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < f1_train:
        #     best_test_rslt = f1_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train f1 : {fpn_train_f1[-1, 0]}, test f1 : {fpn_valid_f1[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_f1, fpn_valid_f1


def fpn_reg(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FpnMlpFsCls_1(prototype_list, std, n_cls, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    fpn_train_mse = torch.empty(0, 1).to(param_config.device)
    fpn_valid_mse = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            mse_train = MSELoss().forward(outputs_train, gnd_train)
            fpn_train_mse = torch.cat([fpn_train_mse, mse_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            mse_val = MSELoss().forward(outputs_val, gnd_val)
            fpn_valid_mse = torch.cat([fpn_valid_mse, mse_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # if best_test_rslt < mse_train:
        #     best_test_rslt = mse_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train mse : {fpn_train_mse[-1, 0]}, test mse : {fpn_valid_mse[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_mse, fpn_valid_mse


def fpn_run_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = NRMSELoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = NRMSELoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training me of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test mse of test traditional FNN: {fnn_test_mse}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    # model: nn.Module = MLP(train_data.fea.shape[1])
    # rules = RuleKmeans()
    # rules.fit(train_data.fea, param_config.n_rules)
    prototype_list = rules.center_list
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    fpn_model: nn.Module = FpnMlpFsReg(prototype_list, param_config.device)
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])

    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    fpn_train_losses = []
    fpn_valid_losses = []

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
                      f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
                      f"k_{current_k}.pkl"
    # #load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, 1).to(param_config.device)
        outputs_val = torch.empty(0, 1).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            loss_train = NRMSELoss().forward(outputs_train, gnd_train)
            fpn_train_losses.append(loss_train.item())
            for i, (data, labels) in enumerate(valid_loader):
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            loss_val = NRMSELoss().forward(outputs_val, gnd_val)
            fpn_valid_losses.append(loss_val.item())
        param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        if best_test_rslt < loss_train:
            torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train loss : {fpn_train_losses[-1]}, test loss : {fpn_valid_losses[-1]}")

    # mlp model
    mlp_model: nn.Module = MlpReg(train_data.fea.shape[1], param_config.device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    mlp_train_losses = []
    mlp_valid_losses = []

    for epoch in range(epochs):
        mlp_model.train()

        for i, (data, labels) in enumerate(train_loader):
            outputs = mlp_model(data)
            # loss = loss_fn(outputs.double(), labels.double().squeeze(1))
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp_model.eval()
        outputs_train = torch.empty(0, 1).to(param_config.device)
        outputs_val = torch.empty(0, 1).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                outputs_temp = mlp_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            loss_train = loss_fn(outputs_train, gnd_train)
            mlp_train_losses.append(loss_train.item())
            for i, (data, labels) in enumerate(valid_loader):
                outputs_temp = mlp_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            loss_val = NRMSELoss().forward(outputs_val, gnd_val)
            mlp_valid_losses.append(loss_val.item())

        param_config.log.info(
            f"mlp epoch : {epoch + 1}, train loss : {mlp_train_losses[-1]}, test loss : {mlp_valid_losses[-1]}")

    param_config.log.info("finished")
    plt.figure(0)
    title = f"FPN MSE of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    # plt.plot(torch.arange(len(mlp_train_losses)), torch.tensor(mlp_train_losses), 'b--', linewidth=2, markersize=5)
    # plt.plot(torch.arange(len(mlp_valid_losses)), torch.tensor(mlp_valid_losses), 'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), fnn_train_mse.cpu().expand_as(torch.tensor(fpn_valid_losses)),
             'b--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), fnn_test_mse.cpu().expand_as(torch.tensor(fpn_valid_losses)),
             'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), torch.tensor(fpn_train_losses).cpu(), 'b:', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), torch.tensor(fpn_valid_losses).cpu(), 'r:', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_losses)), torch.tensor(mlp_train_losses).cpu(), 'b-.', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_losses)), torch.tensor(mlp_valid_losses).cpu(), 'r-.', linewidth=2,
             markersize=5)
    plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test', 'mlp train', 'mlp test'])
    # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
    # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
    # plt.legend(['fpn train', 'fpn test'])
    plt.show()

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_losses"] = torch.tensor(fpn_train_losses).numpy()
    save_dict["fpn_valid_losses"] = torch.tensor(fpn_valid_losses).numpy()
    save_dict["mlp_train_losses"] = torch.tensor(mlp_train_losses).numpy()
    save_dict["mlp_valid_losses"] = torch.tensor(mlp_valid_losses).numpy()
    save_dict["fnn_train_mse"] = fnn_train_mse.numpy()
    save_dict["fnn_test_mse"] = fnn_test_mse.numpy()
    data_save_file = f"{data_save_dir}/mse_bpfnn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_lr_{param_config.lr:.6f}" \
                     f"_k_{current_k}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_losses, fpn_valid_losses


# def fpn_run_cls_cov(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
#     """
#     todo: this is the method for fuzzy Neuron network using back propagation
#     :param param_config:
#     :param train_data: training dataset
#     :param test_data: test dataset
#     :param current_k: current k
#     :return:
#     """
#     h_computer = HNormal()
#     rules = RuleKmeans()
#     rules.fit(train_data.fea, param_config.n_rules)
#     h_train, _ = h_computer.comute_h(train_data.fea, rules)
#     # run FNN solver for given rule number
#     fnn_solver = FnnSolveReg()
#     fnn_solver.h = h_train
#     fnn_solver.y = train_data.gnd
#     fnn_solver.para_mu = 0.1
#     w_optimal = fnn_solver.solve().squeeze()
#
#     rules.consequent_list = w_optimal
#
#     n_rule_train = h_train.shape[0]
#     n_smpl_train = h_train.shape[1]
#     n_fea_train = h_train.shape[2]
#     h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
#     h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
#     y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())
#
#     fnn_train_mse = LikelyLoss().forward(train_data.gnd, y_train_hat)
#
#     h_test, _ = h_computer.comute_h(test_data.fea, rules)
#     n_rule_test = h_test.shape[0]
#     n_smpl_test = h_test.shape[1]
#     n_fea_test = h_test.shape[2]
#     h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
#     h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
#     y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())
#
#     fnn_test_mse = LikelyLoss().forward(test_data.gnd, y_test_hat)
#
#     param_config.log.info(f"Training f1 of traditional FNN: {fnn_train_mse}")
#     param_config.log.info(f"Test f1 of test traditional FNN: {fnn_test_mse}")
#
#     train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
#     valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)
#
#     train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
#     # model: nn.Module = MLP(train_data.fea.shape[1])
#     # rules = RuleKmeans()
#     # rules.fit(train_data.fea, param_config.n_rules)
#     prototype_list = rules.center_list
#     # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
#     # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
#     n_cls = train_data.gnd.unique().shape[0]
#     fpn_model: nn.Module = FpnCov1dFSCls(prototype_list, n_cls, param_config.device)
#     # fpn_model = fpn_model.cuda()
#     # initiate model parameter
#     # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
#     # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
#
#     optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
#     # loss_fn = nn.MSELoss()
#     loss_fn = nn.CrossEntropyLoss()
#     epochs = param_config.n_epoch
#
#     fpn_train_f1 = []
#     fpn_valid_f1 = []
#
#     data_save_dir = f"./results/{param_config.dataset_folder}"
#
#     if not os.path.exists(data_save_dir):
#         os.makedirs(data_save_dir)
#     model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
#                       f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
#                       f"k_{current_k}.pkl"
#     # #load the exist model
#     # if os.path.exists(model_save_file):
#     #     fpn_model.load_state_dict(torch.load(model_save_file))
#     best_test_rslt = 0
#     for epoch in range(epochs):
#         fpn_model.train()
#
#         for i, (data, labels) in enumerate(train_loader):
#             # data = data.cuda()
#             # labels = labels.cuda()
#             outputs_temp = fpn_model(data, True)
#             loss = loss_fn(outputs_temp, labels.squeeze().long())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         fpn_model.eval()
#         outputs_train = torch.empty(0, n_cls).to(param_config.device)
#         outputs_val = torch.empty(0, n_cls).to(param_config.device)
#
#         gnd_train = torch.empty(0, 1).to(param_config.device)
#         gnd_val = torch.empty(0, 1).to(param_config.device)
#         with torch.no_grad():
#             for i, (data, labels) in enumerate(train_loader):
#                 # data = data.cuda()
#                 # labels = labels.cuda()
#                 outputs_temp = fpn_model(data, False)
#                 outputs_train = torch.cat((outputs_train, outputs_temp), 0)
#                 gnd_train = torch.cat((gnd_train, labels), 0)
#             _, predicted_train = torch.max(outputs_train, 1)
#             correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
#             f1_train = correct_train_num/gnd_train.shape[0]
#             fpn_train_f1.append(f1_train)
#             for i, (data, labels) in enumerate(valid_loader):
#                 # data = data.cuda()
#                 # labels = labels.cuda()
#                 outputs_temp = fpn_model(data, False)
#                 outputs_val = torch.cat((outputs_val, outputs_temp), 0)
#                 gnd_val = torch.cat((gnd_val, labels), 0)
#             _, predicted_val = torch.max(outputs_val, 1)
#             correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
#             f1_val = correct_val_num/gnd_val.shape[0]
#             fpn_valid_f1.append(f1_val)
#         param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
#         if best_test_rslt < f1_train:
#             best_test_rslt = f1_train
#             torch.save(fpn_model.state_dict(), model_save_file)
#         param_config.log.info(
#             f"fpn epoch : {epoch + 1}, train f1 : {fpn_train_f1[-1]}, test f1 : {fpn_valid_f1[-1]}")
#
#     # mlp model
#     mlp_model: nn.Module = MlpCls(train_data.fea.shape[1], n_cls, param_config.device)
#     optimizer = torch.optim.Adam(mlp_model.parameters(), lr=param_config.lr)
#     loss_fn = nn.CrossEntropyLoss()
#     epochs = param_config.n_epoch
#
#     mlp_train_f1 = []
#     mlp_valid_f1 = []
#
#     for epoch in range(epochs):
#         mlp_model.train()
#
#         for i, (data, labels) in enumerate(train_loader):
#             # data = data.cuda()
#             # labels = labels.cuda()
#             outputs = mlp_model(data)
#             loss = loss_fn(outputs, labels.squeeze().long())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         mlp_model.eval()
#         outputs_train = torch.empty(0, n_cls).to(param_config.device)
#         outputs_val = torch.empty(0, n_cls).to(param_config.device)
#
#         gnd_train = torch.empty(0, 1).to(param_config.device)
#         gnd_val = torch.empty(0, 1).to(param_config.device)
#         with torch.no_grad():
#             for i, (data, labels) in enumerate(train_loader):
#                 # data = data.cuda()
#                 # labels = labels.cuda()
#                 outputs_temp = mlp_model(data)
#                 outputs_train = torch.cat((outputs_train, outputs_temp), 0)
#                 gnd_train = torch.cat((gnd_train, labels), 0)
#             _, predicted_train = torch.max(outputs_train, 1)
#             correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
#             f1_train = correct_train_num / gnd_train.shape[0]
#             mlp_train_f1.append(f1_train)
#             for i, (data, labels) in enumerate(valid_loader):
#                 # data = data.cuda()
#                 # labels = labels.cuda()
#                 outputs_temp = mlp_model(data)
#                 outputs_val = torch.cat((outputs_val, outputs_temp), 0)
#                 gnd_val = torch.cat((gnd_val, labels), 0)
#             _, predicted_val = torch.max(outputs_val, 1)
#             correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
#             f1_val = correct_val_num / gnd_val.shape[0]
#             mlp_valid_f1.append(f1_val)
#
#         param_config.log.info(
#             f"mlp epoch : {epoch + 1}, train f1 : {mlp_train_f1[-1]}, test f1 : {mlp_valid_f1[-1]}")
#
#     param_config.log.info("finished")
#     plt.figure(0)
#     title = f"FPN f1 of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel('f1')
#     # plt.plot(torch.arange(len(mlp_train_f1)), torch.tensor(mlp_train_f1), 'b--', linewidth=2, markersize=5)
#     # plt.plot(torch.arange(len(mlp_valid_f1)), torch.tensor(mlp_valid_f1), 'r--', linewidth=2, markersize=5)
#     plt.plot(torch.arange(len(fpn_valid_f1)), fnn_train_mse.cpu().expand_as(torch.tensor(fpn_valid_f1)),
#              'b--', linewidth=2, markersize=5)
#     plt.plot(torch.arange(len(fpn_valid_f1)), fnn_test_mse.cpu().expand_as(torch.tensor(fpn_valid_f1)),
#              'r--', linewidth=2, markersize=5)
#     plt.plot(torch.arange(len(fpn_valid_f1)), torch.tensor(fpn_train_f1).cpu(), 'b:', linewidth=2,
#              markersize=5)
#     plt.plot(torch.arange(len(fpn_valid_f1)), torch.tensor(fpn_valid_f1).cpu(), 'r:', linewidth=2,
#              markersize=5)
#     plt.plot(torch.arange(len(mlp_valid_f1)), torch.tensor(mlp_train_f1).cpu(), 'b-.', linewidth=2,
#              markersize=5)
#     plt.plot(torch.arange(len(mlp_valid_f1)), torch.tensor(mlp_valid_f1).cpu(), 'r-.', linewidth=2,
#              markersize=5)
#     plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test', 'mlp train', 'mlp test'])
#     # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
#     # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
#     # plt.legend(['fpn train', 'fpn test'])
#     plt.show()
#
#     # save all the results
#     save_dict = dict()
#     save_dict["fpn_train_f1"] = torch.tensor(fpn_train_f1).numpy()
#     save_dict["fpn_valid_f1"] = torch.tensor(fpn_valid_f1).numpy()
#     save_dict["mlp_train_f1"] = torch.tensor(mlp_train_f1).numpy()
#     save_dict["mlp_valid_f1"] = torch.tensor(mlp_valid_f1).numpy()
#     save_dict["fnn_train_mse"] = fnn_train_mse.cpu().numpy()
#     save_dict["fnn_test_mse"] = fnn_test_mse.cpu().numpy()
#     data_save_file = f"{data_save_dir}/mse_bpfnn_{param_config.dataset_folder}_rule" \
#                      f"_{param_config.n_rules}_lr_{param_config.lr:.6f}" \
#                      f"_k_{current_k}.mat"
#     io.savemat(data_save_file, save_dict)
#     return fpn_train_f1, fpn_valid_f1


def run_cmp_mthds(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # =============================svm===============================
    paras = dict()
    paras['kernel'] = 'rbf'
    svm_train_f1, svm_test_f1 = svc(train_data.fea.cpu(), test_data.fea.cpu(), train_data.gnd.cpu(),
                                      test_data.gnd.cpu(), LikelyLoss(), paras)
    svm_train_f1 = svm_train_f1.to(param_config.device).unsqueeze(0).unsqueeze(1)
    svm_test_f1 = svm_test_f1.to(param_config.device).unsqueeze(0).unsqueeze(1)
    param_config.log.info(f"f1uracy of training data using SVM: {svm_train_f1}")
    param_config.log.info(f"f1uracy of test data using SVM: {svm_test_f1}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_f1, fpn_valid_f1 = fpn_cls(param_config, train_data, train_loader, valid_loader)

    # ============different types of mlp models=========
    mlp_model: nn.Module = MlpCls121(train_data.n_fea, n_cls, param_config.device)
    mlp121_train_f1, mlp121_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp121")

    mlp_model: nn.Module = MlpCls421(train_data.n_fea, n_cls, param_config.device)
    mlp421_train_f1, mlp421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp421")

    mlp_model: nn.Module = MlpCls21(train_data.n_fea, n_cls, param_config.device)
    mlp21_train_f1, mlp21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp21")

    mlp_model: nn.Module = MlpCls212(train_data.n_fea, n_cls, param_config.device)
    mlp212_train_f1, mlp212_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp212")

    mlp_model: nn.Module = MlpCls42124(train_data.n_fea, n_cls, param_config.device)
    mlp42124_train_f1, mlp42124_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp42124")

    mlp_model: nn.Module = MlpCls12421(train_data.n_fea, n_cls, param_config.device)
    mlp12421_train_f1, mlp12421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp12421")

    # ============different types of CNN models===========
    mlp_model: nn.Module = CnnCls11(train_data.n_fea, n_cls, param_config.device)
    cnn11_train_f1, cnn11_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn11")

    mlp_model: nn.Module = CnnCls21(train_data.n_fea, n_cls, param_config.device)
    cnn21_train_f1, cnn21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn21")

    mlp_model: nn.Module = CnnCls12(train_data.n_fea, n_cls, param_config.device)
    cnn12_train_f1, cnn12_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn12")

    mlp_model: nn.Module = CnnCls22(train_data.n_fea, n_cls, param_config.device)
    cnn22_train_f1, cnn22_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn22")

    plt.figure(0)
    title = f"FPN f1 of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('f1')
    # plt.plot(torch.arange(len(mlp_train_f1)), torch.tensor(mlp_train_f1), 'b--', linewidth=2, markersize=5)
    # plt.plot(torch.arange(len(mlp_valid_f1)), torch.tensor(mlp_valid_f1), 'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), svm_train_f1.cpu().expand_as(fpn_valid_f1),
             'k-', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), svm_test_f1.cpu().expand_as(fpn_valid_f1),
             'k--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), fpn_train_f1.cpu(), 'r-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), fpn_valid_f1.cpu(), 'r--', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp12421_valid_f1)), mlp12421_train_f1.cpu(), 'b-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp12421_valid_f1)), mlp12421_valid_f1.cpu(), 'b--', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(cnn22_valid_f1)), cnn22_train_f1.cpu(), 'g-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(cnn22_valid_f1)), cnn22_valid_f1.cpu(), 'g--', linewidth=2,
             markersize=5)
    plt.legend(['svm train', 'svm test', 'fpn train', 'fpn test', 'mlp train', 'mlp test', 'cnn train', 'cnn test'])
    # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
    # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
    # plt.legend(['fpn train', 'fpn test'])
    plt.savefig(f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
                f"_nl_{param_config.noise_level}_k_{current_k+1}.pdf")
    # plt.show()

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_f1"] = fpn_train_f1.cpu().numpy()
    save_dict["fpn_valid_f1"] = fpn_valid_f1.cpu().numpy()
    save_dict["mlp121_train_f1"] = mlp121_train_f1.cpu().numpy()
    save_dict["mlp121_valid_f1"] = mlp121_valid_f1.cpu().numpy()
    save_dict["mlp421_train_f1"] = mlp421_train_f1.cpu().numpy()
    save_dict["mlp421_valid_f1"] = mlp421_valid_f1.cpu().numpy()
    save_dict["mlp21_train_f1"] = mlp21_train_f1.cpu().numpy()
    save_dict["mlp21_valid_f1"] = mlp21_valid_f1.cpu().numpy()
    save_dict["mlp212_train_f1"] = mlp212_train_f1.cpu().numpy()
    save_dict["mlp212_valid_f1"] = mlp212_valid_f1.cpu().numpy()
    save_dict["mlp42124_train_f1"] = mlp42124_train_f1.cpu().numpy()
    save_dict["mlp42124_valid_f1"] = mlp42124_valid_f1.cpu().numpy()
    save_dict["mlp12421_train_f1"] = mlp12421_train_f1.cpu().numpy()
    save_dict["mlp12421_valid_f1"] = mlp12421_valid_f1.cpu().numpy()

    save_dict["cnn11_train_f1"] = cnn11_train_f1.cpu().numpy()
    save_dict["cnn11_valid_f1"] = cnn11_valid_f1.cpu().numpy()
    save_dict["cnn12_train_f1"] = cnn12_train_f1.cpu().numpy()
    save_dict["cnn12_valid_f1"] = cnn12_valid_f1.cpu().numpy()
    save_dict["cnn21_train_f1"] = cnn21_train_f1.cpu().numpy()
    save_dict["cnn21_valid_f1"] = cnn21_valid_f1.cpu().numpy()
    save_dict["cnn22_train_f1"] = cnn22_train_f1.cpu().numpy()
    save_dict["cnn22_valid_f1"] = cnn22_valid_f1.cpu().numpy()

    save_dict["svm_train_f1"] = svm_train_f1.cpu().numpy()
    save_dict["svm_valid_f1"] = svm_test_f1.cpu().numpy()
    data_save_file = f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}" \
                     f"_k_{current_k+1}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_f1, fpn_valid_f1, \
        mlp121_train_f1, mlp421_train_f1, mlp21_train_f1, mlp12421_train_f1, mlp212_train_f1, mlp42124_train_f1, \
        mlp121_valid_f1, mlp421_valid_f1, mlp21_valid_f1, mlp12421_valid_f1, mlp212_valid_f1, mlp42124_valid_f1, \
        cnn11_train_f1, cnn12_train_f1, cnn21_train_f1, cnn22_train_f1, \
        cnn11_valid_f1, cnn12_valid_f1, cnn21_valid_f1, cnn22_valid_f1, \
        svm_train_f1, svm_test_f1


def run_fpn_model(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)

    # ============FPN models===========
    fpn_train_f1, fpn_valid_f1 = fpn_cls(param_config, train_data, train_loader, valid_loader)

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_f1"] = fpn_train_f1.cpu().numpy()
    save_dict["fpn_valid_f1"] = fpn_valid_f1.cpu().numpy()
    data_save_file = f"{data_save_dir}/f1_fpnl_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}" \
                     f"_k_{current_k+1}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_f1, fpn_valid_f1


def run_cmp_mthds_d(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # =============================svm===============================
    paras = dict()
    paras['kernel'] = 'rbf'
    svm_train_f1, svm_test_f1 = svc(train_data.fea.cpu(), test_data.fea.cpu(), train_data.gnd.cpu(),
                                      test_data.gnd.cpu(), LikelyLoss(), paras)
    svm_train_f1 = svm_train_f1.to(param_config.device).unsqueeze(0).unsqueeze(1)
    svm_test_f1 = svm_test_f1.to(param_config.device).unsqueeze(0).unsqueeze(1)
    param_config.log.info(f"f1uracy of training data using SVM: {svm_train_f1}")
    param_config.log.info(f"f1uracy of test data using SVM: {svm_test_f1}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============different types of mlp models=========
    mlp_model: nn.Module = MlpCls121D(train_data.n_fea, n_cls, param_config.device)
    mlp121_train_f1, mlp121_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp121d")

    mlp_model: nn.Module = MlpCls421D(train_data.n_fea, n_cls, param_config.device)
    mlp421_train_f1, mlp421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp421d")

    mlp_model: nn.Module = MlpCls21D(train_data.n_fea, n_cls, param_config.device)
    mlp21_train_f1, mlp21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp21d")

    mlp_model: nn.Module = MlpCls212D(train_data.n_fea, n_cls, param_config.device)
    mlp212_train_f1, mlp212_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp212d")

    mlp_model: nn.Module = MlpCls42124D(train_data.n_fea, n_cls, param_config.device)
    mlp42124_train_f1, mlp42124_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp42124d")

    mlp_model: nn.Module = MlpCls12421D(train_data.n_fea, n_cls, param_config.device)
    mlp12421_train_f1, mlp12421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp12421d")

    # ============different types of CNN models===========
    mlp_model: nn.Module = CnnCls11D(train_data.n_fea, n_cls, param_config.device)
    cnn11_train_f1, cnn11_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn11d")

    mlp_model: nn.Module = CnnCls21D(train_data.n_fea, n_cls, param_config.device)
    cnn21_train_f1, cnn21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn21d")

    mlp_model: nn.Module = CnnCls12D(train_data.n_fea, n_cls, param_config.device)
    cnn12_train_f1, cnn12_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn12d")

    mlp_model: nn.Module = CnnCls22D(train_data.n_fea, n_cls, param_config.device)
    cnn22_train_f1, cnn22_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn22d")

    # ============FPN models===========
    fpn_train_f1, fpn_valid_f1 = fpn_cls(param_config, train_data, train_loader, valid_loader)

    plt.figure(0)
    title = f"FPN f1 of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('f1')
    # plt.plot(torch.arange(len(mlp_train_f1)), torch.tensor(mlp_train_f1), 'b--', linewidth=2, markersize=5)
    # plt.plot(torch.arange(len(mlp_valid_f1)), torch.tensor(mlp_valid_f1), 'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), svm_train_f1.cpu().expand_as(fpn_valid_f1),
             'k-', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), svm_test_f1.cpu().expand_as(fpn_valid_f1),
             'k--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), fpn_train_f1.cpu(), 'r-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), fpn_valid_f1.cpu(), 'r--', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp12421_valid_f1)), mlp12421_train_f1.cpu(), 'b-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp12421_valid_f1)), mlp12421_valid_f1.cpu(), 'b--', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(cnn22_valid_f1)), cnn22_train_f1.cpu(), 'g-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(cnn22_valid_f1)), cnn22_valid_f1.cpu(), 'g--', linewidth=2,
             markersize=5)
    plt.legend(['svm train', 'svm test', 'fpn train', 'fpn test', 'mlp train', 'mlp test', 'cnn train', 'cnn test'])
    # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
    # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
    # plt.legend(['fpn train', 'fpn test'])
    plt.savefig(f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
                f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_f1"] = fpn_train_f1.cpu().numpy()
    save_dict["fpn_valid_f1"] = fpn_valid_f1.cpu().numpy()
    save_dict["mlp121_train_f1"] = mlp121_train_f1.cpu().numpy()
    save_dict["mlp121_valid_f1"] = mlp121_valid_f1.cpu().numpy()
    save_dict["mlp421_train_f1"] = mlp421_train_f1.cpu().numpy()
    save_dict["mlp421_valid_f1"] = mlp421_valid_f1.cpu().numpy()
    save_dict["mlp21_train_f1"] = mlp21_train_f1.cpu().numpy()
    save_dict["mlp21_valid_f1"] = mlp21_valid_f1.cpu().numpy()
    save_dict["mlp212_train_f1"] = mlp212_train_f1.cpu().numpy()
    save_dict["mlp212_valid_f1"] = mlp212_valid_f1.cpu().numpy()
    save_dict["mlp42124_train_f1"] = mlp42124_train_f1.cpu().numpy()
    save_dict["mlp42124_valid_f1"] = mlp42124_valid_f1.cpu().numpy()
    save_dict["mlp12421_train_f1"] = mlp12421_train_f1.cpu().numpy()
    save_dict["mlp12421_valid_f1"] = mlp12421_valid_f1.cpu().numpy()

    save_dict["cnn11_train_f1"] = cnn11_train_f1.cpu().numpy()
    save_dict["cnn11_valid_f1"] = cnn11_valid_f1.cpu().numpy()
    save_dict["cnn12_train_f1"] = cnn12_train_f1.cpu().numpy()
    save_dict["cnn12_valid_f1"] = cnn12_valid_f1.cpu().numpy()
    save_dict["cnn21_train_f1"] = cnn21_train_f1.cpu().numpy()
    save_dict["cnn21_valid_f1"] = cnn21_valid_f1.cpu().numpy()
    save_dict["cnn22_train_f1"] = cnn22_train_f1.cpu().numpy()
    save_dict["cnn22_valid_f1"] = cnn22_valid_f1.cpu().numpy()

    save_dict["svm_train_f1"] = svm_train_f1.cpu().numpy()
    save_dict["svm_valid_f1"] = svm_test_f1.cpu().numpy()
    data_save_file = f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}" \
                     f"_k_{current_k + 1}_d.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_f1, fpn_valid_f1, \
           mlp121_train_f1, mlp421_train_f1, mlp21_train_f1, mlp12421_train_f1, mlp212_train_f1, mlp42124_train_f1, \
           mlp121_valid_f1, mlp421_valid_f1, mlp21_valid_f1, mlp12421_valid_f1, mlp212_valid_f1, mlp42124_valid_f1, \
           cnn11_train_f1, cnn12_train_f1, cnn21_train_f1, cnn22_train_f1, \
           cnn11_valid_f1, cnn12_valid_f1, cnn21_valid_f1, cnn22_valid_f1, \
           svm_train_f1, svm_test_f1


def dgp_cmp_mthds(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for deep gaussian process
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============different types of mlp models=========
    X = train_loader.dataset.x
    y = train_loader.dataset.y.squeeze()
    Xu = torch.from_numpy(kmeans2(X.cpu().numpy(), 30)[0]).to(param_config.device)
    # computes the weight for mean function of the first layer;
    # it is PCA of X (from 784D to 30D).
    # _, _, V = np.linalg.svd(X.cpu().numpy(), full_matrices=False)
    # W = torch.from_numpy(V[:30, :])
    # n_fea = X.shape[1]
    # mean_fn = LinearT(n_fea, 30)
    # mean_fn.linear.weight.data = W
    # mean_fn.linear.weight.requires_grad_(False)
    # mean_fn.cuda()
    dgp_model: nn.Module = DeepGP21(X, y, train_data.n_fea, n_cls, Xu, param_config.device)
    dgp21_train_f1, dgp21_valid_f1 = dgp_cls(dgp_model, param_config, train_loader, valid_loader, "dgp121")
    # dgp_model: nn.Module = DeepGP121(train_data.n_fea, n_cls, param_config.device)
    # dgp121_train_f1, dgp121_valid_f1 = dgp_cls1(dgp_model, param_config, train_loader, valid_loader, "dgp121")
    #
    # dgp_model: nn.Module = DeepGP421(train_data.n_fea, n_cls, param_config.device)
    # dgp421_train_f1, dgp421_valid_f1 = dgp_cls1(dgp_model, param_config, train_loader, valid_loader, "dgp421")
    #
    # dgp_model: nn.Module = DeepGP21(train_data.n_fea, n_cls, param_config.device)
    # dgp21_train_f1, dgp21_valid_f1 = dgp_cls1(dgp_model, param_config, train_loader, valid_loader, "dgp21")
    #
    # dgp_model: nn.Module = DeepGP212(train_data.n_fea, n_cls, param_config.device)
    # dgp212_train_f1, dgp212_valid_f1 = dgp_cls1(dgp_model, param_config, train_loader, valid_loader, "dgp212")
    #
    # dgp_model: nn.Module = DeepGP42124(train_data.n_fea, n_cls, param_config.device)
    # dgp42124_train_f1, dgp42124_valid_f1 = dgp_cls1(dgp_model, param_config, train_loader, valid_loader, "dgp42124")
    #
    # dgp_model: nn.Module = DeepGP12421(train_data.n_fea, n_cls, param_config.device)
    # dgp12421_train_f1, dgp12421_valid_f1 = dgp_cls1(dgp_model, param_config, train_loader, valid_loader, "dgp12421")

    # save all the results
    save_dict = dict()
    # save_dict["dgp121_train_f1"] = dgp121_train_f1.cpu().numpy()
    # save_dict["dgp121_valid_f1"] = dgp121_valid_f1.cpu().numpy()
    # save_dict["dgp421_train_f1"] = dgp421_train_f1.cpu().numpy()
    # save_dict["dgp421_valid_f1"] = dgp421_valid_f1.cpu().numpy()
    save_dict["dgp21_train_f1"] = dgp21_train_f1.cpu().numpy()
    save_dict["dgp21_valid_f1"] = dgp21_valid_f1.cpu().numpy()
    # save_dict["dgp212_train_f1"] = dgp212_train_f1.cpu().numpy()
    # save_dict["dgp212_valid_f1"] = dgp212_valid_f1.cpu().numpy()
    # save_dict["dgp42124_train_f1"] = dgp42124_train_f1.cpu().numpy()
    # save_dict["dgp42124_valid_f1"] = dgp42124_valid_f1.cpu().numpy()
    # save_dict["dgp12421_train_f1"] = dgp12421_train_f1.cpu().numpy()
    # save_dict["dgp12421_valid_f1"] = dgp12421_valid_f1.cpu().numpy()

    data_save_file = f"{data_save_dir}/f1_dgp_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}" \
                     f"_k_{current_k + 1}.mat"
    io.savemat(data_save_file, save_dict)
    return dgp21_train_f1, dgp21_valid_f1
    # return dgp121_train_f1, dgp421_train_f1, dgp21_train_f1, dgp12421_train_f1, dgp212_train_f1, dgp42124_train_f1, \
    #        dgp121_valid_f1, dgp421_valid_f1, dgp21_valid_f1, dgp12421_valid_f1, dgp212_valid_f1, dgp42124_valid_f1


def gnia_cmp_mthds(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation using the Gaussian noise Injection in
    activate layer
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============different types of mlp models=========
    mlp_model: nn.Module = MlpCls121GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    mlp121_train_f1, mlp121_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp121")

    mlp_model: nn.Module = MlpCls421GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    mlp421_train_f1, mlp421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp421")

    mlp_model: nn.Module = MlpCls21GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    mlp21_train_f1, mlp21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp21")

    mlp_model: nn.Module = MlpCls212GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    mlp212_train_f1, mlp212_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp212")

    mlp_model: nn.Module = MlpCls42124GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    mlp42124_train_f1, mlp42124_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp42124")

    mlp_model: nn.Module = MlpCls12421GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    mlp12421_train_f1, mlp12421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp12421")

    # ============different types of CNN models===========
    mlp_model: nn.Module = CnnCls11GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    cnn11_train_f1, cnn11_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn11")

    mlp_model: nn.Module = CnnCls21GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    cnn21_train_f1, cnn21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn21")

    mlp_model: nn.Module = CnnCls12GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    cnn12_train_f1, cnn12_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn12")

    mlp_model: nn.Module = CnnCls22GNIA(train_data.n_fea, n_cls, param_config.device, param_config.gni_sigma)
    cnn22_train_f1, cnn22_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn22")


    # save all the results
    save_dict = dict()
    save_dict["mlp121_train_f1"] = mlp121_train_f1.cpu().numpy()
    save_dict["mlp121_valid_f1"] = mlp121_valid_f1.cpu().numpy()
    save_dict["mlp421_train_f1"] = mlp421_train_f1.cpu().numpy()
    save_dict["mlp421_valid_f1"] = mlp421_valid_f1.cpu().numpy()
    save_dict["mlp21_train_f1"] = mlp21_train_f1.cpu().numpy()
    save_dict["mlp21_valid_f1"] = mlp21_valid_f1.cpu().numpy()
    save_dict["mlp212_train_f1"] = mlp212_train_f1.cpu().numpy()
    save_dict["mlp212_valid_f1"] = mlp212_valid_f1.cpu().numpy()
    save_dict["mlp42124_train_f1"] = mlp42124_train_f1.cpu().numpy()
    save_dict["mlp42124_valid_f1"] = mlp42124_valid_f1.cpu().numpy()
    save_dict["mlp12421_train_f1"] = mlp12421_train_f1.cpu().numpy()
    save_dict["mlp12421_valid_f1"] = mlp12421_valid_f1.cpu().numpy()

    save_dict["cnn11_train_f1"] = cnn11_train_f1.cpu().numpy()
    save_dict["cnn11_valid_f1"] = cnn11_valid_f1.cpu().numpy()
    save_dict["cnn12_train_f1"] = cnn12_train_f1.cpu().numpy()
    save_dict["cnn12_valid_f1"] = cnn12_valid_f1.cpu().numpy()
    save_dict["cnn21_train_f1"] = cnn21_train_f1.cpu().numpy()
    save_dict["cnn21_valid_f1"] = cnn21_valid_f1.cpu().numpy()
    save_dict["cnn22_train_f1"] = cnn22_train_f1.cpu().numpy()
    save_dict["cnn22_valid_f1"] = cnn22_valid_f1.cpu().numpy()

    data_save_file = f"{data_save_dir}/f1_fpn_gnia_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}_sig_{param_config.gni_sigma}" \
                     f"_k_{current_k + 1}.mat"
    io.savemat(data_save_file, save_dict)
    return mlp121_train_f1, mlp421_train_f1, mlp21_train_f1, mlp12421_train_f1, mlp212_train_f1, mlp42124_train_f1, \
           mlp121_valid_f1, mlp421_valid_f1, mlp21_valid_f1, mlp12421_valid_f1, mlp212_valid_f1, mlp42124_valid_f1, \
           cnn11_train_f1, cnn12_train_f1, cnn21_train_f1, cnn22_train_f1, \
           cnn11_valid_f1, cnn12_valid_f1, cnn21_valid_f1, cnn22_valid_f1


def drop_cmp_mthds(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation using the Gaussian noise Injection in
    activate layer
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============different types of mlp models=========
    mlp_model: nn.Module = MlpCls121Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    mlp121_train_f1, mlp121_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp121")

    mlp_model: nn.Module = MlpCls421Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    mlp421_train_f1, mlp421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp421")

    mlp_model: nn.Module = MlpCls21Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    mlp21_train_f1, mlp21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp21")

    mlp_model: nn.Module = MlpCls212Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    mlp212_train_f1, mlp212_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp212")

    mlp_model: nn.Module = MlpCls42124Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    mlp42124_train_f1, mlp42124_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp42124")

    mlp_model: nn.Module = MlpCls12421Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    mlp12421_train_f1, mlp12421_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp12421")

    # ============different types of CNN models===========
    mlp_model: nn.Module = CnnCls11Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    cnn11_train_f1, cnn11_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn11")

    mlp_model: nn.Module = CnnCls21Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    cnn21_train_f1, cnn21_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn21")

    mlp_model: nn.Module = CnnCls12Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    cnn12_train_f1, cnn12_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn12")

    mlp_model: nn.Module = CnnCls22Drop(train_data.n_fea, n_cls, param_config.device, param_config.drop_rate)
    cnn22_train_f1, cnn22_valid_f1 = dnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn22")

    # save all the results
    save_dict = dict()
    save_dict["mlp121_train_f1"] = mlp121_train_f1.cpu().numpy()
    save_dict["mlp121_valid_f1"] = mlp121_valid_f1.cpu().numpy()
    save_dict["mlp421_train_f1"] = mlp421_train_f1.cpu().numpy()
    save_dict["mlp421_valid_f1"] = mlp421_valid_f1.cpu().numpy()
    save_dict["mlp21_train_f1"] = mlp21_train_f1.cpu().numpy()
    save_dict["mlp21_valid_f1"] = mlp21_valid_f1.cpu().numpy()
    save_dict["mlp212_train_f1"] = mlp212_train_f1.cpu().numpy()
    save_dict["mlp212_valid_f1"] = mlp212_valid_f1.cpu().numpy()
    save_dict["mlp42124_train_f1"] = mlp42124_train_f1.cpu().numpy()
    save_dict["mlp42124_valid_f1"] = mlp42124_valid_f1.cpu().numpy()
    save_dict["mlp12421_train_f1"] = mlp12421_train_f1.cpu().numpy()
    save_dict["mlp12421_valid_f1"] = mlp12421_valid_f1.cpu().numpy()

    save_dict["cnn11_train_f1"] = cnn11_train_f1.cpu().numpy()
    save_dict["cnn11_valid_f1"] = cnn11_valid_f1.cpu().numpy()
    save_dict["cnn12_train_f1"] = cnn12_train_f1.cpu().numpy()
    save_dict["cnn12_valid_f1"] = cnn12_valid_f1.cpu().numpy()
    save_dict["cnn21_train_f1"] = cnn21_train_f1.cpu().numpy()
    save_dict["cnn21_valid_f1"] = cnn21_valid_f1.cpu().numpy()
    save_dict["cnn22_train_f1"] = cnn22_train_f1.cpu().numpy()
    save_dict["cnn22_valid_f1"] = cnn22_valid_f1.cpu().numpy()

    data_save_file = f"{data_save_dir}/f1_fpn_drop_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}_drop_{param_config.drop_rate}" \
                     f"_k_{current_k + 1}.mat"
    io.savemat(data_save_file, save_dict)
    return mlp121_train_f1, mlp421_train_f1, mlp21_train_f1, mlp12421_train_f1, mlp212_train_f1, mlp42124_train_f1, \
           mlp121_valid_f1, mlp421_valid_f1, mlp21_valid_f1, mlp12421_valid_f1, mlp212_valid_f1, mlp42124_valid_f1, \
           cnn11_train_f1, cnn12_train_f1, cnn21_train_f1, cnn22_train_f1, \
           cnn11_valid_f1, cnn12_valid_f1, cnn21_valid_f1, cnn22_valid_f1


def bnn_cmp_mthds(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation using HMC Baysian NN
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============different types of mlp models=========
    mlp_model: nn.Module = MlpCls121(train_data.n_fea, n_cls, param_config.device)
    mlp121_train_f1, mlp121_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp121_bnn")

    mlp_model: nn.Module = MlpCls421(train_data.n_fea, n_cls, param_config.device)
    mlp421_train_f1, mlp421_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp421_bnn")

    mlp_model: nn.Module = MlpCls21(train_data.n_fea, n_cls, param_config.device)
    mlp21_train_f1, mlp21_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp21_bnn")

    mlp_model: nn.Module = MlpCls212(train_data.n_fea, n_cls, param_config.device)
    mlp212_train_f1, mlp212_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp212_bnn")

    mlp_model: nn.Module = MlpCls42124(train_data.n_fea, n_cls, param_config.device)
    mlp42124_train_f1, mlp42124_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp42124_bnn")

    mlp_model: nn.Module = MlpCls12421(train_data.n_fea, n_cls, param_config.device)
    mlp12421_train_f1, mlp12421_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "mlp12421_bnn")

    # ============different types of CNN models===========
    mlp_model: nn.Module = CnnCls11(train_data.n_fea, n_cls, param_config.device)
    cnn11_train_f1, cnn11_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn11_bnn")

    mlp_model: nn.Module = CnnCls21(train_data.n_fea, n_cls, param_config.device)
    cnn21_train_f1, cnn21_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn21_bnn")

    mlp_model: nn.Module = CnnCls12(train_data.n_fea, n_cls, param_config.device)
    cnn12_train_f1, cnn12_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn12_bnn")

    mlp_model: nn.Module = CnnCls22(train_data.n_fea, n_cls, param_config.device)
    cnn22_train_f1, cnn22_valid_f1 = bnn_cls(mlp_model, param_config, train_loader, valid_loader, "cnn22_bnn")

    # save all the results
    save_dict = dict()
    save_dict["mlp121_train_f1"] = mlp121_train_f1.cpu().numpy()
    save_dict["mlp121_valid_f1"] = mlp121_valid_f1.cpu().numpy()
    save_dict["mlp421_train_f1"] = mlp421_train_f1.cpu().numpy()
    save_dict["mlp421_valid_f1"] = mlp421_valid_f1.cpu().numpy()
    save_dict["mlp21_train_f1"] = mlp21_train_f1.cpu().numpy()
    save_dict["mlp21_valid_f1"] = mlp21_valid_f1.cpu().numpy()
    save_dict["mlp212_train_f1"] = mlp212_train_f1.cpu().numpy()
    save_dict["mlp212_valid_f1"] = mlp212_valid_f1.cpu().numpy()
    save_dict["mlp42124_train_f1"] = mlp42124_train_f1.cpu().numpy()
    save_dict["mlp42124_valid_f1"] = mlp42124_valid_f1.cpu().numpy()
    save_dict["mlp12421_train_f1"] = mlp12421_train_f1.cpu().numpy()
    save_dict["mlp12421_valid_f1"] = mlp12421_valid_f1.cpu().numpy()

    save_dict["cnn11_train_f1"] = cnn11_train_f1.cpu().numpy()
    save_dict["cnn11_valid_f1"] = cnn11_valid_f1.cpu().numpy()
    save_dict["cnn12_train_f1"] = cnn12_train_f1.cpu().numpy()
    save_dict["cnn12_valid_f1"] = cnn12_valid_f1.cpu().numpy()
    save_dict["cnn21_train_f1"] = cnn21_train_f1.cpu().numpy()
    save_dict["cnn21_valid_f1"] = cnn21_valid_f1.cpu().numpy()
    save_dict["cnn22_train_f1"] = cnn22_train_f1.cpu().numpy()
    save_dict["cnn22_valid_f1"] = cnn22_valid_f1.cpu().numpy()

    data_save_file = f"{data_save_dir}/f1_bnn_{param_config.inference}_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_nl_{param_config.noise_level}" \
                     f"_k_{current_k + 1}.mat"
    io.savemat(data_save_file, save_dict)
    return mlp121_train_f1, mlp421_train_f1, mlp21_train_f1, mlp12421_train_f1, mlp212_train_f1, mlp42124_train_f1, \
           mlp121_valid_f1, mlp421_valid_f1, mlp21_valid_f1, mlp12421_valid_f1, mlp212_valid_f1, mlp42124_valid_f1, \
           cnn11_train_f1, cnn12_train_f1, cnn21_train_f1, cnn22_train_f1, \
           cnn11_valid_f1, cnn12_valid_f1, cnn21_valid_f1, cnn22_valid_f1


def fpn_run_cls_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_f1, fpn_valid_f1 = fpn_cls(param_config, train_data, train_loader, valid_loader)

    plt.figure(0)
    title = f"FPN f1 of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('f1')
    plt.plot(torch.arange(len(fpn_valid_f1)), fpn_train_f1.cpu(), 'b-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_f1)), fpn_valid_f1.cpu(), 'r--', linewidth=2,
             markersize=5)
    plt.legend(['fpn train', 'fpn test'])
    plt.savefig(f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
                f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    plt.show()

    return fpn_train_f1, fpn_valid_f1


def fpn_run_reg_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_mse, fpn_valid_mse = fpn_reg(param_config, train_data, train_loader, valid_loader)

    plt.figure(0)
    title = f"FPN mse of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('mse')
    plt.plot(torch.arange(len(fpn_valid_mse)), fpn_train_mse.cpu(), 'r-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_mse)), fpn_valid_mse.cpu(), 'r--', linewidth=2,
             markersize=5)
    plt.legend(['fpn train', 'fpn test'])
    plt.savefig(f"{data_save_dir}/mse_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
                f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_mse, fpn_valid_mse


def fnn_cmp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, rule_list, current_k):
    """
    todo: this is the method for normal fuzzy Neuron network
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    train_f1 = torch.empty(0, 1).to(param_config.device)
    test_f1 = torch.empty(0, 1).to(param_config.device)
    # train_data.fea = train_data.fea.to(param_config.device)
    # train_data.gnd = train_data.fea.to(param_config.device)
    # test_data.fea = test_data.gnd.to(param_config.device)
    # test_data.gnd = test_data.gnd.to(param_config.device)
    for i in rule_list:
        train_f1_tmp, test_f1_tmp = fnn_cls(i, train_data, test_data)
        train_f1 = torch.cat([train_f1, train_f1_tmp.unsqueeze(0).unsqueeze(0)], 0)
        test_f1 = torch.cat([test_f1, test_f1_tmp.unsqueeze(0).unsqueeze(0)], 0)
        param_config.log.info(f"FNN: the {current_k + 1}-th fold: rule: {i}, train {train_f1_tmp}, test {test_f1_tmp}")

    return train_f1, test_f1


def gp_cmp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for normal Gaussian Process
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    train_f1 = torch.empty(0, 1)
    test_f1 = torch.empty(0, 1)
    length_scale_list = torch.linspace(0.1, 2, 10)
    for i in length_scale_list:
        train_f1_tmp, test_f1_tmp = gp_cls(train_data, test_data, i)
        train_f1 = torch.cat([train_f1, torch.tensor(train_f1_tmp).unsqueeze(0).unsqueeze(0)], 0)
        test_f1 = torch.cat([test_f1, torch.tensor(test_f1_tmp).unsqueeze(0).unsqueeze(0)], 0)

    param_config.log.info(f"GP: the {current_k}-th fold: train {train_f1}, test {test_f1}")
    return train_f1, test_f1


def run_fnn_fnn_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_f1, fpn_valid_f1 = fnn_fnn_mlp(param_config, train_data, train_loader, valid_loader)

    # plt.figure(0)
    # title = f"FPN f1 of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    # plt.title(title)
    # plt.xlabel('Epoch')
    # plt.ylabel('f1')
    # plt.plot(torch.arange(len(fpn_valid_f1)), fpn_train_f1.cpu(), 'b-', linewidth=2,
    #          markersize=5)
    # plt.plot(torch.arange(len(fpn_valid_f1)), fpn_valid_f1.cpu(), 'r--', linewidth=2,
    #          markersize=5)
    # plt.legend(['fpn train', 'fpn test'])
    # plt.savefig(f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
    #             f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_f1, fpn_valid_f1


def fnn_fnn_mlp(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    # prototype_ids, prototype_list = kmeans(
    #     X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
    #     device=torch.device(train_data.fea.device)
    # )
    kmeans = KMeans(n_clusters=param_config.n_rules, random_state=0).fit(train_data.fea.cpu())
    prototype_ids = torch.tensor(kmeans.labels_)
    prototype_list = torch.tensor(kmeans.cluster_centers_).float()
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FnnMlpFnnMlpIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_f1 = torch.empty(0, 1).to(param_config.device)
    fpn_valid_f1 = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            f1 = F1Score(num_classes=n_cls).to(param_config.device)
            f1_train = f1(predicted_train, gnd_train.long().squeeze())
            fpn_train_f1 = torch.cat([fpn_train_f1, f1_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            f1_val = f1(predicted_val, gnd_val.long().squeeze())
            fpn_valid_f1 = torch.cat([fpn_valid_f1, f1_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < f1_train:
        #     best_test_rslt = f1_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train f1 : {fpn_train_f1[-1, 0]}, test f1 : {fpn_valid_f1[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_f1, fpn_valid_f1


def run_fnn_fnn_fc(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    n_cls = train_data.gnd.unique().shape[0]

    # ============FPN models===========
    fpn_train_f1, fpn_valid_f1 = fnn_fnn_fc(param_config, train_data, train_loader, valid_loader)

    # plt.figure(0)
    # title = f"FPN f1 of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    # plt.title(title)
    # plt.xlabel('Epoch')
    # plt.ylabel('f1')
    # plt.plot(torch.arange(len(fpn_valid_f1)), fpn_train_f1.cpu(), 'b-', linewidth=2,
    #          markersize=5)
    # plt.plot(torch.arange(len(fpn_valid_f1)), fpn_valid_f1.cpu(), 'r--', linewidth=2,
    #          markersize=5)
    # plt.legend(['fpn train', 'fpn test'])
    # plt.savefig(f"{data_save_dir}/f1_fpn_{param_config.dataset_folder}_rule_{param_config.n_rules}"
    #             f"_nl_{param_config.noise_level}_k_{current_k + 1}.pdf")
    # plt.show()

    return fpn_train_f1, fpn_valid_f1


def fnn_fnn_fc(param_config: ParamConfig, train_data: Dataset, train_loader: DataLoader, valid_loader: DataLoader):
    """
        todo: this is the method for fuzzy Neuron network using kmeans,
        firing strength generating with mlp and consequent layer with mlp as well
        :param param_config:
        :param train_data: training dataset
        :param train_loader: training dataset
        :param valid_loader: test dataset
        :return:
    """
    prototype_ids, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean',
        device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # get the std of data x
    std = torch.empty((0, train_data.fea.shape[1])).to(train_data.fea.device)
    for i in range(param_config.n_rules):
        mask = prototype_ids == i
        cluster_samples = train_data.fea[mask]
        std_tmp = torch.sqrt(torch.sum((cluster_samples - prototype_list[i, :]) ** 2, 0) / torch.tensor(
            cluster_samples.shape[0]).float())
        # std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
        std = torch.cat((std, std_tmp.unsqueeze(0)), 0)
    std = torch.where(std < 10 ** -5,
                      10 ** -5 * torch.ones(param_config.n_rules, train_data.fea.shape[1]).to(param_config.device), std)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FnnFcFnnFCIni(prototype_list, std, n_cls, param_config.n_rules_fs, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])
    param_config.log.info("fpn epoch:=======================start===========================")
    n_para = sum(param.numel() for param in fpn_model.parameters())
    param_config.log.info(f'# generator parameters: {n_para}')
    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_f1 = torch.empty(0, 1).to(param_config.device)
    fpn_valid_f1 = torch.empty(0, 1).to(param_config.device)

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    # model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
    #                   f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
    #                   f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp, _ = fpn_model(data)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            f1 = F1Score(num_classes=n_cls).to(param_config.device)
            f1_train = f1(predicted_train, gnd_train.long().squeeze())
            fpn_train_f1 = torch.cat([fpn_train_f1, f1_train.unsqueeze(0).unsqueeze(1)], 0)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp, _ = fpn_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            f1_val = f1(predicted_val, gnd_val.long().squeeze())
            fpn_valid_f1 = torch.cat([fpn_valid_f1, f1_val.unsqueeze(0).unsqueeze(1)], 0)

        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        # idx = fpn_model.fire_strength.max(1)[1]
        # idx_unique = idx.unique(sorted=True)
        # idx_unique_count = torch.stack([(idx == idx_u).sum() for idx_u in idx_unique])
        # param_config.log.info(f"cluster index count of data:\n{idx_unique_count.data}")
        # if best_test_rslt < f1_train:
        #     best_test_rslt = f1_train
        #     torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train f1 : {fpn_train_f1[-1, 0]}, test f1 : {fpn_valid_f1[-1, 0]}")

    param_config.log.info("fpn epoch:=======================finished===========================")
    return fpn_train_f1, fpn_valid_f1

from utils.param_config import ParamConfig
from utils.loss_utils import RMSELoss, LikelyLoss
from models.model_run import run_fnn_fnn_mlp
import torch
import os
import scipy.io as io
import argparse


# Dataset configuration
# Dataset
parser = argparse.ArgumentParser()
parser.add_argument(
        "--d",
        type=str,
        default="mnist",
        help="the name of dataset",
    )
parser.add_argument(
        "--nl",
        type=float,
        default=None,
        help="the name of dataset",
    )
args = parser.parse_args()
# init the parameters statlib_calhousing_config
param_config = ParamConfig()
param_config.config_parse(f"{args.d}_config")

if args.nl is not None:
    param_config.noise_level = args.nl

param_config.log.info(f"dataset : {param_config.dataset_folder}")
param_config.log.info(f"rule number : {param_config.n_rules}")
param_config.log.info(f"rule number in FS layer : {param_config.n_rules_fs}")
param_config.log.info(f"batch_size : {param_config.n_batch}")
param_config.log.info(f"epoch_size : {param_config.n_epoch}")
param_config.log.info(f"noise_level : {param_config.noise_level}")



for i in torch.arange(len(param_config.dataset_list)):
    # load dataset
    dataset = param_config.get_dataset_mat(int(i))

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mispara_consq_bias_rsion: Regression=======")
        param_config.loss_fun = RMSELoss()

    fpn_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
    fpn_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)

    for kfold_idx in torch.arange(param_config.n_kfolds):
        param_config.log.war(f"=====k_fold: {kfold_idx + 1}=======")
        train_data, test_data = dataset.get_kfold_data(kfold_idx)

        noise_level = param_config.noise_level
        if noise_level > 0.0:
            element_num = train_data.n_smpl * train_data.n_fea

            noise_mean = torch.zeros(train_data.n_smpl, train_data.n_fea)
            noise_std = 0.8 * torch.ones(train_data.n_smpl, train_data.n_fea)
            noise = torch.normal(noise_mean, noise_std).to(param_config.device)
            # noise = torch.randn(train_data.n_smpl, train_data.n_fea).to(param_config.device)
            # element wise
            noise_num = int(noise_level * element_num)
            mask = torch.zeros(element_num, 1)
            mask[0:noise_num, :] = 1
            mask = mask[torch.randperm(element_num), :].view(train_data.n_smpl, train_data.n_fea)
            mask = mask == 1

            # train_data.fea[mask] = 0.1*noise[mask] + train_data.fea[mask]
            train_data.fea[mask] = noise[mask] + train_data.fea[mask]

            # # shuffle the samples
            # shuffle_idx = torch.randperm(train_data.n_smpl)
            # train_data.fea = train_data.fea[shuffle_idx, :]
            # train_data.gnd = train_data.gnd[shuffle_idx, :]

        fpn_train_acc, fpn_valid_acc = \
            run_fnn_fnn_mlp(param_config, train_data, test_data, kfold_idx)

        fpn_test_acc_tsr = torch.cat([fpn_test_acc_tsr, fpn_valid_acc], 1)
        fpn_train_acc_tsr = torch.cat([fpn_train_acc_tsr, fpn_train_acc], 1)
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    save_dict = dict()
    save_dict["fpn_test_acc_tsr"] = fpn_test_acc_tsr.cpu().numpy()
    save_dict["fpn_train_acc_tsr"] = fpn_train_acc_tsr.cpu().numpy()

    data_save_file = f"{data_save_dir}/acc_fnn_mlp_{param_config.dataset_folder}" \
                     f"_rule{param_config.n_rules}_nl_{param_config.noise_level}_epoch_{param_config.n_epoch}_only.mat"
    io.savemat(data_save_file, save_dict)

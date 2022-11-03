from utils.param_config import ParamConfig
from utils.loss_utils import RMSELoss, LikelyLoss
from models.model_run_acc import fnn_cmp
import torch
import os
import scipy.io as io


def run_fnn(param_config: ParamConfig):
    """
    todo: run the codes for gni model for activation layer
    :param param_config: parameter configurations
    :return:
    """
    # Dataset configuration
    param_config.log.info(f"dataset : {param_config.dataset_folder}")
    param_config.log.info(f"prototype number : {param_config.n_rules}")

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

        # dgp saving tensors
        rule_list = torch.arange(2, 100, 3)
        fnn_train_acc_tsr = torch.empty(rule_list.shape[0], 0).to(param_config.device)
        fnn_test_acc_tsr = torch.zeros(rule_list.shape[0], 0).to(param_config.device)

        for kfold_idx in torch.arange(param_config.n_kfolds):
            param_config.log.war(f"=====k_fold: {kfold_idx + 1}=======")
            train_data, test_data = dataset.get_kfold_data(kfold_idx)

            # n_smpl_cls = 10
            # n_cls = torch.unique(train_data.gnd).shape[0]
            # x_tmp = torch.empty(0, train_data.n_fea).to(param_config.device)
            # y_tmp = torch.empty(0, 1).to(param_config.device)
            # for gnd_idx in torch.arange(n_cls):
            #     idx_tmp, _ = torch.where(train_data.gnd.cpu() == gnd_idx)
            #     x_tmp = torch.cat([x_tmp, train_data.fea[idx_tmp[0:n_smpl_cls], :]], 0)
            #     y_tmp = torch.cat([y_tmp, train_data.gnd[idx_tmp[0:n_smpl_cls], :]], 0)
            # train_data.fea = x_tmp
            # train_data.gnd = y_tmp
            # train_data.n_smpl = train_data.fea.shape[0]

            # add random guassian noise
            # # shuffle the samples
            # shuffle_idx = torch.randperm(train_data.n_smpl)
            # train_data.fea = train_data.fea[shuffle_idx, :]
            # train_data.gnd = train_data.gnd[shuffle_idx, :]
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
                # # sample wise
                # noise_num = int(noise_level * train_data.n_smpl)
                # mask = torch.zeros(train_data.n_smpl, train_data.n_fea)
                # mask[0:noise_num, :] = 1
                # mask = mask[torch.randperm(train_data.n_smpl), :]
                # mask = mask == 1
                # train_data.fea[mask] = 0.1*noise[mask] + train_data.fea[mask]
                train_data.fea[mask] = noise[mask] + train_data.fea[mask]
                # noise = torch.empty(0, train_data.n_fea).to(param_config.device)
                # for instance_idx in torch.arange(int(noise_level*train_data.n_smpl)):
                #     # noise_tmp = torch.normal(torch.zeros(1, train_data.n_fea), 0.01 * torch.ones(1, train_data.n_fea)).to(
                #     #     param_config.device)
                #     noise_tmp = torch.normal(torch.zeros(1, train_data.n_fea).to(
                #         param_config.device), train_data.fea.std(0))
                #     noise = torch.cat([noise, noise_tmp], 0)
                #     train_data.fea[instance_idx, :] = train_data.fea[instance_idx, :] + noise_tmp

                # # shuffle the samples
                # shuffle_idx = torch.randperm(train_data.n_smpl)
                # train_data.fea = train_data.fea[shuffle_idx, :]
                # train_data.gnd = train_data.gnd[shuffle_idx, :]

            fnn_train_acc, fnn_valid_acc = fnn_cmp(param_config, train_data, test_data, rule_list, kfold_idx)

            fnn_test_acc_tsr = torch.cat([fnn_test_acc_tsr, fnn_valid_acc], 1)
            fnn_train_acc_tsr = torch.cat([fnn_train_acc_tsr, fnn_train_acc], 1)

        data_save_dir = f"./results/{param_config.dataset_folder}"

        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
        save_dict = dict()
        # save fnn results
        save_dict["fnn_test_acc_tsr"] = fnn_test_acc_tsr.cpu().numpy()
        save_dict["fnn_train_acc_tsr"] = fnn_train_acc_tsr.cpu().numpy()

        data_save_file = f"{data_save_dir}/acc_fnn_{param_config.dataset_folder}" \
                         f"_nl_{param_config.noise_level}" \
                         f"_all.mat"
        io.savemat(data_save_file, save_dict)

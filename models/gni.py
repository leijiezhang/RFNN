from utils.param_config import ParamConfig
from utils.loss_utils import RMSELoss, LikelyLoss
from models.model_run import gnia_cmp_mthds
import torch
import os
import scipy.io as io


def run_gnia(param_config: ParamConfig):
    """
    todo: run the codes for gni model for activation layer
    :param param_config: parameter configurations
    :return:
    """
    # Dataset configuration
    param_config.log.info(f"dataset : {param_config.dataset_folder}")
    param_config.log.info(f"device : {param_config.device}")
    param_config.log.info(f"prototype number : {param_config.n_rules}")
    param_config.log.info(f"batch_size : {param_config.n_batch}")
    param_config.log.info(f"epoch_size : {param_config.n_epoch}")
    param_config.log.info(f"noise level : {param_config.noise_level}")
    param_config.log.info(f"GUI sigma : {param_config.gni_sigma}")

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

        # mlp saving tensors
        mlp121_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        mlp121_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        mlp421_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        mlp421_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        mlp21_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        mlp21_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        mlp12421_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        mlp12421_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        mlp212_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        mlp212_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        mlp42124_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        mlp42124_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)

        # 1d-cnn saving tensors
        cnn11_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        cnn11_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        cnn12_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        cnn12_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        cnn21_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        cnn21_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
        cnn22_train_acc_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
        cnn22_test_acc_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)

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

            mlp121_train_acc, mlp421_train_acc, mlp21_train_acc, mlp12421_train_acc, mlp212_train_acc, mlp42124_train_acc, \
            mlp121_valid_acc, mlp421_valid_acc, mlp21_valid_acc, mlp12421_valid_acc, mlp212_valid_acc, mlp42124_valid_acc, \
            cnn11_train_acc, cnn12_train_acc, cnn21_train_acc, cnn22_train_acc, \
            cnn11_valid_acc, cnn12_valid_acc, cnn21_valid_acc, cnn22_valid_acc = \
                gnia_cmp_mthds(param_config, train_data, test_data, kfold_idx)

            # mlp saving tensors
            mlp121_test_acc_tsr = torch.cat([mlp121_test_acc_tsr, mlp121_valid_acc], 1)
            mlp121_train_acc_tsr = torch.cat([mlp121_train_acc_tsr, mlp121_train_acc], 1)
            mlp421_test_acc_tsr = torch.cat([mlp421_test_acc_tsr, mlp421_valid_acc], 1)
            mlp421_train_acc_tsr = torch.cat([mlp421_train_acc_tsr, mlp421_train_acc], 1)
            mlp21_test_acc_tsr = torch.cat([mlp21_test_acc_tsr, mlp21_valid_acc], 1)
            mlp21_train_acc_tsr = torch.cat([mlp21_train_acc_tsr, mlp21_train_acc], 1)
            mlp12421_test_acc_tsr = torch.cat([mlp12421_test_acc_tsr, mlp12421_valid_acc], 1)
            mlp12421_train_acc_tsr = torch.cat([mlp12421_train_acc_tsr, mlp12421_train_acc], 1)
            mlp212_test_acc_tsr = torch.cat([mlp212_test_acc_tsr, mlp212_valid_acc], 1)
            mlp212_train_acc_tsr = torch.cat([mlp212_train_acc_tsr, mlp212_train_acc], 1)
            mlp42124_test_acc_tsr = torch.cat([mlp42124_test_acc_tsr, mlp42124_valid_acc], 1)
            mlp42124_train_acc_tsr = torch.cat([mlp42124_train_acc_tsr, mlp42124_train_acc], 1)

            # 1d-cnn saving tensors
            cnn11_test_acc_tsr = torch.cat([cnn11_test_acc_tsr, cnn11_valid_acc], 1)
            cnn11_train_acc_tsr = torch.cat([cnn11_train_acc_tsr, cnn11_train_acc], 1)
            cnn21_test_acc_tsr = torch.cat([cnn21_test_acc_tsr, cnn21_valid_acc], 1)
            cnn21_train_acc_tsr = torch.cat([cnn21_train_acc_tsr, cnn21_train_acc], 1)
            cnn12_test_acc_tsr = torch.cat([cnn12_test_acc_tsr, cnn12_valid_acc], 1)
            cnn12_train_acc_tsr = torch.cat([cnn12_train_acc_tsr, cnn12_train_acc], 1)
            cnn22_test_acc_tsr = torch.cat([cnn22_test_acc_tsr, cnn22_valid_acc], 1)
            cnn22_train_acc_tsr = torch.cat([cnn22_train_acc_tsr, cnn22_train_acc], 1)

        data_save_dir = f"./results/{param_config.dataset_folder}"

        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
        save_dict = dict()
        # save mlp results
        save_dict["mlp121_test_acc_tsr"] = mlp121_test_acc_tsr.cpu().numpy()
        save_dict["mlp121_train_acc_tsr"] = mlp121_train_acc_tsr.cpu().numpy()
        save_dict["mlp421_test_acc_tsr"] = mlp421_test_acc_tsr.cpu().numpy()
        save_dict["mlp421_train_acc_tsr"] = mlp421_train_acc_tsr.cpu().numpy()
        save_dict["mlp21_test_acc_tsr"] = mlp21_test_acc_tsr.cpu().numpy()
        save_dict["mlp21_train_acc_tsr"] = mlp21_train_acc_tsr.cpu().numpy()
        save_dict["mlp12421_test_acc_tsr"] = mlp12421_test_acc_tsr.cpu().numpy()
        save_dict["mlp12421_train_acc_tsr"] = mlp12421_train_acc_tsr.cpu().numpy()
        save_dict["mlp212_test_acc_tsr"] = mlp212_test_acc_tsr.cpu().numpy()
        save_dict["mlp212_train_acc_tsr"] = mlp212_train_acc_tsr.cpu().numpy()
        save_dict["mlp42124_test_acc_tsr"] = mlp42124_test_acc_tsr.cpu().numpy()
        save_dict["mlp42124_train_acc_tsr"] = mlp42124_train_acc_tsr.cpu().numpy()

        # save 1d-cnn results
        save_dict["cnn11_test_acc_tsr"] = cnn11_test_acc_tsr.cpu().numpy()
        save_dict["cnn11_train_acc_tsr"] = cnn11_train_acc_tsr.cpu().numpy()
        save_dict["cnn12_test_acc_tsr"] = cnn12_test_acc_tsr.cpu().numpy()
        save_dict["cnn12_train_acc_tsr"] = cnn12_train_acc_tsr.cpu().numpy()
        save_dict["cnn22_test_acc_tsr"] = cnn22_test_acc_tsr.cpu().numpy()
        save_dict["cnn22_train_acc_tsr"] = cnn22_train_acc_tsr.cpu().numpy()
        save_dict["cnn21_test_acc_tsr"] = cnn21_test_acc_tsr.cpu().numpy()
        save_dict["cnn21_train_acc_tsr"] = cnn21_train_acc_tsr.cpu().numpy()

        data_save_file = f"{data_save_dir}/acc_fpn_gnia_{param_config.dataset_folder}" \
                         f"_rule{param_config.n_rules}_nl_{param_config.noise_level}_sig_{param_config.gni_sigma}" \
                         f"_epoch_{param_config.n_epoch}_all.mat"
        io.savemat(data_save_file, save_dict)

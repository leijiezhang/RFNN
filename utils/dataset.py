import torch
from utils.partition import PartitionStrategy
from utils.math_utils import mapminmax
from torch.utils.data import Dataset as Dataset_nn


class DatasetTorch(Dataset_nn):
    def __init__(self, x, y=None):
        super(DatasetTorch, self).__init__()
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x: torch.Tensor = self.x[index, :]
        y: torch.Tensor = self.y[index, :]
        return x, y


class DatasetTorchB(Dataset_nn):
    """
    for binary labels
    """
    def __init__(self, x, y=None):
        super(DatasetTorchB, self).__init__()
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y
        self.class_num = int(y.max())+1

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x: torch.Tensor = self.x[index, :]
        y: torch.Tensor = self.y[index, :]
        batch_size = 1
        y_one_hot = torch.zeros(batch_size, self.class_num)
        y_one_hot[0, int(y.cpu())] = 1
        return x, y_one_hot


class Dataset(object):
    """
        we suppose the data structure is X: N x D (N is the number of data samples and D is the data sample dimention)
        and the label set Y as: N x 1
    """
    def __init__(self, fea: torch.Tensor, gnd, task, name):
        """
        init the Dataset class
        :param fea: the features of data
        :param gnd: the ground true label for classification or regression task
        :param name: the name of data set
        :param task: R for regression C for classification
        """
        self.name = name
        self.task = task

        self.n_brunch = 0
        self.n_agents = 0

        # for normal data
        self.fea = fea
        self.gnd = gnd

        self.n_fea = fea.shape[1]
        self.n_smpl = fea.shape[0]

        # data sequance disorder
        self.shuffle = True

        # partition dataset into several test data and training data
        # centralized partition strategies
        self.partition: PartitionStrategy = PartitionStrategy()

    def set_partition(self, partition: PartitionStrategy):
        self.partition = partition

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def get_kfold_data(self, fold_idx=None):
        """
        todo:generate training dataset and test dataset by k-folds
        :param n_agents: the number of distributed agents
        :param n_brunch: the number of hierarchy brunches
        :param fold_idx:
        :return: 1st fold datasets for run by default or specified n fold runabel datasets
        """
        if fold_idx is not None:
            self.partition.set_current_folds(fold_idx)
        train_idx = self.partition.get_train_indexes()
        test_idx = self.partition.get_test_indexes()

        train_name = f"{self.name}_train"
        test_name = f"{self.name}_test"

        # if the dataset is like a eeg data, which has trails hold sample blocks
        if self.fea.shape.__len__() == 3:
            # reform training dataset
            train_data = Dataset(self.fea[train_idx[0], :, :], self.gnd[train_idx[0], :, :], self.task, train_name)
            gnd_train = torch.empty(0, 1).float()
            fea_train = torch.empty(0, self.fea.shape[2]).float()
            for ii in torch.arange(train_data.gnd.shape[0]):
                fea_train = torch.cat((fea_train, train_data.fea[ii]), 0)
                size_smpl_ii = train_data.fea[ii].shape[0]
                gnd_train_tmp = train_data.gnd[ii].repeat(size_smpl_ii, 1)
                gnd_train = torch.cat((gnd_train, gnd_train_tmp), 0)
            train_data = Dataset(fea_train, gnd_train, train_data.task, train_data.name)

            # reform test dataset
            test_data = Dataset(self.fea[test_idx[0], :], self.gnd[test_idx[0], :], self.task, test_name)
            gnd_test = torch.empty(0, 1).float()
            fea_test = torch.empty(0, self.fea.shape[2]).float()
            for ii in torch.arange(test_data.gnd.shape[0]):
                fea_test = torch.cat((fea_test, test_data.fea[ii]), 0)
                size_smpl_ii = test_data.fea[ii].shape[0]
                gnd_test_tmp = test_data.gnd[ii].repeat(size_smpl_ii, 1)
                gnd_test = torch.cat((gnd_test, gnd_test_tmp), 0)
            test_data = Dataset(fea_test, gnd_test, test_data.task, test_data.name)
        else:
            train_data = Dataset(self.fea[train_idx[0], :], self.gnd[train_idx[0], :], self.task, train_name)
            test_data = Dataset(self.fea[test_idx[0], :], self.gnd[test_idx[0], :], self.task, test_name)

        # normalize data
        fea_all = torch.cat([train_data.fea, test_data.fea], 0)
        fea_normalize = mapminmax(fea_all)
        # fea_normalize = fea_all
        train_data.fea = fea_normalize[:train_data.n_smpl]
        test_data.fea = fea_normalize[train_data.n_smpl:]

        return train_data, test_data

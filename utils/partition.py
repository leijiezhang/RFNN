import abc
import torch
from sklearn.model_selection import KFold


class PartitionStrategy(object):
    """
    the base class of partition
    """
    def __init__(self):
        self.current_fold = 0
        self.num_folds = 0

    @abc.abstractmethod
    def partition(self, y: torch.Tensor, random_state, is_shuffle):
        return self

    @abc.abstractmethod
    def get_train_indexes(self):
        train_idx = []
        num_train_idx = []
        return train_idx, num_train_idx

    @abc.abstractmethod
    def get_test_indexes(self):
        test_idx = []
        num_test_idx = []
        return test_idx, num_test_idx

    @abc.abstractmethod
    def get_description(self):
        d = []
        return d

    def get_num_folds(self):
        return self.num_folds

    def set_current_folds(self, cur_fold):
        self.current_fold = cur_fold


class KFoldPartition(PartitionStrategy):
    def __init__(self, k):
        super(KFoldPartition, self).__init__()
        self.K = k
        self.train_indexes = []
        self.test_indexes = []
        self.num_folds = k

    def partition(self, gnd: torch.Tensor, is_shuffle=True, random_state=None):
        n_smpl = gnd.shape[0]
        total_index = torch.arange(n_smpl)
        if is_shuffle:
            if random_state is not None:
                torch.random.manual_seed(random_state)

            total_index = torch.randperm(n_smpl)

        n_test_smpl = round(n_smpl / self.K)
        self.train_indexes = []
        self.test_indexes = []
        for i in torch.arange(self.K):
            test_start = n_test_smpl * i
            test_end = n_test_smpl * (i+1)

            test_index = total_index[test_start:(test_end-1)]

            if test_start != 0:
                train_index_part1 = total_index[0:test_start-1]
                train_index_part2 = total_index[test_end::]
                train_index = torch.cat([train_index_part1, train_index_part2], 0)
            else:
                train_index = total_index[test_end::]

            self.train_indexes.append(train_index)
            self.test_indexes.append(test_index)
        self.set_current_folds(0)

    def get_train_indexes(self):
        train_idx = self.train_indexes[self.current_fold]
        num_train_idx = train_idx.shape[0]
        return train_idx, num_train_idx

    def get_test_indexes(self):
        test_idx = self.test_indexes[self.current_fold]
        num_test_idx = test_idx.shape[0]
        return test_idx, num_test_idx

    def get_description(self):
        d = ('%i-fold partition', self.K)
        return d

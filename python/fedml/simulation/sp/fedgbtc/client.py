import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset, TensorDataset


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.share_training_data_dict = {}  # 盟主客户收获的联盟共享数据集
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, local_training_data, local_test_data, local_sample_number):
        # self.client_idx = client_idx  # 客户id是静态的，没必要重复更新其id
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        # self.model_trainer.set_id(client_idx)

    def update_share_data(self):
        # Step 1: 收集并合并所有数据集
        all_datasets = [self.local_training_data.dataset]
        for cid, (share_data, share_number) in self.share_training_data_dict.items():
            all_datasets.append(share_data)
            self.local_sample_number += share_number

        combined_dataset = ConcatDataset(all_datasets)
        # Step 2: 使用合并后的数据集创建新的DataLoader
        self.local_training_data = DataLoader(combined_dataset, batch_size=self.args.batch_size, shuffle=True,
                                              num_workers=self.args.num_workers)

    def get_shared_data(self, cid, share_training_data, share_number):
        self.share_training_data_dict[cid] = (share_training_data, share_number)

    def share_data_up(self, share_rate):  # 将共享的数据直接提取出来，而不是subset
        print(len(self.local_training_data.dataset))
        print(self.local_sample_number)
        # 确定共享的样本数量
        share_number = int(self.local_sample_number * share_rate)
        # 生成随机索引
        indices = np.arange(self.local_sample_number)
        np.random.shuffle(indices)
        share_indices = indices[:share_number]
        print(share_indices)
        # 初始化列表来存储选中的数据和标签
        shared_data_list = []
        shared_targets_list = []
        # 从原始dataset中抽取对应的数据和标签
        for idx in share_indices:
            data, target = self.local_training_data.dataset[idx]
            shared_data_list.append(data)
            shared_targets_list.append(target)
        # 将列表转换为张量
        shared_data = torch.stack(shared_data_list)
        shared_targets = torch.tensor(shared_targets_list, dtype=torch.long)
        # 创建新的TensorDataset
        shared_dataset = TensorDataset(shared_data, shared_targets)

        return shared_dataset, share_number


    def get_sample_number(self):
        return self.local_sample_number

    # 获取某class（第j个class）下的样本数
    def get_sample_class_number(self, j):
        class_count = 0
        for i, train_batch in enumerate(self.local_training_data):
            # 获取每个客户端的训练数据
            labels = train_batch[1]
            if self.args.dataset in ["fed_shakespeare"]:
                # 统计指定类别的样本数量
                class_count += torch.sum(torch.eq(labels, j)).detach().item()
            else:# 统计指定类别的样本数量
                class_count += sum(1 for label in labels if label == j)
        return class_count

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

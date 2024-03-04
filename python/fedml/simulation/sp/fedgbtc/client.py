import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset, DataLoader, ConcatDataset, TensorDataset
from torch.utils.data.dataloader import default_collate


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
        self.model_trainer.set_id(client_idx)  # 设置客户端id

    def update_local_dataset(self, local_training_data, local_test_data, local_sample_number):
        # self.client_idx = client_idx  # 客户id是静态的，没必要重复更新其id
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        # self.model_trainer.set_id(client_idx)

    def share_data_up(self, share_rate):  # 现在传入的只会是dataloader对象
        indices = np.random.permutation(len(self.local_training_data.dataset))[
                  :int(len(self.local_training_data.dataset) * share_rate)]
        shared_data_list = []
        shared_targets_list = []
        # 从原始dataset中抽取对应的数据和标签
        for idx in indices:
            data, target = self.local_training_data.dataset[idx]

            # 检查data是否为Tensor，如果不是，则尝试转换
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            # 检查target是否为Tensor，如果不是，则尝试转换
            if isinstance(target, (np.integer, int)):
                target = torch.tensor(target, dtype=torch.long)

            # 检查data是否为空
            if data.nelement() == 0:  # 或者使用data.numel() == 0
                print(f"Warning: Empty data item at index {idx}. Skipping this item.")
                continue
            # 将列表转换为张量
            shared_data_list.append(data)
            shared_targets_list.append(target)

        # 检查是否有数据被添加，以防所有项都为空
        if not shared_data_list or not shared_targets_list:
            raise ValueError("All shared data items are empty. Please check the dataset.")

        # 创建新的TensorDataset
        shared_data = torch.stack(shared_data_list)
        shared_targets = torch.stack(shared_targets_list)
        shared_dataset = TensorDataset(shared_data, shared_targets)

        return shared_dataset, len(shared_data_list)

    def get_shared_data(self, cid, share_training_data, share_number):
        if share_number > 0:  # 防止后面合并到空dataset
            self.share_training_data_dict[cid] = (share_training_data, share_number)

    def update_share_data(self):
        all_datasets = [self.local_training_data.dataset]
        total_samples = len(self.local_training_data.dataset)
        for cid, (share_data, share_number) in self.share_training_data_dict.items():
            all_datasets.append(share_data)
            total_samples += share_number
        combined_dataset = ConcatDataset(all_datasets)
        self.local_training_data = DataLoader(combined_dataset, batch_size=self.local_training_data.batch_size,
                                              shuffle=True, num_workers=self.local_training_data.num_workers,
                                              collate_fn=self.custom_collate_fn)  # 使用自定义的collate_fn
        self.local_sample_number = total_samples

    def custom_collate_fn(self, batch):
        data_list = []
        target_list = []
        for data, target in batch:
            # 确保数据是Tensor
            data_list.append(data if isinstance(data, torch.Tensor) else torch.tensor(data))
            target_list.append(target if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.long))
        # 对数据进行填充
        data_padded = pad_sequence(data_list, batch_first=True)
        # 对标签使用默认的collate处理（如果标签也是变长的，可能需要相似的处理）
        target_collated = default_collate(target_list)
        return data_padded, target_collated

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

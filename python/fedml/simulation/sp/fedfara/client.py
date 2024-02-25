import torch


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

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

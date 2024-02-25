import random

# 记录60个客户端每轮报告的估计上传时间，60*50的二维数组，并且初始化为全0
client_upload_time = [[0 for j in range(60)] for i in range(50)]
# 记录60个客户端每轮报告的估计聚合时间
client_aggregation_time = [[0 for j in range(60)] for i in range(50)]

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

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def estimate_upload_time(self, round_idx):
        # 生成的随机浮点数大概率在1.0附近波动，有较小的概率会产生0.05（附近）这样的较小值，或者10（附近）这样的较大值。
        # 此外，client_idx变量为1~60之间的某个整数，index越小越容易产生极端值。
        if client_upload_time[int(round_idx/5)][self.client_idx] != 0:
            return client_upload_time[int(round_idx/5)][self.client_idx]
        # 根据client_idx变量的值增加生成极端值的概率
        weight = 0.05 + 0.05 * (1 - self.client_idx / 60)
        # 生成一个随机浮点数，大概率在1.0附近波动，较小的概率为0.05或10
        upload_time = random.choices([1.0, 0.05, 10], weights=[1 - 2 * weight, weight, weight], k=1)[0]
        # 在1.0附近波动
        upload_time += random.uniform(-0.3, 0.3)
        # 确保生成的值在0.05和10之间
        upload_time = min(max(upload_time, 0.05), 10) / 10

        client_upload_time[int(round_idx/5)][self.client_idx] = upload_time
        # print("Upload time: %s" % str(client_upload_time))
        return upload_time

    def estimate_aggregation_time(self, round_idx):
        if client_aggregation_time[int(round_idx/5)][self.client_idx] != 0:
            return client_aggregation_time[int(round_idx/5)][self.client_idx]
        # 根据client_idx变量的值增加生成极端值的概率
        weight = 0.05 + 0.05 * (1 - self.client_idx / 60)
        # 生成一个随机浮点数，大概率在1.0附近波动，较小的概率为0.05或10
        aggregation_time = random.choices([1.0, 0.05, 10], weights=[1 - 2 * weight, weight, weight], k=1)[0]
        # 在1.0附近波动
        aggregation_time += random.uniform(-0.3, 0.3)
        # 确保生成的值在0.05和10之间
        aggregation_time = (min(max(aggregation_time, 0.05), 10)) / 20

        client_aggregation_time[int(round_idx/5)][self.client_idx] = aggregation_time
        # print("Aggregation time: %s" % str(client_aggregation_time))
        return aggregation_time

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

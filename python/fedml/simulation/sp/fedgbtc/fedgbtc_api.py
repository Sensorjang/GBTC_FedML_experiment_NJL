import copy
import logging
import random
from collections import OrderedDict
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
import openpyxl
import time


# 常超参数
interval_params = {
    'N': (20, 30),
    'D': (200, 1600),
    'miu': (15, 20),
    'P': (0.5, 1),
    'F': (0.05, 0.4),
    'Z': -114,
    'B': 10,
    'delta': 6272,
    'M': 3.4*10**6,
    'sigma': 1,
    'rho': 8,
    'I': 5,
    'lr': 0.01,
    'e': 1e-28,
    'd': (0, 100),
    'k': 10
}
quality_weights = {'cpu': 0.25, 'ram': 0.25, 'bm': 0.25, 'q': 0.25}

accuracy_list = []
loss_list = []
k_list = []
# 参数0
# 统计global_client_num_in_total个客户每个人的被选择次数
plt.figure(1, figsize=(16, 4))

# 创建一个新的Excel工作簿
wb = openpyxl.Workbook()
# 创建工作表
client_ws = wb.create_sheet('Clients Info')
# 写入损失指标的标头行
client_ws.append(['Round', 'ClientIdx', 'Loss', 'Accuracy', 'Time'])
# 创建工作表
round_ws = wb.create_sheet('Round Info')
# 写入精度指标的标头行
round_ws.append(['Round', 'Loss', 'Accuracy', 'Time', 'Selected Client Indexs', 'Total Selected Client Times', 'K'])
# 创建工作表
bid_quality_ws = wb.create_sheet('Bid and Quality Info')
# 写入精度指标的标头行
bid_quality_ws.append(['Round', 'ClientIdx', 'Bid', 'Quality Score'])
# 设置时间间隔（以秒为单位）
interval = 5


class AutoIncrementDict:
    def __init__(self):
        self._data = {}
        self._next_id = 0
    def add(self, value):
        self._data[self._next_id] = value
        current_id = self._next_id
        self._next_id += 1
        return current_id
    def remove(self, key):
        if key in self._data:
            del self._data[key]
    def get(self, key):
        return self._data.get(key, None)
    def __getitem__(self, key):
        return self.get(key)
    def __delitem__(self, key):
        self.remove(key)
    def __repr__(self):
        return repr(self._data)
    def items(self):
        return self._data.items()
    def keys(self):
        return self._data.keys()
    def values(self):
        return self._data.values()
    def remove_empty_values(self):
        keys_to_remove = [key for key, value in self._data.items() if not value]
        for key in keys_to_remove:
            del self._data[key]



class FedGBTCAPI(object):   #变量参考FARA代码
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.class_num = class_num
        self.client_num_in_total = np.random.randint(interval_params['N'])
        self.client_list = []  # 原始的客户集合（全部），不以联盟区分
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        # 乐享联盟参数
        self.k = interval_params['k']
        self.quality_weights = quality_weights
        self.client_params = {i: {} for i in range(self.client_num_in_total)}  # 存储每个客户的交互参数
        self.client_quality_list = {i: {'cpu': 0, 'ram': 0, 'bm': 0, 'q': 0} for i in range(self.client_num_in_total)}  # 存储客户的质量属性（每轮博弈结束后会更新）
        self.trust_matrix = self.create_trust_graph()  # 初始化直接生成信任矩阵，全局静态
        self.client_unions = {}   # 联盟结构（可能包含多个联盟，键为联盟主的id，值为联盟成员的id集合，暂不考虑联盟结构的变化）
        self.his_client_unions = {i: {} for i in range(self.client_num_in_total)}  # 历史联盟(客户历史所参与的,还没有盟主，字典存放联盟的id-客户对其偏好值)
        # 第二阶段博弈参数
        # 第一阶段博弈参数
        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )


    def params_generator(self):
        for key, value in interval_params.items():
            if key == 'sigma':
                rl = np.random.rayleigh(scale=value, size=self.client_num_in_total)
                for i, sigma_i in enumerate(rl):
                    self.client_params[i]['sigma'] = sigma_i
            elif key == 'rho':
                ln = lognorm(s=value, scale=self.client_num_in_total)
                for i, rho_i in enumerate(ln.rvs()):
                    self.client_params[i]['rho'] = rho_i
            if value.type == 'tuple':
                for i in range(self.client_num_in_total):
                    self.client_params[key] = np.random.rand(value)

    def create_trust_graph(self, value_range=(1, 10), distrust_probability=0.1):
        """
        修改后的函数，以一定的概率生成表示完全不信任的负无穷值。
        :param client_num_in_total: 客户总数，决定了信任图的大小
        :param value_range: 信任值的范围，为(min_value, max_value)
        :param distrust_probability: 完全不信任的概率
        :return: 带有表示不信任关系的随机信任图
        """
        shape = (self.client_num_in_total, self.client_num_in_total)
        min_value, max_value = value_range
        # 生成随机信任值数组
        trust_array = np.random.rand(*shape) * (max_value - min_value) + min_value
        # 以一定概率设置某些信任值为负无穷，表示完全不信任
        distrust_mask = np.random.rand(*shape) < distrust_probability
        trust_array[distrust_mask] = -np.inf
        return trust_array

    def upgrade_union_uid(self, union, uid_old, uid_new, nid=None):  # 新旧联盟的交替(通常发生在某客户离开/加入某联盟)
        for cid in union:  # 这里可选则将新增客户id传入,避免删除旧联盟id记录的操作
            if cid != nid:
                self.his_client_unions[cid].remove(uid_old)
            self.his_client_unions[cid][uid_new] = self.cal_preference(cid, union)
    def unions_formation(self):
        """
        生成初始联盟划分，同时避免联盟中存在信任值为负无穷的客户关系。
        """
        customer_ids = np.arange(self.client_num_in_total)
        np.random.shuffle(customer_ids)
        unions = AutoIncrementDict()
        # 初始化联盟字典（自动计数id）
        for customer_id in customer_ids:
            added = False
            for uid, members in unions.items():
                can_add = True
                for member in members:
                    if self.trust_matrix[customer_id, member] == -np.inf or self.trust_matrix[member, customer_id] == -np.inf:
                        can_add = False
                        break
                if can_add:
                    unions[uid].append(customer_id)
                    self.his_client_unions[customer_id].append(uid)
                    added = True
                    break
            if not added:
                new_uid = unions.add([customer_id])
                self.his_client_unions[customer_id].append(new_uid)
        unions.remove_empty_values() # 清除空联盟

        stable = False
        while not stable:
            stable = True  # 假设本轮次不再发生联盟变化
            for cid in range(self.client_num_in_total):  # 遍历每一位客户,初始将目前定义为最优
                uid_i = next(reversed(self.his_client_unions[cid].keys()))   # 找到当前客户i所在的联盟id
                best_pre_i = self.cal_preference(cid, unions[uid_i])  #
                best_uid_i = uid_i
                for uid, union in unions.items():
                    if uid == best_uid_i:  # 避免重新评估当前联盟
                        continue
                    else:
                        pre = self.cal_preference(cid, union)  # 计算客户对该联盟的偏好值
                        if pre > best_pre_i:
                            best_pre_i = pre
                            best_uid_i = uid
                            stable = False  # 如果发生了联盟变化，则本轮次不再稳定

                if best_uid_i != uid_i:  # 如果找到了更好的联盟,双更新,也要更新原来两个联盟中客户绑定的uid
                    # 更新旧联盟
                    union_former = unions[uid_i]
                    union_former.remove(cid)
                    unions.remove(uid_i)
                    new_uid_former = unions.add(union_former)
                    self.upgrade_union_uid(union_former, uid_i, new_uid_former)
                    # 更新新联盟
                    union_latter = unions[best_uid_i]
                    unions.remove(best_uid_i)
                    union_latter.append(cid)
                    new_uid_latter = unions.add(union_latter)
                    self.his_client_unions[cid].remove(uid_i)
                    self.upgrade_union_uid(union_latter, best_uid_i, new_uid_latter, cid)
                unions.remove_empty_values() # 清除空联盟
        # 返回稳定的联盟
        return unions

    def cal_client_quality(self, cid):  # 计算客户i的加权质量属性值
        score = 0.0
        for key, value in self.client_quality_list[cid].items():
            score += value * self.quality_weights[key]
        return score
    def cm_election(self, unions):  # 联盟主选举算法
        for uid, union in unions.items():
            best_score = -np.inf
            best_cid = -1
            for cid in union:
                score = self.cal_client_quality(cid)
                if score > best_score:
                    best_score = score
                    best_cid = cid
            if best_cid == -1:
                raise ValueError("No client in the union")
            # 从联盟中移除联盟主，准备更新联盟成员信息
            self.client_unions[best_cid] = [cid for cid in union if cid != best_cid]

    def cal_preference(self, i, union): # 客户对联盟的偏好函数,暂时不考虑历史联盟
        pre = 0.0
        for j in union:
            if self.trust_matrix[i][j] == -np.inf:
                pre = -np.inf
                break
            # elif
            else:
                pre += self.trust_matrix[i][j]
        return pre

    def cal_client_train_time(self, ):
    def cal_client_train_cost(self, ):

    def _setup_clients(self, train_data_local_num_dict,
                       train_data_local_dict, test_data_local_dict, model_trainer,):
        logging.info("############setup_clients (START)#############")
        # 参数1
        for client_idx in range(self.client_num_in_total):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)

        # 客户交互属性的生成
        self.params_generator()
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_unions = {}  # 暂存联盟主返回的模型，用于全局聚合， 以联盟主id为键
            g_weights = {}  # 暂存全局聚合的权重，以联盟主id为键

            # 联盟生成与最优化阶段
            unions = self.unions_formation()  # 初始联盟划分，并得到最优化的划分
            self.cm_election(unions)  # 联盟主选举，并得到完整的联盟划分
            for u_cid, union in self.client_unions.items():
                w_locals = []  # 暂存联盟内客户端返回的模型
                u_weights = []  # 暂存联盟内部聚合权重
                # 联盟主及其联盟成员训练
                for cid in union+[u_cid]:
                    client = self.client_list[cid]
                    client.update_local_dataset(
                        self.train_data_local_dict[cid],
                        self.test_data_local_dict[cid],
                        self.train_data_local_num_dict[cid]
                    )  # 加入该成员的聚合权重
                    u_weights.append(client.local_sample_number)
                    mlops.event("train", event_started=True,
                                event_value="{}_{}".format(str(round_idx), str(client.client_idx)))
                    w = client.train(copy.deepcopy(w_global))
                    mlops.event("train", event_started=False,
                                event_value="{}_{}".format(str(round_idx), str(client.client_idx)))
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w), client.client_idx))

                # 联盟内部聚合
                mlops.event("agg", event_started=True, event_value="轮次{}_联盟主{}".format(str(round_idx),str(u_cid)))
                w_unions[u_cid] = self._aggregate(w_locals, u_weights)
                mlops.event("agg", event_started=True, event_value="轮次{}_联盟主{}".format(str(round_idx),str(u_cid)))

            # 主从博弈求解与支付、带宽分配阶段




            # 全局聚合
            mlops.event("agg", event_started=True, event_value="轮次{}_全局".format(str(round_idx)))
            w_global = self._aggregate(list(w_unions.values()), list(g_weights.values()))
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value="轮次{}_全局".format(str(round_idx)))

            # test results
            # at last round
            train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
            if round_idx == self.args.comm_round - 1:
                train_acc, train_loss, test_acc, test_loss = self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    train_acc, train_loss, test_acc, test_loss = self._local_test_on_all_clients(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

            # round_ws.append([round_idx,
            #                     train_loss,
            #                     train_acc,
            #                     time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            #                     str(client_indexes),
            #                     str(client_selected_times),
            #                     k])

            # 保存Excel文件到self.args.excel_save_path+文件名
            wb.save(self.args.excel_save_path + self.args.model + "_[" + self.args.dataset +"]_GBTC_training_results_NIID"+ str(self.args.experiment_niid_level) +".xlsx")
            # 休眠一段时间，以便下一个循环开始前有一些时间
            time.sleep(interval)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_list, banned_client_indexs):
        # round 0 select all, then select client which is not banned
        if round_idx == 0:
            client_indexes = [client.client_idx for client in client_list]
        else:
            client_indexes = [client.client_idx for client in client_list if
                              client.client_idx not in banned_client_indexs]

        # if client_num_in_total == client_num_per_round:
        #     client_indexes = [client_index for client_index in range(client_num_in_total)]
        # else:
        #     num_clients = min(client_num_per_round, client_num_in_total)
        #     np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        #     client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals, eweights):
        # training_num = 0
        # for idx in range(len(w_locals)):
        #     (sample_num, averaged_params) = w_locals[idx]
        #     training_num += sample_num
        print("_aggregate.w_locals.length = " + str(len(w_locals)) + "\n" + "_aggregate.eweights.length = " + str(
            len(eweights)))

        eweights_sum = sum(eweights)

        (sample_num, averaged_params, _) = w_locals[0]
        for k in averaged_params.keys():
            print("______k = " + str(k))
            for i in range(0, len(w_locals)):
                # local_sample_number, local_model_params = w_locals[i]
                local_model_params = w_locals[i][1]
                local_wight_number = eweights[i]

                w = local_wight_number / eweights_sum

                # print("w = " + str(w))

                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_resnet(self, w_locals, eweights): # 弃用
        eweights_sum = sum(eweights)

        averaged_params = OrderedDict()
        for (_, local_model_params), w in zip(w_locals, eweights):
            for key, param in local_model_params.items():
                if key not in averaged_params:
                    averaged_params[key] = []

                    # 获取对应权重
                local_wight = w / eweights_sum

                # 根据类型分别计算加权平均
                if 'conv' in key:
                    averaged_params[key] += local_wight * param.clone().detach()
                elif 'bn' in key:
                    averaged_params[key] += local_wight * param.clone().detach()
                elif 'fc' in key:
                    averaged_params[key] += local_wight * param.clone().detach()

        for key in averaged_params:
            averaged_params[key] /= eweights_sum

        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))


            client_ws.append([round_idx,
                              client_idx,
                              train_metrics["losses"][client_idx] / train_metrics["num_samples"][client_idx],
                              train_metrics["num_correct"][client_idx] / train_metrics["num_samples"][client_idx],
                              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())])

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}

        # 绘制精度图
        accuracy_list.append(train_acc)
        loss_list.append(train_loss)
        plot_accuracy_and_loss(self, round_idx)

        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        return train_acc, train_loss, test_acc, test_loss

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)


# 定义绘制精度图的函数
def plot_accuracy_and_loss(self, round_idx):
    plt.ion()

    print("accuracy_list: ", accuracy_list)
    print("loss_list: ", loss_list)

    plt.clf()
    plt.suptitle("FedGBTC" + "_[" + self.args.dataset +"]_NIID"+ str(self.args.experiment_niid_level))

    # 第1个子图
    plt.subplot(2, 2, 1)
    plt.title("accuracy")
    plt.xlabel("num of epoch")
    plt.ylabel("value of accuracy")
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, 'b-', linewidth=2)

    # 第2个子图
    plt.subplot(2, 2, 2)
    plt.title("loss")
    plt.xlabel("num of epoch")
    plt.ylabel("value of loss")
    plt.plot(range(1, len(loss_list) + 1), loss_list, 'b-', linewidth=2)

    # 第3个子图
    plt.subplot(2, 2, 3)
    plt.title("k")
    plt.xlabel("num of epoch")
    plt.ylabel("value of k")
    plt.plot(range(1, len(k_list) + 1), k_list, 'b-', linewidth=2)

    # 第4个子图，使用条形图展示每个客户的被选择次数
    plt.subplot(2, 2, 4)
    plt.title("num of selected")
    plt.xlabel("num of epoch")
    plt.ylabel("value of num of selected")
    plt.bar(range(1, len(client_selected_times) + 1), client_selected_times, width=0.5, fc='b')

    plt.tight_layout()
    plt.pause(0.005)
    plt.ioff()

    if (round_idx == self.args.comm_round - 1):
        plt.show()

    return

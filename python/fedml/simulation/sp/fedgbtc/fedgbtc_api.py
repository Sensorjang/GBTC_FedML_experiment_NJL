import copy
import logging
import random
from collections import OrderedDict
from scipy.optimize import minimize
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
    'N': (20, 30),  # 客户端数量范围
    'D': (200, 1600),  # 数据集大小范围
    'miu': (15, 20),  # 处理单位样本所需的CPU周期数
    'p': (0.5, 1),  # 客户端发射功率
    'f': (0.05, 0.4),  # 客户端的cpu频率范围
    'Z': -114,  # 噪声Z0
    'B': 10,  # 总上行带宽B
    'delta': 6272,  # 单位数据样本大小（每个样本的物理大小/bit）
    'M': 3.4 * 10 ** 6,  # 模型大小
    'sigma': 1,  # 小尺度衰落，瑞利分布的方差
    'rho': 8,  # 阴影衰落， 对数正态分布的标准差
    'I': 5,  # 本地训练epoch大小
    'batch_size': 10,  # 本地训练批大小
    'lr': 0.01,  # 本地训练学习率
    'e': 1e-28,  # 计算芯片组有效电容参数
    'd': (0, 0.08),  # 客户端与联邦服务器的物理距离/km
    'k': 10,  # 初始联盟的个数
    'omega': 900,  # 联盟主对支付成本与训练时间的权衡系数,
    'lambda_cm': (1, 2),  # 客户单位通信开销范围
    'lambda_cp': (1, 2),  # 客户单位计算开销范围
    'trust_threshold': 0.5,  # 联盟信任阈值（低于阈值的剔除）
}
quality_weights = {'cpu': 0.25, 'ram': 0.25, 'bm': 0.25, 'q': 0.25}
quality_ranges = {'cpu': (1, 5), 'ram': (1, 5), 'bm': (1, 5)}
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


class FedGBTCAPI(object):  # 变量参考FARA代码
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
        self.client_num_in_total = np.random.randint(low=interval_params['N'][0], high=interval_params['N'][1] + 1)
        self.client_list = []  # 原始的客户集合（全部），不以联盟区分
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        # 乐享联盟参数
        self.common_params = {}
        self.quality_weights = quality_weights
        self.client_data_distribution = {i: {} for i in range(self.client_num_in_total)}  # 存储每个客户的数据概率分布
        self.client_params = {i: {} for i in range(self.client_num_in_total)}  # 存储每个客户的交互参数
        self.client_quality_list = {  # 存储客户的特征向量（质量属性）
            i: {'cpu': np.random.randint(low=quality_ranges['cpu'][0], high=quality_ranges['cpu'][1] + 1),
                'ram': np.random.randint(low=quality_ranges['ram'][0], high=quality_ranges['ram'][1] + 1),
                'bm': np.random.randint(low=quality_ranges['bm'][0], high=quality_ranges['bm'][1] + 1), 'q': 0.0}
            for i in range(self.client_num_in_total)}
        self.trust_matrix = None  # 初始化直接生成信任矩阵，全局静态
        self.client_unions = {}  # 稳定的联盟结构（可能包含多个联盟，键为联盟主的id，值为联盟成员的id集合，暂不考虑联盟结构的变化）
        self.time_unions = {}  # 记录每轮每个稳定联盟的整体训练时间，以盟主id为键
        self.imp_unions = {}  # 记录每个联盟对不同资源的重视程度(元组字典)
        # 历史联盟(客户历史所参与的,还没有盟主，字典存放联盟的id-客户对其偏好值)
        self.his_client_unions = {i: {} for i in range(self.client_num_in_total)}
        self.client_banned_List = {}  # 存储被剔除的客户（键为id、值为 1. 不成盟 2. 累计信任值不足 3. 均衡解为0）
        self.client_rewards = {}  # 存储每个客户的奖励值

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def params_generator(self):  # 客户独立参数的赋值
        for key, value in interval_params.items():
            if key == 'sigma':
                rl = np.random.rayleigh(scale=value, size=self.client_num_in_total)
                print(rl)
                for i, sigma_i in enumerate(rl):
                    self.client_params[i]['sigma'] = sigma_i
            elif key == 'rho':
                ln = lognorm(s=value)
                for i, rho_i in enumerate(ln.rvs(size=self.client_num_in_total)):
                    self.client_params[i]['rho'] = rho_i
            elif isinstance(value, tuple) and key != 'N':
                for i in range(self.client_num_in_total):
                    if key == 'D':
                        self.client_params[i][key] = int(np.random.uniform(*value))
                    else:
                        self.client_params[i][key] = np.random.uniform(*value)
            else:
                self.common_params[key] = value
        self.args.learning_rate = interval_params['lr']
        self.args.batch_size = interval_params['batch_size']
        self.args.epochs = interval_params['I']

    def create_trust_graph(self, value_range=(1, 10), distrust_probability=0.01):
        """
        修改后的函数，以一定的概率生成表示完全不信任的负无穷值，并将对角线元素设置为0。
        :param value_range: 信任值的范围，为(min_value, max_value)
        :param distrust_probability: 完全不信任的概率
        :return: 带有表示不信任关系的随机信任图，对角线元素为0
        """
        shape = (self.client_num_in_total, self.client_num_in_total)
        min_value, max_value = value_range
        # 生成随机信任值数组
        trust_array = np.random.rand(*shape) * (max_value - min_value) + min_value
        # 以一定概率设置某些信任值为负无穷，表示完全不信任
        distrust_mask = np.random.rand(*shape) < distrust_probability
        trust_array[distrust_mask] = -np.inf
        # 将对角线元素设置为0
        np.fill_diagonal(trust_array, 0)
        return trust_array

    def upgrade_union_uid(self, union, uid_old, uid_new, nid=None):  # 新旧联盟的交替(通常发生在某客户离开/加入某联盟，会产生新的联盟，此时需要修改其余成员的记录)
        for cid in union:  # 这里可选则将新增客户id传入,避免删除旧联盟id记录的操作
            if cid != nid:
                self.his_client_unions[cid].pop(uid_old)
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
                    if (self.trust_matrix[customer_id, member] == -np.inf or
                            self.trust_matrix[member, customer_id] == -np.inf):
                        can_add = False
                        break
                if can_add:
                    unions[uid].append(customer_id)
                    self.his_client_unions[customer_id][uid] = None  # 此时联盟还未稳定，先不计算偏好值
                    added = True
                    break
            if not added:
                new_uid = unions.add([customer_id])
                self.his_client_unions[customer_id][new_uid] = None  # 此时联盟还未稳定，先不计算偏好值
        unions.remove_empty_values()  # 清除空联盟
        # print(unions)

        stable = False
        while not stable:
            stable = True  # 假设本轮次不再发生联盟变化
            for cid in range(self.client_num_in_total):  # 遍历每一位客户,初始将目前定义为最优
                uid_i = next(reversed(self.his_client_unions[cid].keys()))  # 找到当前客户i所在的联盟id
                best_pre_i = self.cal_preference(cid, unions[uid_i])  # 先计算客户对当前联盟的偏好值
                best_uid_i = uid_i
                for uid, union in unions.items():
                    if uid == best_uid_i:  # 避免重新评估当前联盟
                        continue
                    else:
                        pre = self.cal_preference(cid, union) \
                            if uid not in self.his_client_unions[cid] else 0  # 计算客户对该新联盟的偏好值（此时需要考虑历史联盟的问题）
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
                    self.his_client_unions[cid].pop(uid_i)
                    self.upgrade_union_uid(union_latter, best_uid_i, new_uid_latter, cid)
                unions.remove_empty_values()  # 清除空联盟

        # 新增代码：在最终联盟确定后，清除只有一个客户的联盟
        for uid, union in unions.items():  # 使用list来避免在遍历时修改字典
            if len(union) == 1:
                self.client_banned_List[union[0]] = 1  # 因为不成盟被剔除
                unions.remove(uid)
            else:
                for cid in union:  # 清除社会信任不足阈值的成员
                    if self.cal_trust_sum(union, cid) < self.common_params['trust_threshold']:
                        self.his_client_unions[cid].pop(uid)
                        self.client_banned_List[cid] = 2  # 因为社会信任不足被剔除
                        union.remove(cid)

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
            self.time_unions[best_cid] = {}  # 初始化每个稳定联盟的时间记录容器，以round_idx为键
            imp_compute = np.random.rand(0, 1)
            imp_data = 1 - imp_compute
            self.imp_unions[best_cid] = (imp_compute, imp_data)

    def cal_preference(self, i, union):  # 客户对联盟的偏好函数,暂时不考虑历史联盟 (计算时不考虑历史参与)
        pre = 0.0
        for j in union:
            if self.trust_matrix[i][j] == -np.inf:
                pre = -np.inf
                break
            # elif
            else:
                pre += self.trust_matrix[i][j]
        return pre

    def cal_trust_sum(self, union, cid):
        return np.sum([self.trust_matrix[i][cid] for i in union])

    def cal_data_quality(self, u_cid):  # 计算联盟内成员的数据质量(这的数据质量不太一样，是按类别在整体的权熵，不是EMD)
        entropy = {}  # 暂存每位客户数据概率分布的熵
        label_num_per_client = {}  # 统计联盟内每个客户的每个类别的样本量
        for cid in self.client_unions[u_cid]:
            label_num_per_client[cid] = []
            total_sample_num = self.client_list[cid].get_sample_number()
            for j in range(self.class_num):
                this_sample_num = self.client_list[cid].get_sample_class_number(j)
                label_num_per_client[cid].append((this_sample_num + 1)
                                                 / total_sample_num + self.class_num)
        # 计算每位客户的类别熵权
        for cid in self.client_unions[u_cid]:
            sum_entropy = 0
            for j in range(self.class_num):
                sum_entropy -= label_num_per_client[cid][j] * np.log(label_num_per_client[cid][j])
            entropy[cid] = sum_entropy
        sum_entropy = sum([(1 - entropy) for entropy in list(entropy.values())])
        wi = {cid: (1 - entropy) / sum_entropy for cid, entropy in entropy.items()}
        # 计算并记录每位客户的数据质量
        for cid, eweight in wi.items():
            self.client_quality_list[cid]['q'] = eweight * sum(label_num_per_client[cid])

    def cal_client_utility(self, u_cid, bandwidth_vector, payment_vector):  # 计算联盟内所有客户的效用(传入初始分配默认以顺序作为成员索引)
        union = self.client_unions[u_cid]
        utilities = {}
        for i, cid in enumerate(union):
            pay = payment_vector[i]
            band_per = bandwidth_vector[i]
            g_i = self.client_params[cid]['rho'] * self.client_params[cid]['sigma'] * (
                    128.1 + 37.6 * np.log2(self.client_params[cid]['d']))
            c_cm = (self.client_params[cid]['lambda_cm'] * self.client_params[cid]['p'] *
                    self.common_params['M'] / (band_per * self.common_params['B'] * np.log2(1 + g_i *
                                                                                            self.client_params[cid][
                                                                                                'p'] /
                                                                                            self.common_params['Z'])))
            c_cp = (self.client_params[cid]['lambda_cp'] * self.client_params[cid]['miu'] * self.common_params['delta']
                    * self.client_params[cid]['D'] * self.common_params['I']
                    * self.common_params['e'] * self.client_params[cid]['f'] ** 2)
            utilities[cid] = (self.imp_unions[u_cid][0] * self.client_params[cid]['f'] + self.imp_unions[u_cid][1]
                              * self.client_quality_list[cid]['q'] * pay - c_cm - c_cp)
            return utilities

    def ms_obj(self, x, uid):  # 优化变量分别为：支付向量、带宽分配向量，整合到一个决策变量中(便于优化器)
        n = len(x) // 2
        price = x[:n]  # 解耦出两类决策变量
        band = x[n:2 * n]
        # print(band)
        t_cm = []
        t_cp = []
        values = []
        for i, cid in enumerate(self.client_unions[uid]):
            t_cp.append(self.client_params[cid]['miu'] * self.common_params['delta'] * self.client_params[cid]['D']
                        * self.common_params['I'] * self.common_params['e'] / self.client_params[cid]['f'])
            g_i = self.client_params[cid]['rho'] * self.client_params[cid]['sigma'] * (
                    128.1 + 37.6 * np.log2(self.client_params[cid]['d']))
            t_cm.append(self.common_params['M'] / (band[i] * self.common_params['B'] * np.log2(1 + g_i *
                                                                                               self.client_params[cid][
                                                                                                   'p'] /
                                                                                               self.common_params[
                                                                                                   'Z'])))
            data_value = self.client_quality_list[cid]['q'] * self.imp_unions[uid][1]
            cp_value = self.imp_unions[uid][0] ** 2 * price[i] / (2 * self.client_params[cid]['lambda_cp']
                                                                  * self.client_params[cid]['miu'] * self.common_params[
                                                                      'delta']
                                                                  * self.client_params[cid]['D'] * self.common_params[
                                                                      'I'] * self.common_params['e'])
            values.append((data_value + cp_value) * price[i])
        t = t_cm + t_cp
        t_max = np.max(t)
        obj = np.sum(values) + t_max * self.common_params['omega']
        return obj

    def _calculate_flag_max(self, uid, cid, f_min):
        flag_max = (2 * self.client_params[cid]['lambda_cp'] * self.client_params[cid]['miu']
                    * self.common_params['delta'] * self.client_params[cid]['D']
                    * self.common_params['I'] * f_min / self.imp_unions[uid][0])
        return flag_max

    def _calculate_flag_min(self, uid, cid, f_max):
        flag_min = (2 * self.client_params[cid]['lambda_cp'] * self.client_params[cid]['miu']
                    * self.common_params['delta'] * self.client_params[cid]['D']
                    * self.common_params['I'] * f_max / self.imp_unions[uid][0])
        return flag_min

    def ms_cons(self, x, uid):
        n = len(x) // 2
        price = x[:n]  # 解耦出价格决策变量
        p_min = np.min(price)
        p_max = np.max(price)
        f_min = np.min([self.client_params[cid]['f'] for cid in self.client_unions[uid]])
        f_max = np.max([self.client_params[cid]['f'] for cid in self.client_unions[uid]])
        price_max = []
        price_min = []
        for cid in self.client_unions[uid]:
            flag_max = self._calculate_flag_max(uid, cid, f_min)
            flag_min = self._calculate_flag_min(uid, cid, f_max)
            price_max.append(max(p_min, flag_max))
            price_min.append(min(p_max, flag_min))
        price_max = np.array(price_max)
        price_min = np.array(price_min)  # 大于/小于，没有等号的话，需要加入一个微小的正数
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[n:2 * n]) - 1},  # 带宽占比的和为1
            {'type': 'ineq', 'fun': lambda x: np.min(x[:n] - price_min)},  # 每个单位支付大于等于最小值
            {'type': 'ineq', 'fun': lambda x: np.max(price_max - x[:n])},  # 每个单位支付小于等于最大值
            {'type': 'ineq', 'fun': lambda x: np.min(x[n:2 * n] - 1e-8)},  # 带宽占比大于0
            {'type': 'ineq', 'fun': lambda x: np.min(1 - x[n:2 * n] - 1e-8)}  # 带宽占比小于1
        ]
        return cons

    def ms_game_solution(self, u_cid):  # 主从博弈最优求解（按盟主id来）不用计算时间开销，问题定义的比较清晰，只用带入
        # 先生成一组随机分配(支付向量、带宽分配比向量)
        pay_vector, band_vector = [], []
        member_count = 0
        for _ in self.client_unions[u_cid]:
            band_vector.append(np.random.rand())
            pay_vector.append(np.random.uniform(1, 10))
            member_count += 1
        band_vector = np.array(pay_vector)
        band_vector = band_vector / np.sum(band_vector)
        pay_min = min(pay_vector)
        pay_max = max(pay_vector)
        f_max = interval_params['f'][1]
        f_min = interval_params['f'][0]
        pay_vector = [np.random.uniform(max(pay_min, self._calculate_flag_max(u_cid, cid, f_min)),
                                        min(pay_max, self._calculate_flag_min(u_cid, cid, f_max))) for cid in
                      self.client_unions[u_cid]]
        # 第二阶段：求出初始分配下，成员的均衡解-计算每位成员此时的效用，根据效用取值决定其均衡解
        # 先计算联盟中每个成员的数据质量：
        self.cal_data_quality(u_cid)
        # 然后计算每位客户此时的效用
        utility_per_client = self.cal_client_utility(u_cid, band_vector, pay_vector)
        strategy_per_client = {}  # 每位成员最终的付出值(二阶段的均衡解)
        for cid, u in utility_per_client.items():  # 根据当前效用值来确定每位成员的均衡解
            if u <= 0:
                strategy_per_client[cid] = 0.0
            else:
                f_flag = (self.imp_unions[u_cid][0] ** 2 * pay_vector[cid] / 2 * self.client_params[cid][
                    'lambda_cp']
                          * self.client_params[cid]['miu'] * self.common_params['delta'] * self.client_params[cid][
                              'D'] * self.common_params['I'] * self.common_params['e'])
                if f_flag >= f_max:
                    strategy_per_client[cid] = f_max
                elif f_flag <= f_min:
                    strategy_per_client[cid] = f_min
                else:
                    strategy_per_client[cid] = f_flag
        # 第一阶段，求出当前盟主的均衡解, 调用minimize函数时传入额外参数
        x0 = np.concatenate((pay_vector, band_vector))
        print(x0)
        solution = minimize(
            fun=self.ms_obj,
            x0=x0,
            args=(u_cid,),
            method='SLSQP',  # 这里为什么报类型错误不是很清楚，官方文档里写的可以字典或者字典列表
            constraints=self.ms_cons(x0, u_cid),
            tol=1e-3,
            options={'disp': True}
        )
        # 检查解决方案
        if solution.success:
            print("Optimal solution for union of {} found.".format(u_cid))
            # 最优解
            optimal_x = solution.x
            pay_vector = optimal_x[:member_count]
            band_vector = optimal_x[member_count:2 * member_count]
            # 目标函数的最优值
            optimal_fun = solution.fun
        else:
            print("Optimal solution for union of {} failed.".format(u_cid))
        print(strategy_per_client)
        for i, cid in enumerate(self.client_unions[u_cid]):  # 每位成员最终的分配值(一阶段的均衡解)
            if strategy_per_client[cid] > 0:
                self.client_rewards[cid] = (pay_vector[i], band_vector[i])
            else:
                self.client_rewards[cid] = (0, 0)
                self.client_banned_List[cid] = 3  # 因为均衡解为0被剔除

    def _setup_clients(self, train_data_local_num_dict,
                       train_data_local_dict, test_data_local_dict, model_trainer, ):
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
        print(self.client_params)
        print(self.common_params)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        mlops.event("信任图生成", event_started=True)
        self.trust_matrix = self.create_trust_graph()  # 客户交互图生成
        print(self.trust_matrix)
        mlops.event("信任图生成", event_started=False)

        mlops.event("联盟生成", event_started=True)
        unions = self.unions_formation()  # 初始联盟划分，并得到最优化的划分
        # print(unions)
        mlops.event("联盟生成", event_started=False)

        mlops.event("联盟主选举", event_started=True)
        self.cm_election(unions)  # 联盟主选举，并得到完整的联盟划分
        # print(self.client_unions)
        mlops.event("联盟主选举", event_started=False)

        for u_cid, union in self.client_unions.items():  # 先把两阶段的东西全部弄完，主要是淘汰掉失信客户
            mlops.event("主从博弈求解", event_started=True, event_value="盟主：{}".format(str(u_cid)))
            self.ms_game_solution(u_cid)  # 完成主从博弈最优求解（求出带宽、支付分配，及客户淘汰）
            mlops.event("主从博弈求解", event_started=False, event_value="盟主：{}".format(str(u_cid)))
        print(self.client_rewards)
        print(self.client_banned_List)
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_unions = {}  # 暂存联盟主返回的模型，用于全局聚合， 以联盟主id为键
            g_weights = {}  # 暂存全局聚合的权重，以联盟主id为键
            for u_cid, union in self.client_unions.items():
                time_s = time.time()
                w_locals = []  # 暂存联盟内客户端返回的模型
                u_weights = []  # 暂存联盟内部聚合权重
                # 联盟主及其联盟成员训练
                for cid in union + [u_cid]:
                    band_id_list = list(self.client_banned_List.keys())
                    if cid in band_id_list:  # 排除不参与训练的客户
                        continue
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
                mlops.event("agg", event_started=True, event_value="轮次{}_联盟主{}".format(str(round_idx), str(u_cid)))
                w_unions[u_cid] = self._aggregate(w_locals, u_weights)
                mlops.event("agg", event_started=True, event_value="轮次{}_联盟主{}".format(str(round_idx), str(u_cid)))
                time_e = time.time()  # 记录每个联盟的整体执行时间
                self.time_unions[u_cid][round_idx] = time_e - time_s

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
            wb.save(
                self.args.excel_save_path + self.args.model + "_[" + self.args.dataset + "]_GBTC_training_results_NIID" + str(
                    self.args.experiment_niid_level) + ".xlsx")
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

    def _aggregate_resnet(self, w_locals, eweights):  # 弃用
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
    plt.suptitle("FedGBTC" + "_[" + self.args.dataset + "]_NIID" + str(self.args.experiment_niid_level))

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

    # # 第4个子图，使用条形图展示每个客户的被选择次数
    # plt.subplot(2, 2, 4)
    # plt.title("num of selected")
    # plt.xlabel("num of epoch")
    # plt.ylabel("value of num of selected")
    # plt.bar(range(1, len(client_selected_times) + 1), client_selected_times, width=0.5, fc='b')

    plt.tight_layout()
    plt.pause(0.005)
    plt.ioff()

    if (round_idx == self.args.comm_round - 1):
        plt.show()

    return

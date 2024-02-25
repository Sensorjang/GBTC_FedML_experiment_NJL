import json
import os
import random

import numpy as np
import wget
from ...ml.engine import ml_engine_adapter

cwd = os.getcwd()

import zipfile

from ...constants import FEDML_DATA_MNIST_URL
import logging


def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir, "MNIST.zip")
    logging.info(file_path)

    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = sorted(cdata["users"])

    return clients, groups, train_data, test_data


def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size, device_id, train_path="MNIST_mobile", test_path="MNIST_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_mnist(batch_size, train_path, test_path)

def load_partition_data_mnist(args, batch_size, train_path=os.path.join(os.getcwd(), "MNIST", "train"),
                              test_path=os.path.join(os.getcwd(), "MNIST", "test")):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]

    # NIID可控逻辑

    # Compute the number of data points in each client
    client_data_num = {}
    for u in users:
        client_data_num[u] = len(train_data[u]["x"])

    # Sort the clients by data point count in descending order
    # sorted_clients = sorted(client_data_num, key=client_data_num.get, reverse=True)

    # 计算排序的数据量
    sort_size = int(len(client_data_num) * args.experiment_niid_level)

    # 对前70%的数据进行排序
    sorted_clients = sorted(client_data_num, key=client_data_num.get, reverse=True)[:sort_size]

    # 对剩下的30%的数据进行随机排序
    unsorted_clients = list(set(client_data_num.keys()) - set(sorted_clients))
    random.shuffle(unsorted_clients)

    # 随机交叉合并排序后的数据和未排序的数据
    result = []
    num_sorted = 0
    num_unsorted = 0
    while num_sorted < len(sorted_clients) and num_unsorted < len(unsorted_clients):
        if random.random() < 0.5:
            result.append(sorted_clients[num_sorted])
            num_sorted += 1
        else:
            result.append(unsorted_clients[num_unsorted])
            num_unsorted += 1

    # 将剩下的数据添加到结果中
    result += sorted_clients[num_sorted:] + unsorted_clients[num_unsorted:]

    # NIID可控逻辑^^^^

    # Initialize the variables to keep track of the data
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()

    logging.info("loading data...")
    for client_idx, client_id in enumerate(result):
        g = groups[users.index(client_id)]
        user_train_data_num = client_data_num[client_id]
        user_test_data_num = len(test_data[client_id]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[client_id], batch_size)
        test_batch = batch_data(args, test_data[client_id], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch

    logging.info("finished the loading data")

    client_num = client_idx
    class_num = 10

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
#
# def load_partition_data_mnist(
#     args, batch_size, train_path=os.path.join(os.getcwd(), "MNIST", "train"),
#         test_path=os.path.join(os.getcwd(), "MNIST", "test")
# ):
#     users, groups, train_data, test_data = read_data(train_path, test_path)
#
#     if len(groups) == 0:
#         groups = [None for _ in users]
#     train_data_num = 0
#     test_data_num = 0
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()
#     train_data_local_num_dict = dict()
#     train_data_global = list()
#     test_data_global = list()
#     client_idx = 0
#     logging.info("loading data...")
#     for u, g in zip(users, groups):
#         user_train_data_num = len(train_data[u]["x"])
#         user_test_data_num = len(test_data[u]["x"])
#         train_data_num += user_train_data_num
#         test_data_num += user_test_data_num
#         train_data_local_num_dict[client_idx] = user_train_data_num
#
#         # transform to batches
#         train_batch = batch_data(args, train_data[u], batch_size)
#         test_batch = batch_data(args, test_data[u], batch_size)
#
#         # index using client index
#         train_data_local_dict[client_idx] = train_batch
#         test_data_local_dict[client_idx] = test_batch
#         train_data_global += train_batch
#         test_data_global += test_batch
#         client_idx += 1
#     logging.info("finished the loading data")
#     client_num = client_idx
#     class_num = 10
#
#     return (
#         client_num,
#         train_data_num,
#         test_data_num,
#         train_data_global,
#         test_data_global,
#         train_data_local_num_dict,
#         train_data_local_dict,
#         test_data_local_dict,
#         class_num,
#     )

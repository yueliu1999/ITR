import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader


class GroupDataset(object):
    def __init__(self, user_path, group_path, num_negatives, dataset="Mafengwo"):
        print(f"[{dataset.upper()}] loading...")
        self.num_negatives = num_negatives

        if dataset == "MafengwoS":
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt", num_users=11026, num_items=1235)
        else:
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.user_test_negatives = load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} "
              f"interactions, sparsity: {(1-len(self.user_train_matrix.keys()) / self.num_users / self.num_items):.5f}")

        self.group_train_matrix = load_rating_file_to_matrix(group_path + "Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_test_negatives = load_negative_file(group_path + "Negative.txt")
        self.num_groups, self.num_group_net_items = self.group_train_matrix.shape
        self.group_member_dict = load_group_member_to_dict(f"./data/{dataset}/groupMember.txt")

        print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, spa"
              f"rsity: {(1-len(self.group_train_matrix.keys()) / self.num_groups / self.group_train_matrix.shape[1]):.5f}")

        self.user_hyper_graph, self.item_hyper_graph, self.full_hg, group_data = build_hyper_graph(
            self.group_member_dict, group_path + "Train.txt", self.num_users, self.num_items, self.num_groups)
        self.overlap_graph = build_group_graph(group_data, self.num_groups)
        self.light_gcn_graph = build_light_gcn_graph(self.group_train_matrix, self.num_groups, self.num_group_net_items)
        print(f"\033[0;30;43m{dataset.upper()} finish loading!\033[0m", end='')

    def get_train_instances(self, train):
        users, pos_items, neg_items = [], [], []

        num_items = train.shape[1]

        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                users.append(u)
                pos_items.append(i)

                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                neg_items.append(j)
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    def get_user_dataloader(self, batch_size):
        users, pos_neg_items = self.get_train_instances(self.user_train_matrix)
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_dataloader(self, batch_size):
        groups, pos_neg_items = self.get_train_instances(self.group_train_matrix)
        train_data = TensorDataset(torch.LongTensor(groups), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)


def load_rating_file_to_list(filename):
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        # Each line: user item
        rating_list.append([int(contents[0]), int(contents[1])])
        
    return rating_list


def load_rating_file_to_matrix(filename, num_users=None, num_items=None):

    if num_users is None:
        num_users, num_items = 0, 0

    lines = open(filename, 'r').readlines()
    for line in lines:
        contents = line.split()
        u, i = int(contents[0]), int(contents[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    for line in lines:
        contents = line.split()
        if len(contents) > 2:
            u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
            if rating > 0:
                mat[u, i] = 1.0
        else:
            u, i = int(contents[0]), int(contents[1])
            mat[u, i] = 1.0
            
    return mat


def load_negative_file(filename):

    negative_list = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        negatives = line.split()[1:]
        negatives = [int(neg_item) for neg_item in negatives]
        negative_list.append(negatives)
        
    return negative_list


def load_group_member_to_dict(user_in_group_path):

    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(','):
            group_member_dict[group].append(int(member))
            
    return group_member_dict


def build_group_graph(group_data, num_groups):

    matrix = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group_a = set(group_data[i])
        for j in range(i + 1, num_groups):
            group_b = set(group_data[j])
            overlap = group_a & group_b
            union = group_a | group_b
            # weight computation
            matrix[i][j] = float(len(overlap) / len(union))
            matrix[j][i] = matrix[i][j]

    matrix = matrix + np.diag([1.0] * num_groups)
    degree = np.sum(np.array(matrix), 1)

    return np.dot(np.diag(1.0 / degree), matrix)


def build_hyper_graph(group_member_dict, group_train_path, num_users, num_items, num_groups, group_item_dict=None):

    if group_item_dict is None:
        group_item_dict = defaultdict(list)

        for line in open(group_train_path, 'r').readlines():
            contents = line.split()
            if len(contents) > 2:
                group, item, rating = int(contents[0]), int(contents[1]), int(contents[2])
                if rating > 0:
                    group_item_dict[group].append(item)
            else:
                group, item = int(contents[0]), int(contents[1])
                group_item_dict[group].append(item)

    def _prepare(group_dict, rows, axis=0):
        nodes, groups = [], []

        for group_id in range(num_groups):
            groups.extend([group_id] * len(group_dict[group_id]))
            nodes.extend(group_dict[group_id])

        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, groups)), shape=(rows, num_groups))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    user_hg, user_hg_deg = _prepare(group_member_dict, num_users)
    item_hg, item_hg_deg = _prepare(group_item_dict, num_items)

    for group_id, items in group_item_dict.items():
        group_item_dict[group_id] = [item + num_users for item in items]
    group_data = [group_member_dict[group_id] + group_item_dict[group_id] for group_id in range(num_groups)]
    full_hg, hg_dg = _prepare(group_data, num_users + num_items, axis=1)

    user_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(user_hg_deg),
                                       convert_sp_mat_to_sp_tensor(user_hg).t())
    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())
    full_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(hg_dg), convert_sp_mat_to_sp_tensor(full_hg))
    print(
        f"User hyper-graph {user_hyper_graph.shape}, Item hyper-graph {item_hyper_graph.shape}, Full hyper-graph {full_hyper_graph.shape}")

    return user_hyper_graph, item_hyper_graph, full_hyper_graph, group_data


def convert_sp_mat_to_sp_tensor(x):

    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def build_light_gcn_graph(group_item_net, num_groups, num_items):

    adj_mat = sp.dok_matrix((num_groups + num_items, num_groups + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = group_item_net.tolil()
    adj_mat[:num_groups, num_groups:] = R
    adj_mat[num_groups:, :num_groups] = R.T
    adj_mat = adj_mat.todok()

    row_sum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    graph = convert_sp_mat_to_sp_tensor(norm_adj)
    return graph.coalesce()

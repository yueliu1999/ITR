import os
import time
import torch
import random
import warnings
import argparse
import numpy as np
import torch.nn as nn
from model import ConsRec
import torch.optim as optim
from metrics import evaluate
from datetime import datetime
import torch.nn.functional as F
from dataloader import GroupDataset
from tensorboardX import SummaryWriter


def dis_fun(x, c):
    xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
    cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
    xx_cc = xx + cc
    xc = x @ c.T
    distance = xx_cc - 2 * xc
    return distance


def no_diag(x, n):
    x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def merge_and_split(init_cluster_center, cluster_density, cluster_radio_dis):

    # merge and split
    cluster_distance = 1 - (init_cluster_center @ init_cluster_center.T)

    sort_list = cluster_density.sort(descending=True)[1].tolist()
    sort_set = set(sort_list)

    cluster_embedding = torch.tensor([]).to(init_cluster_center.device)

    overall_step = 1
    explore_step = 0

    
    last_cluster_embedding = None
    epsilon  = np.exp(-explore_step * explore_step / overall_step)

    max_step = 20

    for i in sort_list:
        if max_step<=0:
            break
        max_step -= 1

        if np.random.rand() < epsilon and last_cluster_embedding!=None:
            explore_step += 1

            tmp_cluster_embedding = (init_cluster_center[i] + last_cluster_embedding) / 2
            distance = 1 - (tmp_cluster_embedding @ init_cluster_center.T)
            density_all = torch.tensor([]).to(init_cluster_center.device)
            radio_dis_all = torch.tensor([]).to(init_cluster_center.device)

            for radio in [0.1, 0.2, 0.3]:
                tmp_dis = distance.min(-1)[0] + (distance.max(-1)[0] - distance.min(-1)[0]) * radio
                density = (distance < tmp_dis).int().sum(-1) / (3.14 * tmp_dis**2)
                density_all = torch.cat([density_all, density.reshape(-1, 1)], dim=-1)
                radio_dis_all = torch.cat([radio_dis_all, tmp_dis.reshape(-1, 1)], dim=-1)

            radio_dis_max_density = torch.gather(radio_dis_all, 1, density_all.max(-1)[1].reshape(-1, 1)).reshape(-1,)
            max_density = density_all.max(-1)[0]

            init_cluster_center = torch.concat([init_cluster_center, tmp_cluster_embedding.reshape(1, -1)], dim=0)
            cluster_density = torch.concat([cluster_density, max_density], dim=0)
            cluster_radio_dis = torch.concat([cluster_radio_dis, radio_dis_max_density], dim=0)
            sort_set.add(init_cluster_center.shape[0]-1)
            sort_list.append(init_cluster_center.shape[0]-1)

            cluster_distance = torch.concat([cluster_distance, distance.reshape(1, -1)], dim=0)
            cluster_distance = torch.concat([cluster_distance, torch.concat([distance.reshape(-1, 1), torch.tensor(0).reshape(-1, 1).to(cluster_distance.device)], dim=0)], dim=1)

        else:
            if i in sort_set:
                cluster_index = torch.nonzero((cluster_distance[i] < cluster_radio_dis[i]).long()).reshape(-1)
                cluster_index_ = torch.tensor([]).reshape(-1).long().to(init_cluster_center.device)
                for j in cluster_index:
                    if cluster_distance[j][i] <= cluster_radio_dis[j]:
                        cluster_index_ = torch.cat([cluster_index_, j.reshape(-1)], dim=-1)
        
                cluster_embedding_i = init_cluster_center[cluster_index_].mean(dim=0)

                cluster_embedding = torch.cat([cluster_embedding, cluster_embedding_i.reshape(1, -1)], dim=0)
                
                sort_set.difference_update(set(cluster_index_.tolist()))
        
        overall_step += 1

        epsilon  = np.exp(-explore_step * explore_step / overall_step)
        last_cluster_embedding = init_cluster_center[i]

    return cluster_embedding
    

def init_density_cluster_center(user_embedding_online):

    user_embedding_online_normalized = F.normalize(user_embedding_online, p=2, dim=-1)
    distance = 1 - (user_embedding_online_normalized @ user_embedding_online_normalized.T)

    density_all = torch.tensor([]).to(user_embedding_online.device)
    radio_dis_all = torch.tensor([]).to(user_embedding_online.device)
    for radio in [0.1, 0.2, 0.3]:
        tmp_dis = distance.min(-1)[0] + (distance.max(-1)[0] - distance.min(-1)[0]) * radio
        density = (distance < tmp_dis).int().sum(-1) / (3.14 * tmp_dis**2)
        density_all = torch.cat([density_all, density.reshape(-1, 1)], dim=-1)
        radio_dis_all = torch.cat([radio_dis_all, tmp_dis.reshape(-1, 1)], dim=-1)


    radio_dis_max_density = torch.gather(radio_dis_all, 1, density_all.max(-1)[1].reshape(-1, 1)).reshape(-1,)
    max_density = density_all.max(-1)[0]
    
    init_cluster_center_index = torch.nonzero((max_density > max_density.mean()).int()).reshape(-1)
    cluster_radio_dis = radio_dis_max_density[init_cluster_center_index]
    cluster_density = max_density[init_cluster_center_index]
    init_cluster_center = user_embedding_online_normalized[init_cluster_center_index]

    cluster_radio_dis = radio_dis_max_density
    cluster_density = max_density
    init_cluster_center = user_embedding_online_normalized

    cluster_embedding = merge_and_split(init_cluster_center, cluster_density, cluster_radio_dis)

    return cluster_embedding


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True


def training(train_loader, epoch, type_m="group"):

    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    for batch_id, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(running_device), pi_ni[:, 1].to(running_device)


        pos_prediction = train_model(None, user_input, pos_items_input)
        neg_prediction = train_model(None, user_input, neg_items_input)

        optimizer.zero_grad()
        if args.loss_type == "BPR":
            loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        else:
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)


        losses.append(loss)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(
            f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMERa2011]", default="Mafengwo")
    parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

    parser.add_argument("--layers", type=int, help="# HyperConv & OverlapConv layers", default=3)
    parser.add_argument("--emb_dim", type=int, help="User/Item/Group embedding dimensions", default=32)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--cluster", type=float, default=10)
    parser.add_argument("--g2i", type=float, default=10)
    parser.add_argument("--topK", type=list, default=[1, 5, 10])

    parser.add_argument("--epoch", type=int, default=100, help="# running epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--loss_type", type=str, default="BPR")

    args = parser.parse_args()
    set_seed(args.seed)


    print('= ' * 20)
    print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(args)

    writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(writer_dir)

    running_device = torch.device(args.device)

    user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
    dataset = GroupDataset(user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset)
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    print(f" #Users {num_users}, #Items {num_items}, #Groups {num_groups}\n")


    train_model = ConsRec(num_users, num_items, num_groups, args, dataset.group_member_dict)
    train_model.to(running_device)
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)

    for epoch_id in range(args.epoch+1):

        train_model.train()

        clustering_loss = 0
        g2i_loss = 0

        if epoch_id % 20 == 0:

            optimizer.zero_grad()
            cluster_centers = init_density_cluster_center(train_model.user_embedding.weight)

            train_model.user_group_embedding = nn.Parameter(cluster_centers)

        user_embedding_online_normalized = F.normalize(train_model.user_embedding.weight, p=2, dim=-1)
        cluster_center_normalized = F.normalize(train_model.user_group_embedding, p=2, dim=-1)

        sample_center_distance = dis_fun(user_embedding_online_normalized, cluster_center_normalized)

        center_distance = dis_fun(cluster_center_normalized, cluster_center_normalized)

        no_diag(center_distance, cluster_center_normalized.shape[0])
        clustering_loss = args.cluster * (sample_center_distance.mean() - center_distance.mean())

        clustering_loss.backward()
        optimizer.step()

        assignment_matrix = (sample_center_distance>sample_center_distance.mean()).float()

        g2i = assignment_matrix.T @ torch.tensor(dataset.user_train_matrix.todense()).to(assignment_matrix.device)
        g2i = (g2i>0).float()

        optimizer.zero_grad()
        g2i_pred = train_model.user_group_embedding @ train_model.item_embedding.weight.T
        g2i_loss = args.g2i * F.mse_loss(g2i_pred, g2i)

        g2i_loss.backward()
        optimizer.step()

        user_loss = training(dataset.get_user_dataloader(args.batch_size), epoch_id, "user")


        if (epoch_id + 1) % 100 == 0:
            hits, ndcgs = evaluate(train_model, dataset.group_test_ratings, dataset.group_test_negatives, running_device,
                                args.topK, 'group')

    
            print(f"[Epoch {epoch_id}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
            

            hrs, ngs = evaluate(train_model, dataset.user_test_ratings, dataset.user_test_negatives, running_device,
                                args.topK, 'user')

            print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hrs}, NDCG@{args.topK}: {ngs}")
            print()

    
    hits, ndcgs = evaluate(train_model, dataset.group_test_ratings, dataset.group_test_negatives, running_device,
                        args.topK, 'group')
    print(f"[Epoch {epoch_id}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
    

    hrs, ngs = evaluate(train_model, dataset.user_test_ratings, dataset.user_test_negatives, running_device,
                        args.topK, 'user')
    print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hrs}, NDCG@{args.topK}: {ngs}")
    print()

    print()
    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
    print("Done!")

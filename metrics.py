
import math
import torch
import numpy as np


def get_hit_k(pred_rank, k):
    
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return round(hit, 5)


def get_ndcg_k(pred_rank, k):

    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)


def evaluate(model, test_ratings, test_negatives, device, k_list, type_m='group'):

    model.eval()
    hits, ndcgs = [], []
    user_test, item_test = [], []

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        items = [rating[1]]
        items.extend(test_negatives[idx])

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

    users_var = torch.LongTensor(np.array(user_test)).to(device)
    items_var = torch.LongTensor(np.array(item_test)).to(device)

    bsz = len(test_ratings)
    item_len = len(test_negatives[0]) + 1

    users_var = users_var.view(-1)
    items_var = items_var.view(-1)

    if type_m == 'group':
        predictions = model(users_var, None, items_var)
    elif type_m == 'user':
        predictions = model(None, users_var, items_var)

    predictions = torch.reshape(predictions, (bsz, item_len))

    pred_score = predictions.data.cpu().numpy()

    pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        hits.append(get_hit_k(pred_rank, k))
        ndcgs.append(get_ndcg_k(pred_rank, k))

    return hits, ndcgs

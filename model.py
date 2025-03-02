
import torch
import torch.nn as nn


class PredictLayer(nn.Module):
    
    def __init__(self, emb_dim, drop_ratio=0.):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.linear(x)

class OverlapGraphConvolution(nn.Module):

    def __init__(self, layers):
        super(OverlapGraphConvolution, self).__init__()
        self.layers = layers

    def forward(self, embedding, adj):
        group_emb = embedding
        final = [group_emb]

        for _ in range(self.layers):
            group_emb = torch.mm(adj, group_emb)
            final.append(group_emb)

        final_emb = torch.sum(torch.stack(final), dim=0)
        return final_emb


class LightGCN(nn.Module):

    def __init__(self, num_groups, num_items, layers, g):
        super(LightGCN, self).__init__()

        self.num_groups, self.num_items = num_groups, num_items
        self.layers = layers
        self.graph = g

    def compute(self, groups_emb, items_emb):
        all_emb = torch.cat([groups_emb, items_emb])
        embeddings = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embeddings.append(all_emb)
        embeddings = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        groups, _ = torch.split(embeddings, [self.num_groups, self.num_items])
        return groups

    def forward(self, groups_emb, items_emb):
        return self.compute(groups_emb, items_emb)


class HyperGraphBasicConvolution(nn.Module):
    
    def __init__(self, input_dim):
        super(HyperGraphBasicConvolution, self).__init__()
        self.aggregation = nn.Linear(3 * input_dim, input_dim)

    def forward(self, user_emb, item_emb, group_emb, user_hyper_graph, item_hyper_graph, full_hyper):
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)

        item_group_element = item_msg * group_emb
        msg = self.aggregation(torch.cat([user_msg, item_msg, item_group_element], dim=1))
        norm_emb = torch.mm(full_hyper, msg)
        return norm_emb, msg


class HyperGraphConvolution(nn.Module):

    def __init__(self, user_hyper_graph, item_hyper_graph, full_hyper, layers,
                 input_dim, device):
        super(HyperGraphConvolution, self).__init__()
        self.layers = layers
        self.user_hyper, self.item_hyper, self.full_hyper_graph = user_hyper_graph, item_hyper_graph, full_hyper
        self.hgnns = [HyperGraphBasicConvolution(input_dim).to(device) for _ in range(layers)]

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        final = [torch.cat([user_emb, item_emb], dim=0)]
        final_he = [group_emb]
        for i in range(len(self.hgnns)):
            hgnn = self.hgnns[i]
            emb, he_msg = hgnn(user_emb, item_emb, group_emb, self.user_hyper, self.item_hyper, self.full_hyper_graph)
            user_emb, item_emb = torch.split(emb, [num_users, num_items])
            final.append(emb)
            final_he.append(he_msg)

        final_emb = torch.sum(torch.stack(final), dim=0)
        final_he = torch.sum(torch.stack(final_he), dim=0)
        return final_emb, final_he


class ConsRec(nn.Module):
    
    def __init__(self, num_users, num_items, num_groups, args, group_number_dict):
        
        super(ConsRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        self.emb_dim = args.emb_dim
        self.layers = args.layers
        self.device = args.device
        self.predictor_type = args.predictor

        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)

        self.group_number_dict = group_number_dict

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.predict = PredictLayer(self.emb_dim)

    def forward(self, group_inputs, user_inputs, item_inputs):
        
        if (group_inputs is not None) and (user_inputs is None):
            return self.group_forward(group_inputs, item_inputs)
        else:
            return self.user_forward(user_inputs, item_inputs)

    def group_forward(self, group_inputs, item_inputs):

        i_emb = self.item_embedding(item_inputs)

        g_emb = torch.tensor([]).to(i_emb.device)
        for i in group_inputs:
            g_emb = torch.concat([g_emb, torch.mean(self.user_embedding(torch.tensor(self.group_number_dict[i.item()]).to(item_inputs.device)), dim=0).reshape(1, -1)], dim=0)

    
        return torch.sigmoid(self.predict(g_emb * i_emb))


    def user_forward(self, user_inputs, item_inputs):
        
        u_emb = self.user_embedding(user_inputs)
        i_emb = self.item_embedding(item_inputs)

        return torch.sigmoid(self.predict(u_emb * i_emb))


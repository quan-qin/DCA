import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn.conv.gat_conv import GATConv
import poi_graph
import math
from torch_geometric.nn.inits import reset, uniform
from config import CONFIG
import pytorch_warmup as warmup
import time
from utils import GradNormLossBalancer
EPS = 1e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads):
        super(GATEncoder, self).__init__()
        self.gat_conv = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.act_fn = nn.ReLU()
        self.attn_tuple = ...
        
    def get_poi_attn(self):
        return self.attn_tuple[0], torch.round(self.attn_tuple[1].mean(dim=1).cpu(), decimals=6)  # edge_idx, attn_value

    def forward(self, data):
        x, edge_index, edge_weight = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        res = x
        x, self.attn_tuple = self.gat_conv(x, edge_index, edge_weight, return_attention_weights=True)
        x = self.act_fn(x)
        x = x + res
        return x


class DCANet(nn.Module):
    r"""learning POI representations with Mid-type-centered Hierarchical Type InfoMax"""
    def __init__(self, hidden_dim, gat_encoder):
        super(DCANet, self).__init__()
        self.hidden_dim = hidden_dim
        self.poi_emb = torch.tensor(0)
        self.type_emb = torch.tensor(0)
        self.gat_encoder = gat_encoder
        self.weight_big2mid = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_sub2mid = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gat_encoder)
        uniform(self.hidden_dim, self.weight_big2mid)
        uniform(self.hidden_dim, self.weight_sub2mid)
    
    def _init_emb(self, poi_df, type_lvl):
        embedder = nn.Embedding(len(poi_df[f'{type_lvl}'].unique()), self.hidden_dim)
        poi_type_series = poi_df[f'{type_lvl}'].astype('category')
        poi_type_idx = poi_type_series.cat.codes.values
        type_emb = embedder(torch.IntTensor(poi_type_idx)).to(device)
        return type_emb
    
    @staticmethod
    def _truncate(emb, truncate_len=2048):
        # random truncation
        if truncate_len == -1:
            return emb
        else:
            if emb.size(0) > truncate_len:
                random_idx = torch.randint(0, emb.size(0), (truncate_len,), device=device)
                emb = emb[random_idx]
            else:
                return emb
        return emb

    def _agg_mid2big(self, poi_df: pd.DataFrame, mid_emb: torch.Tensor) -> torch.Tensor:
        map_dict = {}
        big_emb = self._init_emb(poi_df, 'big_type')
        mid_emb = mid_emb + big_emb
        poi_df['big_idx'] = [i for i in range(len(poi_df))]
        for _big_type, _df in poi_df.groupby(['big_type']):
            _big_emb_ls = []
            for _mid_type, _mid_df in _df.groupby(['mid_type']):
                _idx = torch.tensor(_mid_df['big_idx'].tolist()).to(device)
                _emb = mid_emb.index_select(0, _idx).to(device)  # [n_big_poi, emb_dim]
                _emb = torch.mean(_emb, dim=0)
                _big_emb_ls.append(_emb)
            _big_emb = torch.stack(_big_emb_ls, dim=0).mean(dim=0)
            map_dict[_big_type] = _big_emb
        big_emb_ls = [map_dict[_big_type] for _big_type in poi_df['big_type'].tolist()]
        big_emb = torch.stack(big_emb_ls, dim=0)  # [poi_num, emb_dim]
        return torch.sigmoid(big_emb)

    def _agg_sub2mid(self, poi_df: pd.DataFrame, sub_emb: torch.Tensor) -> torch.Tensor:
        map_dict = {}
        mid_emb = self._init_emb(poi_df, 'mid_type')
        sub_emb = sub_emb + mid_emb
        poi_df['mid_idx'] = [i for i in range(len(poi_df))]
        for _mid_type, _df in poi_df.groupby(['mid_type']):
            _mid_emb_ls = []
            for _sub_type, _sub_df in _df.groupby(['sub_type']):
                _idx = torch.tensor(_sub_df['mid_idx'].tolist()).to(device)
                _emb = sub_emb.index_select(0, _idx).to(device)  # [n_big_poi, emb_dim]
                _emb = torch.mean(_emb, dim=0)
                _mid_emb_ls.append(_emb)
            _mid_emb = torch.stack(_mid_emb_ls, dim=0).mean(dim=0)
            map_dict[_mid_type] = _mid_emb
        mid_emb_ls = [map_dict[_mid_type] for _mid_type in poi_df['mid_type'].tolist()]
        mid_emb = torch.stack(mid_emb_ls, dim=0)  # [poi_num, emb_dim]
        return torch.sigmoid(mid_emb)

    def forward(self, pig_builder):
        poi_df = pig_builder.poi_df
        pig_sub = pig_builder.get_poi_graph('sub_type')
        pig_sub = pig_sub.to(device)
        sub_emb = self.gat_encoder(pig_sub)
        
        mid_emb = self._agg_sub2mid(poi_df, sub_emb)
        self.type_emb = mid_emb
        pig_mid = pig_builder.get_poi_graph('mid_type', mid_emb)
        pig_mid = pig_mid.to(device)
        mid_emb = self.gat_encoder(pig_mid)
        self.poi_emb = mid_emb

        big_emb = self._agg_mid2big(poi_df, mid_emb)
        pig_big = pig_builder.get_poi_graph('big_type', big_emb)
        pig_big = pig_big.to(device)
        big_emb = self.gat_encoder(pig_big) + pig_sub.x
        
        sub_emb, mid_emb, big_emb = sub_emb.to(device), mid_emb.to(device), big_emb.to(device)
        pos_big_emb_ls, neg_big_emb_ls, pos_sub_emb_ls, neg_sub_emb_ls, mid_emb_ls = [], [], [], [], []
        for midtype_id in range(torch.max(pig_mid.midtype_id)+1):
            idx_of_midtype_within_a_bigtype = (
                pig_mid.bigtype_id == pig_builder.mid2big_map_dict[midtype_id]).nonzero(as_tuple=True)[0]
            idx_of_midtype_without_a_bigtype = (
                pig_mid.bigtype_id != pig_builder.mid2big_map_dict[midtype_id]).nonzero(as_tuple=True)[0]
            pos_big_emb = big_emb[idx_of_midtype_within_a_bigtype]
            neg_big_emb = big_emb[idx_of_midtype_without_a_bigtype]
            idx_of_midtype_within_a_subtype = (pig_mid.midtype_id == midtype_id).nonzero(as_tuple=True)[0]
            idx_of_midtype_without_a_subtype = (pig_mid.midtype_id != midtype_id).nonzero(as_tuple=True)[0]
            pos_sub_emb = sub_emb[idx_of_midtype_within_a_subtype]
            neg_sub_emb = sub_emb[idx_of_midtype_without_a_subtype]
            idx_of_midtype_equal_to_midtype_id = (pig_mid.midtype_id == midtype_id).nonzero(as_tuple=True)[0]
            _mid_emb = mid_emb[idx_of_midtype_equal_to_midtype_id]
            pos_big_emb_ls.append(pos_big_emb)
            neg_big_emb_ls.append(neg_big_emb)
            pos_sub_emb_ls.append(pos_sub_emb)
            neg_sub_emb_ls.append(neg_sub_emb)
            mid_emb_ls.append(_mid_emb)
        return pos_sub_emb_ls, neg_sub_emb_ls, pos_big_emb_ls, neg_big_emb_ls, mid_emb_ls
    
    def _discriminator_possub2mid(self, sub_emb_ls, mid_emb_ls):
        values = []
        for mid_type_idx, sub_emb in enumerate(sub_emb_ls):
            if sub_emb.size(0) > 0:
                sub_emb = self._truncate(sub_emb)
                mid_emb = self._truncate(mid_emb_ls[mid_type_idx])
                value = torch.matmul(sub_emb, torch.matmul(self.weight_sub2mid, mid_emb.T))
                value = -torch.log(torch.sigmoid(value) + EPS).mean()
                values.append(torch.unsqueeze(value, dim=0))
        loss_pos_sub2mid = torch.cat(values, dim=0).mean()
        return loss_pos_sub2mid

    def _discriminator_negsub2mid(self, sub_emb_ls, mid_emb_ls):
        values = []
        for mid_type_idx, sub_emb in enumerate(sub_emb_ls):
            if sub_emb.size(0) > 0:
                sub_emb = self._truncate(sub_emb)
                mid_emb = self._truncate(mid_emb_ls[mid_type_idx])
                value = torch.matmul(sub_emb, torch.matmul(self.weight_sub2mid, mid_emb.T))
                value = -torch.log(1 - torch.sigmoid(value) + EPS).mean()
                values.append(torch.unsqueeze(value, dim=0))
        loss_neg_sub2mid = torch.cat(values, dim=0).mean()
        return loss_neg_sub2mid
    
    def _discriminator_posbig2mid(self, big_emb_ls, mid_emb_ls):
        values = []
        for mid_type_idx, big_emb in enumerate(big_emb_ls):
            if big_emb.size(0) > 0:
                big_emb = self._truncate(big_emb)
                mid_emb = self._truncate(mid_emb_ls[mid_type_idx])
                value = torch.matmul(big_emb, torch.matmul(self.weight_big2mid, mid_emb.T))
                value = -torch.log(torch.sigmoid(value) + EPS).mean()
                values.append(torch.unsqueeze(value, dim=0))
        loss_pos_big2mid = torch.cat(values, dim=0).mean()
        return loss_pos_big2mid
    
    def _discriminator_negbig2mid(self, big_emb_ls, mid_emb_ls):
        values = []
        for mid_type_idx, big_emb in enumerate(big_emb_ls):
            if big_emb.size(0) > 0:
                big_emb = self._truncate(big_emb)
                mid_emb = self._truncate(mid_emb_ls[mid_type_idx])
                value = torch.matmul(big_emb, torch.matmul(self.weight_big2mid, mid_emb.T))
                value = -torch.log(1 - torch.sigmoid(value) + EPS).mean()
                values.append(torch.unsqueeze(value, dim=0))
        loss_neg_big2mid = torch.cat(values, dim=0).mean()
        return loss_neg_big2mid
    
    def criterion(self, pos_sub_emb_ls, neg_sub_emb_ls, pos_big_emb_ls, neg_big_emb_ls, mid_emb_ls):
        loss_pos_sub2mid = self._discriminator_possub2mid(pos_sub_emb_ls, mid_emb_ls)
        loss_neg_sub2mid = self._discriminator_negsub2mid(neg_sub_emb_ls, mid_emb_ls)
        loss_pos_big2mid = self._discriminator_posbig2mid(pos_big_emb_ls, mid_emb_ls)
        loss_neg_big2mid = self._discriminator_negbig2mid(neg_big_emb_ls, mid_emb_ls)
        loss_sub2mid = loss_pos_sub2mid + loss_neg_sub2mid
        loss_big2mid = loss_pos_big2mid + loss_neg_big2mid
        return [loss_sub2mid, loss_big2mid]

    def get_poi_emb(self):
        return self.poi_emb.clone().cpu().detach()
    
    def get_type_emb(self):
        return self.type_emb.clone().cpu().detach()


def trainer(cfg: CONFIG, epoch, save_model: bool, save_attn: bool):
    hyperpms = f'DCA_dim{cfg.args["dim"]}head{cfg.args["attn_head"]}'

    pig_builder = poi_graph.PIG(filepath=fr"{cfg.filepath}/euluc_POI_SJ.csv",
                                taz_id='UUID',
                                lng='WGS84_lon',
                                lat='WGS84_lat',
                                cfg=cfg
                                )
    
    loss_balancer = GradNormLossBalancer(task_num=2, alpha=1.5, weight_loss_gn=1)

    model = DCANet(
        hidden_dim=cfg.args['dim'],
        gat_encoder=GATEncoder(cfg.args['dim'], cfg.args['dim'], heads=cfg.args['attn_head']).to(device),
    ).to(device)
    optimizer = torch.optim.SGD(list(model.parameters()) + [loss_balancer.weights], lr=cfg.args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=cfg.args['gamma'])
    warmup_scheduler = warmup.LinearWarmup(optimizer, cfg.args['warmup_period'])
    model.train()
    loss_min = math.inf
    model_save = None
    for _epoch in range(epoch):
        optimizer.zero_grad()
        pos_sub_emb_ls, neg_sub_emb_ls, pos_big_emb_ls, neg_big_emb_ls, mid_emb_ls = model(pig_builder)
        losses = model.criterion(pos_sub_emb_ls, neg_sub_emb_ls, pos_big_emb_ls, neg_big_emb_ls, mid_emb_ls)
        losses, loss_gradnorm = loss_balancer(losses, model.gat_encoder.parameters())
        loss = losses + loss_gradnorm
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=.9)
        optimizer.step()
        with warmup_scheduler.dampening():
            scheduler.step()
        loss_balancer.norm_weights()
        if loss.item() < loss_min:
            model_save = model
            loss_min = loss.item()
    if save_model:
        torch.save(model_save.cpu(), fr'{cfg.filepath}/{hyperpms}.pt')
    poi_df = pig_builder.poi_df
    poi_df['poi_vec'] = model_save.get_poi_emb().tolist()
    poi_df.to_pickle(fr'{cfg.filepath}/poi_emb_df@{hyperpms}.pkl')
    if save_attn:
        poi_df['poi_vec'] = model_save.get_type_emb().tolist()
        poi_df.to_pickle(fr'{cfg.filepath}/type_emb_df@{hyperpms}.pkl')
        edge_idx, attn = model_save.gat_encoder.get_poi_attn()
        cl_poitype, cl_poix, cl_poiy = poi_df['mid_type'].tolist(), poi_df['WGS84_lon'].tolist(), \
            poi_df['WGS84_lat'].tolist()  # cl(column)
        edge_df_data = [(cl_poitype[_o_idx], cl_poix[_o_idx], cl_poiy[_o_idx], cl_poitype[_d_idx], cl_poix[_d_idx],
                         cl_poiy[_d_idx], _attn) for _o_idx, _d_idx, _attn in zip(edge_idx[0], edge_idx[1], attn)]
        attn_df = pd.DataFrame(
            data=edge_df_data,
            columns=['poi_a_type', 'poi_a_lng', 'poi_a_lat', 'poi_b_type', 'poi_b_lng', 'poi_b_lat', 'attn_value']
        )
        attn_df['edge_id'] = [i for i in range(len(attn_df))]
        attn_df.to_pickle(fr'{cfg.filepath}/attn_midtype_df@{hyperpms}.pkl')
        attn_df.to_csv(fr'{cfg.filepath}/attn_midtype_df@{hyperpms}.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    cfg = CONFIG()
    cfg.set_seed()
    cfg.set_args()
    trainer(cfg, epoch=200, save_model=False, save_attn=False)
    
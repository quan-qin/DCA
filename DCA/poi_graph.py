import pandas as pd
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Literal
from config import CONFIG
from math import radians, cos, sin, asin, sqrt
from torch_geometric.utils import to_undirected


class PIG:
    def __init__(self, filepath, taz_id, lng, lat, cfg: CONFIG = ...):
        self.cfg = cfg
        self.filepath = filepath        # poi csvfile
        self.field_taz = taz_id         # the field name of taz id
        self.field_lng = lng            # the field name of poi lng of the csvfile
        self.field_lat = lat            # the field name of poi lat of the csvfile
        self.type_emb = ...             # embedding of poi types [poi num, poitype_emb_dim]
        self.tri = ...                  # poi-based DT network
        self.pig = ...                  # poi-based weighted undirected graph
        self.poi_df = ...               # the dataframe data of poi csvfile
        self.edge_index = ...
        self.edge_weights = ...
        self.bigtype_id = ...
        self.subtype_id = ...
        self.midtype_id = ...
        self.mid2big_map_dict = ...
        self.sub2mid_map_dict = ...
        self._build_poi_network()
        self._poitype_idx()

    @staticmethod
    def _weight_geodist(dl, dist, iso_region: bool):
        """
        inverse distance weight (IDW) the edges
        ref. DOI: 10.1016/j.isprsjprs.2022.11.021
        :param dl: diagonal length of the minimum bounding rectangle of all the pois
        :param dist: spatial distance between the two POIs that form the edge
        :param iso_region: whether the two pois intra iso-region or cross-region
        :return: weighted haversine distance
        """
        miu_cross_region = .4  # a factor to differentiate cross-region (wr = 0.4) edges
        miu_iso_region = 1     # a factor to differentiate intra-region (wr = 1) edges
        const = 1            # a constant to avoid infinite value
        alpha = 1.5          # an inverse distance factor
        if iso_region:
            weighted_geodist = np.log((const + dl ** alpha) / (const + dist ** alpha)) * miu_iso_region
        else:
            weighted_geodist = np.log((const + dl ** alpha) / (const + dist ** alpha)) * miu_cross_region
        return weighted_geodist

    @staticmethod
    def _geodist(lng1, lat1, lng2, lat2):
        """
        calculate the haversine distance between poi1 & poi2
        :param lng1: poi1 lng
        :param lat1: poi1 lat
        :param lng2: poi2 lng
        :param lat2: poi2 lat
        :return: distance (m)
        """
        lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
        dis1 = 2 * asin(sqrt(a)) * 6371 * 1000
        return round(dis1, 2)

    def _diagonal_mbr(self, pois):
        """
        calculate the diagonal length of the minimum bounding rectangle (mbr) of all the POIs
        :param pois: coordinates of the pois [num_pois, 2]
        :return: diagonal length of the mbr
        """
        hull = ConvexHull(pois)
        hull_vertices = hull.vertices
        hull_points = pois[hull_vertices, :]
        x_min, x_max = np.min(hull_points[:, 0]), np.max(hull_points[:, 0])
        y_min, y_max = np.min(hull_points[:, 1]), np.max(hull_points[:, 1])
        mbr_diagonal_length = self._geodist(x_max, y_max, x_min, y_min)
        return round(mbr_diagonal_length, 2)
    
    def _init_poitype_emb(self, type_lvl):
        embedder = nn.Embedding(len(self.poi_df[f'{type_lvl}'].unique()), self.cfg.args['dim'])
        poi_type_series = self.poi_df[f'{type_lvl}'].astype('category')
        poi_type_idx = poi_type_series.cat.codes.values
        type_emb = embedder(torch.IntTensor(poi_type_idx))
        return type_emb
    
    def _onehot(self, type_lvl):
        type_map = {_type: _idx for _idx, _type in enumerate(list(self.poi_df[f'{type_lvl}'].unique()))}
        coded_types = self.poi_df[f'{type_lvl}'].map(type_map)
        self.num_poi_type = len(coded_types.unique())
        coded_types_ = torch.from_numpy(coded_types.values).reshape(-1, 1)
        type_emb = torch.eye(self.num_poi_type)[coded_types_.squeeze()]
        return type_emb
                
    def _poitype_idx(self):
        bigtype_series = self.poi_df['big_type'].astype('category')
        midtype_series = self.poi_df['mid_type'].astype('category')
        subtype_series = self.poi_df['sub_type'].astype('category')
        bigtype_id = bigtype_series.cat.codes.values
        self.bigtype_id = torch.IntTensor(bigtype_id)
        midtype_id = midtype_series.cat.codes.values
        self.midtype_id = torch.IntTensor(midtype_id)
        subtype_id = subtype_series.cat.codes.values
        self.subtype_id = torch.IntTensor(subtype_id)
        self.mid2big_map_dict = dict(zip(midtype_id, bigtype_id))
        self.sub2mid_map_dict = dict(zip(subtype_id, midtype_id))

    def _build_poi_network(self, network_viz: bool = False):
        """
        :param emb_init_method: {'random', 'onehot'}
        :param network_viz: whether to plot the dt network
        :param graph_viz: whether to plot the dt network-based poi graph
        :return: dt network-based poi graph
        """
        self.poi_df = pd.read_csv(self.filepath)
        self.poi_df.dropna(inplace=True)
        self.poi_df.reset_index(drop=True, inplace=True)
        coords = self.poi_df[[self.field_lng, self.field_lat]].astype('float32').values
        self.tri = Delaunay(coords)
        if network_viz:
            import matplotlib.pyplot as plt
            plt.triplot(coords[:, 0], coords[:, 1], self.tri.simplices, alpha=.6, linewidth=1)
            plt.plot(coords[:, 0], coords[:, 1], 'o', markersize=1)
            plt.show()
        edges = set()
        for edge_a, edge_b, edge_c in self.tri.simplices:
            edges.update([(edge_a, edge_b), (edge_b, edge_c), (edge_c, edge_a)])
        self.edge_index = np.array(list(edges))  # [num_edges, 2]
        iso_taz = [self.poi_df[self.field_taz][_i_idx] == self.poi_df[self.field_taz][_j_idx] for _i_idx, _j_idx in
                   self.edge_index]  # [num_edges,]
        edge_dist = [self._geodist(poi1[0], poi1[1], poi2[0], poi2[1]) for poi1, poi2 in
                     zip(coords[self.edge_index[:, 0]], coords[self.edge_index[:, 1]])]  # [num_edges,]
        diagonal_len = self._diagonal_mbr(coords)
        edge_weights = [self._weight_geodist(diagonal_len, dist, is_cross) for dist, is_cross in
                        zip(edge_dist, iso_taz)]
        edge_weights = np.array(edge_weights)  # [num_edges,]
        self.edge_weights = (edge_weights-np.min(edge_weights)) / (np.max(edge_weights)-np.min(edge_weights))  # norm

    def get_poi_graph(self, poi_type: Literal['sub_type', 'mid_type', 'big_type'],
                      type_emb=...,
                      emb_init_method: Literal['random', 'onehot'] = 'random'
                      ):
        if poi_type == 'sub_type':
            if emb_init_method == 'onehot':
                type_emb_ = self._onehot('sub_type')
            else:
                type_emb_ = self._init_poitype_emb('sub_type')
        elif 'mid_type':
            if type_emb == ...:
                if emb_init_method == 'onehot':
                    type_emb_ = self._onehot('mid_type')
                else:
                    type_emb_ = self._init_poitype_emb('mid_type')
            else:
                type_emb_ = type_emb
        else:
            type_emb_ = type_emb
        pig = Data(
            x=type_emb_,  # [num_nodes, dim_node_features]
            edge_index=torch.tensor(self.edge_index, dtype=torch.long).T,  # [2, num_edges]
            edge_attr=torch.tensor(self.edge_weights, dtype=torch.float).T.unsqueeze(dim=1),  # [num_edges, n_edge_vec]
            bigtype_id=self.bigtype_id,  # tensor[num_nodes,]
            midtype_id=self.midtype_id,  # tensor[num_nodes,]
            subtype_id=self.subtype_id  # tensor[num_nodes,]
        )
        self.pig = pig
        return pig
    
    def pig_viz(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.Graph()
        G.add_nodes_from(range(self.pig.num_nodes))
        G.add_edges_from(self.pig.edge_index.t().tolist())
        pos = nx.spring_layout(G)  # set position of the nodes
        nx.draw(G, pos, with_labels=True, node_color='white', edge_color='black')
        plt.show()

    def get_poi_neighbor(self, target_poi_idx, n_order=1):
        """
        retrieve n-order neighbor pois of the specified target_poi
        :param target_poi: specified a target poi
        :param n_order: the order num of to be retrieved neighbor pois
        :return: idx of the neighbor pois
        """
        from torch_geometric.utils import k_hop_subgraph
        subgraph = k_hop_subgraph(target_poi_idx, n_order, self.pig.edge_index)
        neighbors_idx = subgraph[0].tolist()
        return neighbors_idx


def main(poi_type: Literal['sub_type', 'mid_type', 'big_type'], cfg: CONFIG = ...):
    pig = PIG(cfg=cfg,
              filepath=fr"{cfg.filepath}/poi.csv",
              taz_id='UUID',
              lng='WGS84_lon',
              lat='WGS84_lat',
              )
    graph_data = pig.get_poi_graph(poi_type=poi_type)
    print(graph_data.edge_index.shape, graph_data.x.shape)

    return graph_data, pig.poi_df


if __name__ == '__main__':
    cfg = CONFIG()
    cfg.set_args()
    cfg.set_seed()
    main('mid_type', cfg)

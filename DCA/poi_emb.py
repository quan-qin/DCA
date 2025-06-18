import pandas as pd
import numpy as np
from config import CONFIG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from typing import Literal
from typing import List, Optional
import poi_graph
from collections import Counter

cfg = CONFIG()
cfg.set_seed()
cfg.set_plt(eng=False)


class POIVec:
    def __init__(self,
                 filepath=cfg.filepath,
                 poi_vec_filename=...,
                 field_taz_id='UUID',
                 field_poi_id='JOIN_FID',
                 field_poi_vec='poi_vec',
                 seed=cfg.seed,
                 ):
        self.filepath = filepath
        self.poi_vec_df = pd.read_pickle(fr'{filepath}/{poi_vec_filename}')
        self.field_taz_id = field_taz_id
        self.field_poi_id = field_poi_id
        self.field_poi_vec = field_poi_vec
        self.poi_vec = ...,
        self.seed = seed
        
    def _poitype_color(self):
        poi_chi2eng = np.load(fr'{self.filepath}/bigtype_chi2eng.npy', allow_pickle=True).item()
        self.poi_vec_df['big_type'] = self.poi_vec_df['big_type'].map(poi_chi2eng)
        type_set = self.poi_vec_df['big_type'].unique()
        color_ls = ["#"+''.join([random.choice('0123456789ABCDEF') for _j in range(6)]) for _i in type_set]

        poi_type = self.poi_vec_df['big_type']
        type_color_map = {_type: _color for _type, _color in zip(type_set, color_ls)}
        poitype_color = poi_type.map(type_color_map)
        return poitype_color, type_color_map

    def poi_viz(self, filename=None, hyperpms=..., local_viz: Optional[List] = None):
        poi_vec = self.poi_vec_df[f'{self.field_poi_vec}']

        if filename is None:
            dim_reduced_vec = dim_reducer(self.seed, np.vstack(poi_vec), 'tsne')
            np.save(f'{self.filepath}/figure/poi_tsne/poi_tsne_vec@{hyperpms}.npy', dim_reduced_vec, allow_pickle=True)
        else:
            dim_reduced_vec = np.load(fr'{self.filepath}/figure/poi_tsne/{filename}.npy', allow_pickle=True)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.tick_params(left=False, bottom=False)
        poitype_color, type_color_map = self._poitype_color()
        ax.scatter(dim_reduced_vec[:, 0],
                   dim_reduced_vec[:, 1],
                   marker='o',
                   c=poitype_color,
                   alpha=.7,
                   s=4,
                   zorder=1
                   )
        legend_elements = [Line2D([], [], marker='o', color=f'{_color}', label=f'{_type}', markersize=5) for
                           _type, _color in type_color_map.items()]
        ax.legend(handles=legend_elements, frameon=False, loc='lower left', fontsize='small',
                  prop={'family': 'Arial', 'size': 7})
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        fig.tight_layout()
        fig.savefig(self.filepath+f'/figure/poi_tsne/poi_tsne@{hyperpms}.tif', dpi=600)
        if local_viz:
            tsne1_range, tsne2_range = local_viz[:2], local_viz[2:]
            rect = plt.Rectangle((tsne1_range[0], tsne2_range[0]), tsne1_range[1]-tsne1_range[0],
                                 tsne2_range[1]-tsne2_range[0], edgecolor='black', linewidth=2, facecolor='none')
            ax.add_patch(rect)
            fig.savefig(self.filepath+f'/figure/poi_tsne/poi_tsne@{hyperpms}_x{tsne1_range}_y{tsne2_range}.tif',
                        dpi=600)
            pig = poi_graph.PIG(filepath=fr"{cfg.filepath}/euluc_POI_SJ.csv",
                                taz_id='UUID',
                                lng='WGS84_lon',
                                lat='WGS84_lat',
                                )
            pig.get_poi_graph('mid_type')
            self._poi_subset_stats(hyperpms, dim_reduced_vec, tsne1_range, tsne2_range, pig)

    def get_taz_vec(self) -> pd.DataFrame:
        """
        mean-pooling all pois within taz
        :return: taz embeddings
        """
        taz_vec_ls = []  # [('taz_id', 'taz_vec')]
        for _group, _sub_df in self.poi_vec_df.groupby([f'{self.field_taz_id}']):
            if len(_sub_df) >= 10:  # avoid the sparse meaning and randomness produced by short pois sequences
                _pooling_vec = np.mean(_sub_df[f'{self.field_poi_vec}'].tolist(), axis=0)
                taz_vec_ls.append((_group, _pooling_vec, _sub_df['Level1'].tolist()[0]))  # Level1 | Level2
        taz_vec_df = pd.DataFrame(data=taz_vec_ls, columns=['taz_id', 'taz_vec', 'TrueLabel'])
        return taz_vec_df
        
    def _poi_subset_stats(self, hyperpms, tsne_array, tsne1_range, tsne2_range, pig: poi_graph.PIG):
        tsne1_mask = (tsne_array[:, 0] >= tsne1_range[0]) & (tsne_array[:, 0] <= tsne1_range[1])
        tsne2_mask = (tsne_array[:, 1] >= tsne2_range[0]) & (tsne_array[:, 1] <= tsne2_range[1])
        poi_subset_idx = np.where(tsne1_mask & tsne2_mask)[0].tolist()
        neigbors_idx = pig.get_poi_neighbor(poi_subset_idx, 1)
        neigbors_df = self.poi_vec_df[['mid_type', 'WGS84_lon', 'WGS84_lat']].iloc[neigbors_idx, :]
        
        midtype_neigbors_counts_dict = Counter(neigbors_df['mid_type'].tolist())
        sorted_items = sorted(midtype_neigbors_counts_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_midtype = [_[0] for _ in sorted_items]
        sorted_type_freq = [(_[1]/sum(midtype_neigbors_counts_dict.values())) for _ in sorted_items]
        plt.xticks(rotation=15)
        plt.ylabel('frequency')
        fig1 = plt.figure()
        plt.bar(sorted_midtype[:10], sorted_type_freq[:10], align='center', alpha=1,
                color=plt.cm.viridis(1-np.arange(10)/10))
        plt.savefig(f'{cfg.filepath}/figure/poi_tsne/subset_neigbors_stats@x{tsne1_range}_y{tsne2_range}.png', dpi=600)
        
        midtype_subset_ls = self.poi_vec_df['mid_type'][poi_subset_idx].tolist()
        midtype_counts_dict = Counter(midtype_subset_ls)
        sorted_items_ = sorted(midtype_counts_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_midtype_ = [_[0] for _ in sorted_items_]
        sorted_type_freq_ = [(_[1]/sum(midtype_counts_dict.values())) for _ in sorted_items_]

        plt.bar(sorted_midtype_[:10], sorted_type_freq_[:10], align='center', alpha=1,
                color=plt.cm.viridis(1-np.arange(10)/10))
        plt.savefig(f'{cfg.filepath}/figure/poi_tsne/subset_stats@{hyperpms}_x{tsne1_range}_y{tsne2_range}.png',
                    dpi=600)


def dim_reducer(seed, vec: np.ndarray, method: Literal['tsne', 'umap'] = 'tsne'):
    """
    Reduce the dimension of the input feature vectors to two
    :param seed: random seed
    :param method: {'tsne', 'umap'}
    :param vec: input feature vectors (np array)
    :return: dimensional reduced feature vectors
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(learning_rate='auto',
                       random_state=seed,
                       n_components=2,
                       n_iter=1000,
                       perplexity=30,
                       )
        vec_reduc = reducer.fit_transform(vec)
        x_min, x_max = vec_reduc.min(0), vec_reduc.max(0)
        vec_reduc_norm = (vec_reduc - x_min) / (x_max - x_min)  # [poi_num, 2]
    elif 'umap':
        import umap.umap_ as umap
        from sklearn.preprocessing import StandardScaler
        reducer = umap.UMAP(n_neighbors=200,
                            min_dist=.3,
                            n_components=2,
                            random_state=seed,
                            )
        vec_reduc = reducer.fit_transform(vec)
        x_min, x_max = vec_reduc.min(0), vec_reduc.max(0)
        vec_reduc_norm = (vec_reduc - x_min) / (x_max - x_min)  # [poi_num, 2]
    return vec_reduc_norm

    
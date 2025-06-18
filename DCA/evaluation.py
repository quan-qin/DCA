import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from config import CONFIG
import poi_emb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import LabelEncoder
from typing import List, Literal
import sklearn.metrics as metrics
from utils import calc_eval_result_per_region, EarlyStopping
from poi_emb import POIVec


class MLP(nn.Module):
    def __init__(self, embedding_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Evaluator:
    def __init__(self, task: Literal['func', 'price', 'checkin'], embeddings, labels, region_ids: torch.Tensor,
                 split_ratio: List[float], cfg: CONFIG):
        self.cfg = cfg
        self.cfg.set_seed(-1)
        self.device = self.cfg.device
        self.task = task
        self.embeddings = embeddings
        if task == 'func':
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            self.labels = torch.tensor(labels, device=cfg.device)
        elif task == 'price' or 'checkin':
            self.labels = torch.tensor(labels, device=cfg.device, dtype=torch.float)
        self.model = MLP(
            embeddings.shape[1], self.labels.unique().shape[0] if task == 'func' else 1, hidden_dim=256).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.criterion = {'func': nn.CrossEntropyLoss(), 'price': nn.MSELoss(), 'checkin': nn.MSELoss()}[task]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=.999)
        data_idx = np.arange(embeddings.shape[0])
        train_val_idx, self.test_idx = train_test_split(
            data_idx, test_size=split_ratio[2] / (split_ratio[0] + split_ratio[1] + split_ratio[2]),
            random_state=self.cfg.seed
        )
        self.train_idx, self.val_idx = train_test_split(
            train_val_idx, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]), random_state=self.cfg.seed
        )
        self.test_label = self.labels[self.test_idx]
        self.train_data = TensorDataset(self.embeddings[self.train_idx], self.labels[self.train_idx])
        self.val_data = TensorDataset(self.embeddings[self.val_idx], self.labels[self.val_idx])
        self.test_data = TensorDataset(self.embeddings[self.test_idx], self.test_label,
                                       region_ids[self.test_idx])
        self.train_ldr = DataLoader(self.train_data, batch_size=64, shuffle=True, drop_last=True)
        self.val_ldr = DataLoader(self.val_data, batch_size=64, shuffle=False)
        self.test_ldr = DataLoader(self.test_data, batch_size=64, shuffle=False)
        self.test_results_per_region = []  # [[region_id, eval_result]]

    def tester(self):
        self.model.eval()
        test_size = 0
        if self.task == 'func':
            test_oa, test_kappa, test_f1 = 0, 0, 0
            for data, target, region_id in self.test_ldr:
                output = self.model(data.to(self.device))
                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                output = np.argmax(output, axis=1)
                test_size += data.shape[0]
                test_oa += metrics.accuracy_score(target, output) * data.shape[0]
                test_kappa += metrics.cohen_kappa_score(target, output) * data.shape[0]
                test_f1 += metrics.f1_score(target, output, average='macro') * data.shape[0]
                self.test_results_per_region.extend(calc_eval_result_per_region(region_id, target, output, 'func'))
            test_oa = np.round(test_oa / test_size, 4)
            test_kappa = np.round(test_kappa / test_size, 4)
            test_f1 = np.round(test_f1 / test_size, 4)
            return test_oa, test_kappa, test_f1
        elif 'price' or 'checkin':
            targets, outputs = [], []
            for data, target, region_id in self.test_ldr:
                output = self.model(data.to(self.device))
                output = output.detach().cpu().flatten().numpy()
                target = target.detach().cpu().flatten().numpy()
                outputs.append(output)
                targets.append(target)
                self.test_results_per_region.extend(calc_eval_result_per_region(region_id, target, output, 'price'))
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)
            test_rmse = metrics.root_mean_squared_error(targets, outputs)
            test_mae = metrics.mean_absolute_error(targets, outputs)
            test_r2 = metrics.r2_score(targets, outputs)
            return test_rmse, test_mae, test_r2
    
    def trainer(self, epochs):
        early_stopping = EarlyStopping(patience=10, min_delta=.01)
        val_loss_min = math.inf
        for epoch in range(epochs):
            if early_stopping.early_stop:
                break
            self.model.train()
            for data, target in self.train_ldr:
                self.optimizer.zero_grad()
                output = self.model(data)
                train_loss = self.criterion(output, target)
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_loss = 0
                val_size = 0
                for data, target in self.val_ldr:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item() * data.shape[0]
                    val_size += data.shape[0]
                val_loss /= val_size
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break
                if val_loss < val_loss_min:
                    val_loss_min = val_loss
                    test_results = self.tester(epoch)
                    
        return test_results


def evaluate(embeddings, labels, region_ids, cfg, model_name, task, repeats, epochs):
    eval_results = []
    eval_results_per_region = []
    for _ in range(repeats):
        evaluator = Evaluator(task, embeddings, labels, region_ids, [.6, .2, .2], cfg)
        eval_result = evaluator.trainer(epochs)
        eval_results.append(eval_result)
        eval_results_per_region.extend(evaluator.test_results_per_region)

    eval_per_region_df = pd.DataFrame(
        data=eval_results_per_region, columns=['region_id', 'eval_result'])
    eval_per_region_df = eval_per_region_df.groupby('region_id').mean().reset_index()
    
    eval_results_mean = np.mean(eval_results, axis=0)
    eval_results_std = np.std(eval_results, axis=0)
    eval_results = [[f'{mean:.4f}±{std:.4f}' for mean, std in zip(eval_results_mean, eval_results_std)]]
    if task == 'func':
        eval_df = pd.DataFrame(data=eval_results, columns=['oa', 'kappa', 'macro-f1'])
    elif 'price':
        eval_df = pd.DataFrame(data=eval_results, columns=['rmse', 'mae', 'r2'])
    elif 'checkin':
        eval_df = pd.DataFrame(data=eval_results, columns=['rmse', 'mae', 'r2'])
    eval_df['model'] = model_name
    print(eval_df)
    
    eval_per_region_df.to_csv(
        f'{cfg.filepath}/eval_per_region_{model_name}_{task}.csv', index=False, encoding='utf-8')
    eval_df.to_csv(f'{cfg.filepath}/eval_{model_name}_{task}.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    
    def main(task: Literal['func', 'price', 'checkin'], repeats=1, epochs=100):
        cfg = CONFIG()

        poi2vec = poi_emb.main(poi_vec_name='poi_emb_df@DCA_lamda0.5dim64head4.pkl')
        poi2vec = POIVec(filepath=fr'{cfg.filepath}',
                         poi_vec_filename='poi_emb_df.pkl',
                         field_taz_id='UUID',
                         field_poi_id='JOIN_FID',
                         field_poi_vec='poi_vec',
                         seed=cfg.seed
                         )
        taz_vec_df_ = poi2vec.get_taz_vec()
        embeddings = torch.stack([torch.FloatTensor(_row) for _row in taz_vec_df_['taz_vec']])
        region_id = torch.stack([torch.FloatTensor(_row) for _row in taz_vec_df_['taz_id']]).squeeze(-1)
        
        if task == 'func':
            func_labels_df = pd.read_csv(f'{cfg.filepath}/GroundTruth/urban_func_labels.csv')
            func_labels = [
                func_labels_df.loc[func_labels_df['UUID'] == _region_id, 'label'].values[0] for _region_id in
                region_id.tolist()]
            evaluate(embeddings, func_labels, region_id, cfg, 'DCA', 'func', repeats, epochs)
        elif task == 'price':
            price_labels_df = pd.read_csv(f'{cfg.filepath}/GroundTruth/houseprice_labels.csv')
            price_labels = [
                price_labels_df.loc[price_labels_df['UUID'] == _region_id, 'price'].values[0] if
                _region_id in price_labels_df['UUID'].values else -1 for _region_id in region_id.tolist()]
            price_embeddings = embeddings[np.array(price_labels) != -1]
            region_id = region_id[np.array(price_labels) != -1]
            price_labels = [_ for _ in price_labels if _ != -1]
            evaluate(price_embeddings, price_labels, region_id, cfg, 'DCA', 'price', repeats, epochs)
        elif task == 'checkin':
            checkin_labels_df = pd.read_csv(f'{cfg.filepath}/GroundTruth/checkin_labels.csv')
            checkin_labels = [
                checkin_labels_df.loc[checkin_labels_df['UUID'] == _region_id, 'counts'].values[0] if
                _region_id in checkin_labels_df['UUID'].values else -1 for _region_id in region_id.tolist()]
            checkin_embeddings = embeddings[np.array(checkin_labels) != -1]
            region_id = region_id[np.array(checkin_labels) != -1]
            checkin_labels = [_ for _ in checkin_labels if _ != -1]
            evaluate(checkin_embeddings, checkin_labels, region_id, cfg, 'DCA', 'checkin', repeats, epochs)
            
    main('checkin', False)
    
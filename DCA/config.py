import random
import torch
import numpy as np
import os
import argparse


class CONFIG:
    def __init__(self):
        self.args = {}
        self.filepath = r'./'
        self.dpi = 600
        self.seed = random.randint(0, 0x7fffffff)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--dim', type=int, default=64)
        parser.add_argument('--warmup_period', type=int, default=20)
        parser.add_argument('--gamma', type=float, default=.99)
        parser.add_argument('--attn_head', type=int, default=4)
        args = parser.parse_args()
        self.args.update(vars(args))

    def set_plt(self,
                fig_style='ggplot',
                dpi=600,
                eng: bool = True,
                ):
        import matplotlib.pyplot as plt
        if fig_style is None:
            plt.style.use('default')
        else:
            plt.style.use(f'{fig_style}')
        if eng:
            plt.rcParams['font.sans-serif'] = ['Arial']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.dpi = dpi

    def set_seed(self, seed=42):
        if seed == -1:
            self.seed = random.randint(0, 0x7fffffff)
        else:
            self.seed = seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.manual_seed(self.seed)
        else:
            torch.manual_seed(self.seed)
            
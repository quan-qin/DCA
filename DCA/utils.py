import torch
from typing import Literal, List
import torch.nn as nn
import torch.nn.functional as F


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = .0):
        """
        Parameters:
        patience (int): Number of epochs to wait after last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def calc_eval_result_per_region(region_ids, labels, predictions, task: Literal['func', 'price', 'checkin']) -> List:
    predictions = torch.tensor(predictions).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    labels = torch.tensor(labels).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if task == 'func':
        correct = (predictions == labels).int()
    elif task == 'price' or 'checkin':
        correct = torch.abs(predictions - labels)
    region_predictions_list = []
    for _region_id, _correct in zip(region_ids, correct):
        region_predictions_list.append([_region_id.item(), _correct.item()])
    return region_predictions_list


class GradNormLossBalancer(nn.Module):
    def __init__(self, task_num: int, alpha: float = 1.5, weight_loss_gn: float = .1):
        super().__init__()
        self.alpha = alpha
        self.task_num = task_num
        self.weight_loss_gn = weight_loss_gn
        self.weights = nn.Parameter(torch.ones(task_num, requires_grad=True))
        self.initial_losses = None

    def forward(self, losses, shared_params):
        """
        Args:
            losses: List of scalar loss tensors, one for each task (requires_grad=True)
            shared_params: List or generator of shared model parameters
        Returns:
            total_loss: torch scalar, weighted sum
            gradnorm_loss: torch scalar, grad norm alignment loss
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([_.item() for _ in losses]).to(losses[0].device)

        losses_tensor = torch.stack(losses)
        weighted_losses = self.weights * losses_tensor
        total_loss = torch.sum(weighted_losses)

        grads = []
        shared_params = list(shared_params)

        for i in range(self.task_num):
            grad = torch.autograd.grad(losses[i], shared_params, retain_graph=True, create_graph=True)
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
            grads.append(grad_norm)
        grads = torch.stack(grads)

        # r_i(t)
        with torch.no_grad():
            loss_ratios = losses_tensor / self.initial_losses
            inv_train_rate = loss_ratios / loss_ratios.mean()

        # grad target
        mean_grad = grads.detach().mean()
        target = mean_grad * (inv_train_rate ** self.alpha)

        gradnorm_loss = F.l1_loss(grads, target, reduction='sum')

        return total_loss, gradnorm_loss * self.weight_loss_gn
    
    @torch.no_grad()
    def norm_weights(self):
        self.weights.data = self.task_num * self.weights.data / self.weights.data.sum()
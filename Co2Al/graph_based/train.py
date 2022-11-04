import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import Optional, Dict

def get_optimizer(t: str):
    if t == "adam":
        return optim.Adam
    elif t == "sgd":
        return optim.SGD
    elif t == 'adagrad':
        return optim.Adarad
    elif t == 'adamw':
        return optim.AdamW

def weighted_cross_entropy(w=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if w is None:
        w = torch.tensor([0.1, 1], device=device)
    
    return nn.CrossEntropyLoss(weight=w)
    
get_criterion_ = {
    "nll_loss": nn.CrossEntropyLoss,
    "kl":nn.KLDivLoss,
    "multilabel": nn.MultiLabelSoftMarginLoss,
    "weighted_nll": weighted_cross_entropy
}
def get_criterion(t: str):
  return get_criterion_.get(t, nn.CrossEntropyLoss)


def train_no_hybrid(model,
                    dataset,
                    adj,
                    optimizer: str,
                    criterion: str,
                    device,
                    lr: float = 1e-3,
                    epoches: int = 20,
                    lr_scheduler: bool = True,
                    return_loss: bool = True,
                    loss_kwargs: Optional[Dict] = None):
    """Function for training model

    Args:
        model (nn.Module): A model written in PyTorch.
                            An instance of class inherited nn.Module
        dataset (torch.data.Dataset): PyTorch Dataset. See documentation.
        adj (torch.sparse): PyTorch sparse matrix. Should be coo_tensor.
        optimizer (str): Name of optimizer.
                        Available: adam, sgd, adagrad
        criterion (str): Loss function. Currently only `nll_loss` is accepted
        device (str, torch.device): Device for training. Cuda/cpu
        lr (float, optional): Learning rate. Defaults to 1e-3.
        epoches (int, optional): Number of epoches. Defaults to 20.
        lr_scheduler (bool, optional): True if using scheduler.
                                Defaults to True.
        return_loss(bool, optional): True to trace loss in training progress.

    Returns:
        model: Trained model
    """
    if loss_kwargs is None:
        loss_kwargs = dict()
    optimizer = get_optimizer(optimizer)(model.parameters(), lr=lr)
    criterion = get_criterion(criterion)(**loss_kwargs)
    if lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    if return_loss:
        tracked_loss = []
    for _ in tqdm(range(epoches)):
        x, y = dataset.data, dataset.targets
        x = x.to(device)
        y = y.to(device).flatten()
        adj = adj.to(device)
        pred = model(x, adj)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if return_loss:
            with torch.no_grad():
                tracked_loss.append(loss.item())
        if lr_scheduler:
            scheduler.step(loss)
    if return_loss:
        return model, tracked_loss
    else:
        return model


def train_hybrid(model,
                 dataset,
                 adjs,
                 optimizer: str,
                 criterion: str,
                 device,
                 lr: float = 1e-3,
                 epoches: int = 20,
                 lr_scheduler: bool = True,
                 return_loss: bool = True):
    """Function for training model

    Args:
        model (nn.Module): A model written in PyTorch.
                            An instance of class inherited nn.Module
        dataset (torch.data.Dataset): PyTorch Dataset. See documentation.
        adjs (torch.sparse): List of PyTorch sparse matrix.
                            Each layer contains weights of rel between nodes.
        optimizer (str): Name of optimizer.
                        Available: adam, sgd, adagrad
        criterion (str): Loss function.
                        Available: nll_loss, KLDivLoss
        device (str, torch.device): Device for training. Cuda/cpu
        lr (float, optional): Learning rate. Defaults to 1e-3.
        epoches (int, optional): Number of epoches. Defaults to 20.
        lr_scheduler (bool, optional): True if using scheduler.
                                Defaults to True.
        return_loss(bool, optional): True to trace loss in training progress.

    Returns:
        model: Trained model
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = get_criterion(criterion)()
    if lr_scheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

    if return_loss:
        tracked_loss = []

    adjs = list(map(lambda a: a.to(device).float(), adjs))

    for _ in tqdm(range(1, epoches + 1)):
        x, y = dataset.data, dataset.targets
        x = x.to(device).float()
        y = y.to(device)
        pred = model(x, *adjs).float()

        if isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss):
            y = y.flatten().long()
        elif isinstance(criterion, torch.nn.modules.loss.KLDivLoss):
            y = F.one_hot(y.flatten()).float()
            pred = F.log_softmax(pred, dim=1)
        elif isinstance(criterion,
                        torch.nn.modules.loss.MultiLabelSoftMarginLoss):
            y = F.one_hot(y.flatten()).float()
            pred = F.log_softmax(pred, dim=1)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        if return_loss:
            with torch.no_grad():
                tracked_loss.append(loss.item())
    if return_loss:
        return model, tracked_loss
    else:
        return model
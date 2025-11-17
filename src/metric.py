import torch
import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, f1_score

def avg(l: List[int]):
    return sum(l) / len(l)

def calc_f1(y_true: torch.Tensor | np.ndarray, 
            y_pred: torch.Tensor | np.ndarray,):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    y_pred = y_pred.sigmoid().numpy()

    r_f1 = f1_score(y_true, y_pred > 0.5)
    return r_f1

def calc_acc(y_true: torch.Tensor | np.ndarray, 
             y_pred: torch.Tensor | np.ndarray):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.numpy()

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    y_pred = y_pred.sigmoid().numpy()

    r_acc = accuracy_score(y_true, y_pred > 0.5)
    return r_acc
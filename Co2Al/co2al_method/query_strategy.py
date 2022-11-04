
from typing import List, Optional, Union
import numpy as np
import random
import torch

#Tất cả công cụ dùng hỗ trợ truy vấn Active learning
#----------------------------------------------------------------

# Các công cụ hỗ trợ 

def argsort_by_condition(x: np.array, condition, n) -> np.array:
    """Lấy index các mẫu có giá trị lớn nhất và lớn hơn thresh
    Args:
        x (np.array): array
        condition (float): điều kiện muốn lấy
        n (int): số lượng mẫu muốn lấy
    Returns:
        index (np.array)
        Ví dụ: x = np.array([1 , 3, 2, 7, 5])
        argsort_by_condition(x, 1, 2)
        --> array([3,4])
    """
    idx, = np.where(x > condition)
    return idx[np.argsort(x[idx])[-n:]]

def roll(arr: np.ndarray, r: int):
    """Hàm dùng để hỗ trợ lấy index cho mỗi estimator 
    """
    return np.roll(arr, r, axis=0)

def converter(arr: Optional[Union[np.array, torch.Tensor]]) -> np.array:
    """Chuyển thành np.array
    """
    if not isinstance(arr, np.ndarray):
        return arr.numpy()
    return arr

def get_index_cotrain(x: np.array, condition, n):
    index_cotrain = np.array([argsort_by_condition(x[i],
                                                condition,
                                                n) for i in range(len(x))])
    length_cfg = len(min(index_cotrain, key=len))
    index_cotrain = np.array([index_cotrain[i][-length_cfg:] 
                              for i in range(len(index_cotrain))])
    return index_cotrain
    
#----------------------------------------------------------------
# Các chiến lược truy vấn Active learning

def least_confidence(prob: np.array, n: int):
    """Lấy index các mẫu least confidence từ các estimator
    Args:
        prob (List): list gồm các prob theo từng estimator
        n (int): số lượng mẫu muốn lấy
    Returns:
        uncertainties_index: list chứa index các mẫu least confidence từ các estimator1
        uncertainties_index shape: (n_views,)
    """

    uncertainties_index = prob.max(axis=2).argsort(axis=1)[:, ::1][:, :n]
    return uncertainties_index


def margin_sampling(prob: np.array, n: int):

    prob_copy = prob.copy()
    prob_copy.sort()
    uncertainties_index = (prob_copy[:,:,1] - prob_copy[:,:,0]).argsort(axis=1)[:,::1][:,:n]

    return uncertainties_index


def entropy_sampling(prob: np.array, n: int):

    uncertainties_index = (prob * np.log(prob)).sum(2).argsort(axis=1)[:,::1][:,:n]

    return uncertainties_index


def get_random_items(temp, n=20):

    U = np.array(range(temp[0].shape[0]))
    np.random.shuffle(U)
    random_index = U[-min(len(U), n):]
    return np.array([random_index])


# Hàm dùng để lấy index từ các estimator khác


def split_index_train_test(al_index,
                            ct_index,
                            type_ssl,
                            n_views) :
    """Returns:
            index_train: List, index_val: List
    """
    index_train, index_val = np.empty((n_views, 0), int), np.empty((n_views, 0), int)
    
    if type_ssl == 'coal':
        arrays = [al_index, ct_index]
    else:
        arrays = [al_index]
    for a in arrays:
        [random.shuffle(j) for j in a]
        index_train = np.unique(np.concatenate((index_train,
                                    a[:,round(len(a)*0.2):]),
                                    axis=1),
                                axis= 1)
        index_val = np.unique(np.concatenate((index_val,
                                    a[:,:round(len(a)*0.2)]),
                                    axis=1),
                                axis= 1)
    index_train = np.array(index_train)
    index_val = np.array(index_val)
    return index_train, index_val

def get_query_index(
                    al_index,
                    ct_index,
                    type_ssl,
                    n_views
                    ):
    index_train, index_val = split_index_train_test(al_index,
                                                    ct_index,
                                                    type_ssl,
                                                    n_views)
    
    if type_ssl == 'coal' or type_ssl == 'co2al' :
        out_train = [roll(index_train, r+1) for r in range(len(index_train) - 1)]
        out_val = [roll(index_val, r+1) for r in range(len(index_val) - 1)]
        query_index_train = np.concatenate(out_train, axis=1)
        query_index_val = np.concatenate(out_val, axis=1)
    elif type_ssl == 'al':
        query_index_train, query_index_val = index_train, index_val
    return query_index_train, query_index_val
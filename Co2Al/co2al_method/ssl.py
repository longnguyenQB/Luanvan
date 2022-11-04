from multiprocessing.dummy import Array
from typing import List, Optional, Union
from co2al_method.query_strategy import *
import numpy as np
import torch
import os
import xgboost as xgb
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from graph_based.utils.evaluate import eval_hybrid
from graph_based.utils.loader import  get_dataset
from graph_based.train import train_hybrid
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
def converter(arr: Optional[Union[np.array, torch.Tensor]]) -> np.array:
    if not isinstance(arr, np.ndarray):
        return arr.cpu().numpy()
    return arr


def get_data_graph(nodes,links):
    dataset, _ , adj = get_dataset(nodes, links, nodes.shape[0]) 
    return dataset, adj

class SSLClassifier():
    def __init__(self,
                 estimators: List,
                 X_trains: List,
                 y_train: Array,
                 X_vals: List,
                 y_val: Array,
                 links: None,
                 p: int = 20,
                 n: int = 20,
                 k: int = 50,
                 unlabeled_pool_size: int = 1000,
                 type_ssl: str = 'al',
                 type_estimator: str = 'featurebased'):
        """Khai báo

        Args:
            estimators (List): List các estimators
            X_trains (List): List data train cho các estimators
                            - X_trains length: n_views
                            - X_trains[i] shape: (n_samples, n_features_i)
            y_train (Array): labels data train. 
                            - y_train shape: (n_samples,)
            X_vals (List): Danh sách data validation cho các estimators, nếu estimators_i là graph based thì X_vals[i] = [Nodes,Links].
                            - X_vals length: n_views
                            - X_vals[i] shape: (n_samples, n_features_i) nếu estimators_i không là graph based 
            y_val (Array): labels data val. 
                            - y_val shape: (n_samples,)
            links: Nếu có graphbased thì khai báo thêm links
            p (int, optional): số lượng positive muốn lấy từ data unlabeled vào data train. Defaults to 20.
            n (int, optional): số lượng negative muốn lấy từ data unlabeled vào data train. Defaults to 20.
            type_ssl (str, optional): SemiSupervised learning method, gồm 'al', 'coal', 'co2al' . Defaults to 'al'.
            type_estimator (str, optional): gồm 2 phương pháp 'fb' và 'gb'
        """

        #Nếu không truyền vào estimators_i, SSL method là Active learning
        self.estimators = estimators
        self.links = links
        if len(self.estimators) == 1:
            self.type_ssl = 'al'
            self.type_estimator = 'fb'
        else:
            self.type_ssl = type_ssl
            self.type_estimator = type_estimator
        self.n_views = len(self.estimators)
        self.X_trains, self.y_train = X_trains, y_train
        self.X_vals, self.y_val = X_vals, y_val
        self.class_name_ = "SSLClassifier"
        self.p_, self.n_, self.k, self.unlabeled_pool_size = p, n, k, unlabeled_pool_size

    def fit(self, X_pools: List, y_pool: np.array):
        """_summary_

        Args:
            X_pools (List): Danh sách data unlabeled cho 2 estimators
                            - X_pools length: n_views
                            - X_pools[i] shape: (n_samples, n_features_i)
            y_pool (Array): labels mà user đã report cho unlabeled data. 
                            - y_pool shape: (n_samples,)
            Returns
            -------
            self : returns an instance of self
        """
        acc_1,f_1,acc_2,f_2 = [],[],[],[]
        self.y_train = np.repeat(self.y_train[None, ...], self.n_views, axis=0)
        self.y_val = np.repeat(self.y_val[None, ...], self.n_views, axis=0)
        # machine epsilon
        eps = np.finfo(float).eps
        # number of rounds of co-training
        counter = 0 
        # set of unlabeled samples
        U = np.array(range(X_pools[0].shape[0]))
        # shuffle unlabeled_pool data for easy random access
        np.random.shuffle(U)
        # the small pool of unlabled samples to draw from in training
        unlabeled_pool = U[-min(len(U), self.unlabeled_pool_size):]
        # remove the pool from overall unlabeled data
        U = U[:-len(unlabeled_pool)].tolist()    
        dataset_val, adj_val = get_data_graph(self.X_vals[1], self.links)
        _,  f1, acc, _ = eval_hybrid(self.estimators[1], dataset_val.data, dataset_val.targets, adj_val)
        print(f1, acc)
        while counter < self.k and U:
            counter += 1 
            print(counter)
            temp = [X_pools[0][unlabeled_pool],X_pools[1][unlabeled_pool]]
            # define input cho graph:
            if self.type_estimator == 'gb':
                dataset, adj = get_data_graph(temp[1], self.links)
                # predict
                _, _, _, prob2 = eval_hybrid(self.estimators[1], dataset.data, dataset.targets, adj)
                prob = np.array([
                    converter(self.estimators[0].predict_proba(temp[0][:,1:-1])),
                    converter(prob2)
                ])
            # Lấy index mẫu mới từ unlabeled data bằng 3 Active learning method:
            al_indices = np.unique(np.concatenate(
                (np.repeat(get_random_items(temp[0], round(self.p_ * 0.2)),2,0),
                entropy_sampling(prob, self.p_), margin_sampling(prob, self.p_)),
                axis=1),
                                axis=1)
            # Nếu chỉ dùng Active learning thì ct_indices rỗng
            ct_indices = np.empty((self.n_views, 0), int)
            # Lấy index mẫu mới từ unlabeled data bằng Cotrain method
            if not self.type_ssl == 'al':
                prob = np.log(prob) + eps
                negative_indices = get_index_cotrain(prob[:, :, 0], np.log(0.5),
                                                    self.n_)
                positive_indices = get_index_cotrain(prob[:, :, 1], np.log(0.5),
                                                    self.p_)
                ct_indices = np.unique(np.concatenate(
                    (negative_indices, positive_indices), axis=1),
                                    axis=1)
            # Lấy index train và val:
            query_index_trains, query_index_vals = get_query_index(
                al_indices, ct_indices, self.type_ssl, self.n_views)
            # Cập nhật data train, validation   
            self.y_train = np.concatenate((self.y_train,
                                            np.array(
                                                [y_pool[query_index_trains[i]] for i in range(self.n_views)])),
                                            axis=1)
            self.X_trains = [np.concatenate((self.X_trains[i],
                                             temp[i][query_index_trains[i]]), axis= 0) 
                             for i in range(self.n_views)]
            if self.type_ssl == 'al':
                y_val = np.concatenate((self.y_val,
                                        np.array(
                                            [y_pool[query_index_vals[i]] for i in range(self.n_views)])),
                                        axis=1)
                X_vals = [np.concatenate((self.X_vals[i],
                                                temp[i][query_index_vals[i]]), axis= 0) 
                                for i in range(self.n_views)]
            else:
                self.y_val = np.concatenate((self.y_val,
                                        np.array(
                                            [y_pool[query_index_vals[i]] for i in range(self.n_views)])),
                                        axis=1)
                self.X_vals = [np.concatenate((self.X_vals[i],
                                                temp[i][query_index_vals[i]]), axis= 0) 
                                for i in range(self.n_views)]          
            # fit vào dữ liệu mới
            if self.type_estimator == 'gb':
                dataset_train, adj_train = get_data_graph(self.X_trains[1], self.links)
                dataset_val, adj_val = get_data_graph(self.X_vals[1], self.links)
                self.estimators[1], _ = train_hybrid(self.estimators[1], dataset_train, adj_train, 'adamw', 'multilabel', 'cuda', 5e-4, 500)
                _,  f1, acc, _ = eval_hybrid(self.estimators[1], dataset_val.data, dataset_val.targets, adj_val)
                print(f1, acc)
                acc_2.append(acc)
                f_2.append(f1)    
                #__________________________________________________
                self.estimators[0] = self.estimators[0].fit(self.X_trains[0][:,1:-1], self.y_train[0]) 
                prediction_test1 = self.estimators[0].predict(self.X_vals[0][:,1:-1])
                acc_1.append(accuracy_score(self.y_val[0],prediction_test1)) 
                f_1.append(f1_score(self.y_val[0],prediction_test1))
            unlabeled_pool = U[-min(len(U), self.unlabeled_pool_size):]
            # remove the pool from overall unlabeled data
            U = U[:-len(unlabeled_pool)]
        return acc_1,f_1,acc_2,f_2

    # def get_new_data(self):
    #     y_train_new = np.array([np.concatenate(
    #         (self.y_val,
    #          self.y_train),
    #         axis=1).T])
    #     X_train_new = np.concatenate(
    #         (self.X_vals,
    #          self.X_trains),
    #         axis=1)
    #     array_train_new = np.concatenate((X_train_new, y_train_new),
    #                                      axis=2)
    #     return array_train_new
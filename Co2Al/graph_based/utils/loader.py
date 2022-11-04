from torch.utils.data import Dataset
from graph_based.utils.sparse_matrix import get_matrix, get_adj
from sklearn.preprocessing import PolynomialFeatures, minmax_scale
import torch
import numpy as np
class GraphDataset(Dataset):

    def __init__(self, data, targets) -> None:
        super(GraphDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# def get_dataset(features, links_df, num_nodes):
    
#     links = links_df[links_df['phone_to'].isin(features[:,0])][links_df['phone_in'].isin(features[:,0])]
    
#     links = links.to_numpy().astype(float)
#     labels = torch.from_numpy(features[:, -1]).long()
#     adjs = []
#     for i in range(2, links.shape[1]):
#         adj = get_adj(features, links[:, [0, 1, i]]).float()
#         adjs.append(get_matrix(adj, num_nodes))
#     adj = get_adj(features, links)
#     nrm_adj = get_matrix(adj, num_nodes).float()

#     features = torch.from_numpy(features[:, 1:-1]).float()
#     # features_poly = poly.fit_transform(features)
#     features_poly = minmax_scale(features)
#     features_poly = torch.from_numpy(features_poly).float()
#     dataset = GraphDataset(features_poly, labels)
#     return dataset, nrm_adj, adjs

def get_dataset(features, links_df, num_nodes):
    links = links_df[np.where(np.isin(links_df[:,0],features[:,0]))]
    links = links[np.where(np.isin(links[:,1],features[:,0]))]
    labels = torch.from_numpy(features[:, -1]).long()
    adjs = []
    for i in range(2, links.shape[1]):
        adj = get_adj(features, links[:, [0, 1, i]]).float()
        adjs.append(get_matrix(adj, num_nodes))
    adj = get_adj(features, links)
    nrm_adj = get_matrix(adj, num_nodes).float()

    features = torch.from_numpy(features[:, 1:-1]).float()
    # features_poly = poly.fit_transform(features)
    features_poly = minmax_scale(features)
    features_poly = torch.from_numpy(features_poly).float()
    dataset = GraphDataset(features_poly, labels)
    return dataset, nrm_adj, adjs
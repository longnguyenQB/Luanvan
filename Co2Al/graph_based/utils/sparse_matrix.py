import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


def label_encoder(nodes):
    """Encoder phone numbers

    Args:
        nodes (List): List of all phone numbers

    Returns:
        (sklearn LabelEncoder): Fitted LabelEncoder
    """
    encoder = LabelEncoder()
    encoder.fit(nodes)
    return encoder


def get_indices(encoder, links):
    """Get indices of non-zero position for sparse matrix

    Args:
        encoder (sklearn LabelEncoder): Fitted Encoder
        links (ndarray): ndarray

    Returns:
        ndarray: coordinate of non-zero values
    """
    x = encoder.transform(links[:, 0])
    y = encoder.transform(links[:, 1])
    return np.array([x, y])


def get_adj(nodes, links):
    vals = links[:, 2]
    encoder = label_encoder(nodes[:, 0])
    indices = get_indices(encoder, links)
    return torch.sparse_coo_tensor(indices, vals, [len(nodes), len(nodes)])


def get_normalized_matrix(adj, num_nodes):
    idx = np.array(list(range(num_nodes)))
    vals_diag = np.full((num_nodes, 1), 1).flatten()

    vals = adj.coalesce().values()
    indices = adj.coalesce().indices()

    d_indices = np.append(indices.numpy(), np.array([idx, idx]), axis=1)
    d_vals = np.append(vals.numpy(), vals_diag)
    return torch.sparse_coo_tensor(indices=d_indices,
                                   values=d_vals,
                                   size=(num_nodes, num_nodes))


def get_diag_matrix(adj, num_nodes):
    diag_vals = torch.sparse.sum(adj, dim=1).values()
    diag_vals = 1 / diag_vals
    idx = np.array(list(range(num_nodes)))
    return torch.sparse_coo_tensor(indices=[idx, idx],
                                   values=diag_vals,
                                   size=(num_nodes, num_nodes))


def get_matrix(adj, num_nodes):
    nrm_adj = get_normalized_matrix(adj, num_nodes)
    d_matrix = get_diag_matrix(nrm_adj, num_nodes)
    return torch.sparse.mm(d_matrix, nrm_adj)

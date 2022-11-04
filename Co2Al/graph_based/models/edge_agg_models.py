from torch import nn
from torch.nn import functional as F


class EdgeFC(nn.Module):

    def __init__(self, in_features: int, out_features: int = 1):
        super(EdgeFC, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)

    def forward(self, matrix):
        matrix = self.linear(matrix)
        return F.leaky_relu(matrix).squeeze(-1)


class AggregationEdgeGNN(nn.Module):

    def __init__(self,
                 GNN_Model: nn.Module,
                 edge_in_features: int,
                 edge_out_features: int = 1,
                 n_class: int = 2):
        super(AggregationEdgeGNN, self).__init__()
        self.edge_embedder = EdgeFC(edge_in_features, edge_out_features)
        self.gnn = GNN_Model
        self.linear = nn.LazyLinear(n_class)

    def forward(self, x, adj):
        matrix = self.edge_embedder(adj)
        x = self.gnn(x, matrix)
        out = self.linear(x)
        return F.leaky_relu(out)

import torch
from torch import nn
from torch.nn import functional as F
from torch import nn, optim
from tqdm.auto import tqdm


class ParallelGNN(nn.Module):
    """ParalellGNN
    A simple model for hypergraph with multiple edge properties.
    Each GNN is trained on a edge's feature
    """

    def __init__(self, in_features: int, out_features: int):
        super(ParallelGNN, self).__init__()
        gnns = []
        for i, gnn in enumerate(gnns):
            self.__setattr__(f"gnn_{i}", gnn)
        self.fc = nn.LazyLinear(out_features)

    def forward(self, x, *adjacencies):
        x = x.to(self.device)
        out = torch.tensor([]).to(self.device)
        for i, adj in enumerate(adjacencies):
            adj = adj.to(self.device)
            layer = self.__getattr__(f"gnn_{i}")
            out = torch.cat([out, layer.forward(x, adj)], dim=1)
        out = F.leaky_relu(out)
        out = self.fc(out)
        out = F.leaky_relu(out)
        return out

    
class HybridGNN(nn.Module):

    def __init__(self, out_features, device, *gnns):
        super(HybridGNN, self).__init__()
        self.device = device
        self.leaky_relu = nn.LeakyReLU()
        for i, gnn in enumerate(gnns):
            self.__setattr__(f"gnn_{i}", gnn)
        self.fc = nn.Sequential(nn.LazyLinear(72), nn.LeakyReLU(), nn.Linear(72, 36), nn.ReLU(), nn.Linear(36, out_features))

    def forward(self, x, *adjacencies):
        x = x.to(self.device)
        out = torch.tensor([]).to(self.device)
        for i, adj in enumerate(adjacencies):
            adj = adj.to(self.device)
            layer = self.__getattr__(f"gnn_{i}")
            try:
                out += layer.forward(x, adj)
            except:
                out = torch.cat([out, layer.forward(x, adj)], dim=1)

        out = out/8
        out = self.leaky_relu(out)
        out = self.fc(out)
        out = self.leaky_relu(out)
        return out
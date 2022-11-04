import torch

from models.edge_agg_models import AggregationEdgeGNN
from models.gnn.layers import GraphConvolution
from models.resnet import ResNet

model = ResNet(GraphConvolution, 6, [64, 64, 128, 128, 256], [2, 2, 2, 2, 2],
               None, 4)

x = torch.randn(size=(20, 6)).float()
adj = torch.randn(size=(20, 20, 30))

agg_model = AggregationEdgeGNN(GNN_Model=model,
                               edge_in_features=30,
                               edge_out_features=1,
                               n_class=2)
print(agg_model(x, adj))

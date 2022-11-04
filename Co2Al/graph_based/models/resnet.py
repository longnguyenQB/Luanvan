from torch import nn
from torch.nn import functional as F


def create_res_blocks(Layer, in_features: int, out_features: int,
                      num_residual: int, first_block: bool):
    blk = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            blk.append(
                ResidualBlock(Layer=Layer,
                              in_features=in_features,
                              out_features=out_features,
                              is_first_layer=True))
        else:
            blk.append(
                ResidualBlock(Layer=Layer,
                              in_features=out_features,
                              out_features=out_features,
                              is_first_layer=False))
    return blk


def create_fc_blocks(in_features: int,
                     hidden_layers: list = None,
                     out_features: int = None):
    """Use to make Fully Connected layer

    Args:
        in_features (int): number of features
        hidden_layers (list): configuration for hidden layers
        out_features (int): number of output features

    Returns:
        torch Sequence: MLP
    """
    if out_features is None:
        return None
    if hidden_layers is None:
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features))
    elif len(hidden_layers) == 1:
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_layers[0]),
            nn.Linear(in_features=hidden_layers[0], out_features=out_features))
    else:
        seq = [
            nn.Linear(in_features=in_features, out_features=hidden_layers[0])
        ]
        for i in range(len(hidden_layers) - 1):
            seq.append(
                nn.Linear(in_features=hidden_layers[i],
                          out_features=hidden_layers[i + 1]))
        seq.append(nn.Linear(hidden_layers[-1], out_features))
        return nn.Sequential(*seq)


class ResidualBlock(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, Layer, in_features: int, out_features: int,
                 is_first_layer: bool):
        super(ResidualBlock, self).__init__()
        self.layer_1 = Layer(in_features, out_features)
        self.layer_2 = Layer(out_features, out_features)
        if is_first_layer:
            self.layer_3 = Layer(in_features, out_features)
        else:
            self.layer_3 = None

    def forward(self, X, adj):
        Y = F.leaky_relu(self.layer_1(X, adj))
        Y = F.leaky_relu(self.layer_2(Y, adj))

        if self.layer_3:
            X = self.layer_3(X, adj)
        Y += X
        return F.leaky_relu(Y)


class ResNet(nn.Module):

    def __init__(self,
                 Layer,
                 in_features: int,
                 cfg: list,
                 n_residual_units: list,
                 hidden_layers_fc: list = None,
                 num_classes: int = None):
        """Residual Network with custom Layer

        Args:
            Layer (GraphNeuralLayer): Graph Neural Layer. Graph Neural Layer \
                 with forward method takes (features, adjacency) as input
            in_features (int): Number of features
            cfg (list): configuration for residual block
            n_residual_units (list): number of residual blocks
            hidden_layers_fc (list): number of units in hidden layers. \
                Use in the FullyConnected as classifier.
            num_classes (int, optional): Number of classes. Defaults to 2.
        """
        super(ResNet, self).__init__()
        blk = [Layer(in_features, cfg[0])]
        in_features = cfg[0]
        for i, (num_channels,
                num_residual) in enumerate(zip(cfg[1:], n_residual_units[1:])):
            blk += create_res_blocks(Layer=Layer,
                                     in_features=in_features,
                                     out_features=num_channels,
                                     num_residual=num_residual,
                                     first_block=(i == 0))
            in_features = num_channels
        self.res = nn.ModuleList(blk)
        self.fc = create_fc_blocks(in_features=in_features,
                                   hidden_layers=hidden_layers_fc,
                                   out_features=num_classes)
        # self.dropout = dropout

    def forward(self, x, adjacency):
        for module in self.res:
            x = module(x, adjacency)
        if self.fc is None:
            return x
        x = F.leaky_relu(x)
        for module in self.fc:
            x = module(x)
            x = F.leaky_relu(x)
        return x


class ResNetGAT(nn.Module):

    def __init__(self,
                 Layer,
                 in_features: int,
                 cfg: list,
                 n_residual_units: list,
                 hidden_layers_fc: list = None,
                 num_classes: int = None,
                 n_heads: int = None):
        """Residual Network with custom Layer

        Args:
            Layer (GraphNeuralLayer): Graph Neural Layer. Graph Neural Layer \
                 with forward method takes (features, adjacency) as input
            in_features (int): Number of features
            cfg (list): configuration for residual block
            n_residual_units (list): number of residual blocks
            hidden_layers_fc (list): number of units in hidden layers. \
                Use in the FullyConnected as classifier.
            num_classes (int, optional): Number of classes. Defaults to 2.
            n_heads: n head attention.
        """
        super(ResNetGAT, self).__init__()
        blk = [Layer(in_features, cfg[0], n_heads)]
        in_features = cfg[0]
        for i, (num_channels,
                num_residual) in enumerate(zip(cfg[1:], n_residual_units[1:])):
            blk += create_res_blocks(Layer=Layer,
                                     in_features=in_features,
                                     out_features=num_channels,
                                     num_residual=num_residual,
                                     first_block=(i == 0),
                                     n_heads = n_heads)
            in_features = num_channels
        self.res = nn.ModuleList(blk)
        self.fc = create_fc_blocks(in_features=in_features,
                                   hidden_layers=hidden_layers_fc,
                                   out_features=num_classes)
        # self.dropout = dropout

    def forward(self, x, adjacency):
        for module in self.res:
            x = module(x, adjacency)
        if self.fc is None:
            return x
        x = F.leaky_relu(x)
        for module in self.fc:
            x = module(x)
            x = F.leaky_relu(x)
        return x

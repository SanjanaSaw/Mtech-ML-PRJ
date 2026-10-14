import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    """Configurable-depth Graph Convolutional Network for node classification."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        dimensions = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = torch.nn.ModuleList(
            GCNConv(source, target)
            for source, target in zip(dimensions, dimensions[1:])
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(torch.nn.Module):
    """Configurable-depth Graph Attention Network for node classification."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5, num_layers=3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        dimensions = [in_channels] + [hidden_channels * heads] * (num_layers - 1) + [out_channels]
        layers = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            final_layer = index == num_layers - 1
            layers.append(
                GATConv(
                    source,
                    target if final_layer else hidden_channels,
                    heads=1 if final_layer else heads,
                    concat=False if final_layer else True,
                    dropout=dropout,
                )
            )
        self.convs = torch.nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        x = self.convs[-1](F.dropout(x, p=self.dropout, training=self.training), edge_index)
        return x


from .taf_gnn import TAFGNN

__all__ = ["GCN", "GAT", "TAFGNN"]
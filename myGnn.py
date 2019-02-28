import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, GATConv, SplineConv
# import os.path as osp
# from torch_geometric.nn.glob import global_sort_pool
# from fn import _
# from fn.iters import *
from torch_geometric.data import Data
from myutils import *


class splineN(torch.nn.Module):
    def __init__(self, num_features=None, num_classes=None):
        super(splineN, self).__init__()
        self.conv1 = SplineConv(num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, num_classes, dim=1, kernel_size=2)

    def forward(self, x, edge_index, dropout):
        edge_attr = nt([0.1])
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

def newsplineN(feat,cls):
    gnn=splineN(feat,cls)
    return gnn
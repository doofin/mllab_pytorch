import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from fn import _
# from fn.iters import *
from torch_geometric.data import Data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_features = 1
        num_classes=2
        dim = 32
        # sum mlp : ln -> relu ->ln
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim) # normalize data

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        # read out
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index):
        print("x",x)
        x = F.relu(self.conv1(x, edge_index))


        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        # print("pre norm ",x)
        x = self.bn5(x)
        # print("normed ",x)
        # x = global_add_pool(x, 3)

        x = F.relu(self.fc1(x))
        # print("fc1",x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # print("fc2",x.shape)
        # while(1):{}
        print("x2",x)
        softmaxed = F.log_softmax(x, dim=-1)
        return softmaxed

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# graph level.for node classify,add weight to the node?
def train(data):
    modelp=model
    modelp.train()
    optimizer.zero_grad()

    x_in = data.x
    output = modelp(x_in, data.edge_index) # forward(self, x, edge_index, batch):
    # print("x shape" ,x_in.shape , data.edge_index.shape)
    outputT = torch.t(output)
    # print(outputT.shape, data.y.shape)
    loss = F.nll_loss(outputT, data.y)
    # loss = F.binary_cross_entropy(output, data.y)
    loss.backward()
    loss_all = loss.item()
    optimizer.step()
    print(loss_all)
    return loss_all

def test(dataset):
    modelp=model
    modelp.eval()
    correct = 0
    for data in dataset:
        output = modelp(data.x, data.edge_index)
        pred = output.max(dim=0)[1]
        print("pred : ",pred,"real : " ,data.y)
        correct += pred.eq(data.y).sum().item()
        print("corr",correct)
    return correct / len(dataset)


def newData(nodeFeats,edgeSyms,graphLab):
    return Data(x=torch.tensor(nodeFeats, dtype=torch.float),
                edge_index=torch.tensor(edgeSyms).t().contiguous(),
                y=torch.tensor(graphLab))

dt2=newData([[-1], [0], [1],[1],[1]],
            [[0, 1],
             [1, 0],
             [1, 2],
             [2, 1]],
            [0,1])
for epoch in range(1, 2):
    train(dt2)
    test([dt2])


# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long) # edge connections
# x = torch.tensor([[-1], [0], [1],[1],[1]], dtype=torch.float) # node features
#
# data = Data(x=x, edge_index=edge_index.t().contiguous(),y=torch.tensor([0,1]))

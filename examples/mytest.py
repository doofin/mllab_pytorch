import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GATConv, SplineConv
from fn import _
# from fn.iters import *
from torch_geometric.data import Data

# num_features = 1
num_features = 61
num_classes = 2
dimHid = 32


def nt(x): return torch.tensor(x)


class splineN(torch.nn.Module):
    def __init__(self):
        super(splineN, self).__init__()
        self.conv1 = SplineConv(num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, num_classes, dim=1, kernel_size=2)

    def forward(self, x, edge_index, edge_attr, dropout):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class gatN(torch.nn.Module):
    def __init__(self):
        super(gatN, self).__init__()
        self.att1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.att2 = GATConv(8 * 8, num_classes, dropout=0.6)

    def forward(self, x, edge_index, dropout):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.att1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.att2(x, edge_index)
        return F.log_softmax(x, dim=1)


class gcnNet(torch.nn.Module):
    def __init__(self):
        super(gcnNet, self).__init__()
        self.conv1 = GCNConv(num_features, dimHid, improved=False)
        self.conv2 = GCNConv(dimHid, num_classes, improved=False)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index, dropout):
        x = self.conv2(F.dropout(F.relu(self.conv1(x, edge_index)), training=self.training), edge_index)
        return F.log_softmax(x, dim=1)


class ginNet(torch.nn.Module):
    def __init__(self):
        super(ginNet, self).__init__()

        # sum mlp : ln -> relu ->ln
        nn1 = Sequential(Linear(num_features, dimHid), ReLU(), Linear(dimHid, dimHid))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dimHid)  # normalize data

        nn2 = Sequential(Linear(dimHid, dimHid), ReLU(), Linear(dimHid, dimHid))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dimHid)

        nn3 = Sequential(Linear(dimHid, dimHid), ReLU(), Linear(dimHid, dimHid))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dimHid)

        nn4 = Sequential(Linear(dimHid, dimHid), ReLU(), Linear(dimHid, dimHid))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dimHid)

        nn5 = Sequential(Linear(dimHid, dimHid), ReLU(), Linear(dimHid, dimHid))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dimHid)

        # read out
        self.fc1 = Linear(dimHid, dimHid)
        self.fc2 = Linear(dimHid, num_classes)

    def forward(self, x, edge_index, dropout):
        # print("x",x.shape)
        x = F.relu(self.conv1(x, edge_index))
        # print("x conv1",x.shape)
        # while(1):{}
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x = F.relu(self.fc1(x))
        # print("fc1",x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        # x = F.dropout(x, p=0.5, training=dropout)
        x = self.fc2(x)

        # print("x2",x)
        softmaxed = F.log_softmax(x, dim=-1)
        # print("softmaxed",softmaxed.shape)
        # while(1):{}
        return softmaxed


model = ginNet()  # best 1499 th, 0.6666666666666666 ,highest :  0.7126436781609196
# model = gcnNet()
# model = gatN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


# graph level.for node classify,add weight to the node?
def train(data):
    modelp = model
    modelp.train()
    optimizer.zero_grad()

    x_in = data.x
    output = modelp(x_in, data.edge_index, True)  # forward(self, x, edge_index, batch):
    summed = torch.sum(output, 0)
    # print("sumed:", output.shape,",,", summed.shape)
    # outputT = torch.t(output)
    # print("x shape" ,x_in.shape , data.edge_index.shape)
    # print(outputT.shape, data.y.shape)
    # while(1):{}
    # loss = F.nll_loss(outputT, data.y)
    ypred = data.y
    loss = F.binary_cross_entropy(F.sigmoid(summed), ypred.float())
    # loss = F.binary_cross_entropy(output, data.y)
    loss.backward()
    loss_all = loss.item()
    optimizer.step()
    # print(loss_all)
    return loss_all


def train1(data):
    modelp = model
    modelp.train()
    optimizer.zero_grad()

    x_in = data.x
    output = modelp(x_in, data.edge_index, True)  # forward(self, x, edge_index, batch):
    outputT = torch.t(output)

    # print("x shape" ,x_in.shape , data.edge_index.shape)
    # print(outputT.shape, data.y.shape)
    # while (1): {}
    loss = F.nll_loss(outputT, data.y) # incorrect
    # loss = F.binary_cross_entropy(output, data.y)
    loss.backward()
    loss_all = loss.item()
    optimizer.step()
    # print(loss_all)
    return loss_all


def test(dataset):
    modelp = model
    modelp.eval()
    correct = 0
    datasetLen = len(dataset)
    for data in dataset:
        yreal = data.y
        output = modelp(data.x, data.edge_index, False)
        pred = output.max(dim=0)[1]
        yrealI = torch.argmax(yreal)
        ypredI = torch.argmax(pred)
        eqTotal = 0
        isEq = yrealI.eq(ypredI).item()  # 0 or 1
        eqTotal += isEq
        correct += isEq
    corr = correct / datasetLen
    # print("corr :" ,corr)
    return corr


def newtensor(x): return torch.tensor(x)


def newData(nodeFeats, edgeSyms, graphLab):
    return Data(x=torch.tensor(nodeFeats, dtype=torch.float),  # node features
                edge_index=torch.tensor(edgeSyms).t().contiguous(),  # edge
                y=torch.tensor(graphLab))  # graph label


dt2 = newData([[-1], [0], [1], [1], [1]],
              [[0, 1],
               [1, 0],
               [1, 2],
               [2, 1]],
              [0, 1])


def startT():
    for epoch in range(1, 2):
        train(dt2)
        test([dt2])


def trainOnce(ndfeats, edge, graphlab):
    dt = newData(ndfeats, edge, graphlab)
    train(dt)


# [([ndsFeat],[edges],graphlab)]
def readAsLines(fn):
    f = open(fn, 'r')
    r = f.read().strip().split('\n')
    f.close()
    return r


def fmap(f, xs): return list(map(f, xs))


def readdata(datalen):
    graphList = lambda x: (readAsLines(x + ".n"),
                           readAsLines(x + ".e"),
                           readAsLines(x + ".g"))

    # [([ndsFeatStr],[edges],graphlab)]
    # gdata
    # pat="/home/da/mass/n4jdata1/"
    pat = "/home/da/mass/gdata/"
    graphListTup = fmap(graphList, [pat + str(i) for i in range(0, datalen)])

    # print(graphListTup)
    def edg(e):
        es = e.split(',')
        return (int(es[0]), int(es[1]))

    splitByComma = lambda xx: fmap(lambda x: int(x), xx.split(','))
    # [([ndsFeat Seq],[edgesTup],graphlab Seq)]

    tst = graphListTup[0][2]
    # print(tst)
    print(newtensor([1, 0]).shape)
    # print("ts")
    # print(newtensor(fmap(nodeFeatF,tst)).reshape(2))
    datas = fmap(lambda x: (fmap(splitByComma, x[0]),
                            fmap(edg, x[1]),
                            flatten(fmap(splitByComma, x[2]))),
                 graphListTup)
    # print(datas[0])
    return datas


flatten = lambda l: [item for sublist in l for item in sublist]


def start():
    dataset = readdata(1100) # 1100
    splitat = 87
    dataOk = fmap(lambda x: newData(x[0], x[1], x[2]), dataset)
    trainset = dataOk[splitat:]
    testset = dataOk[:splitat]
    hcoor = 0
    coor = 0
    for epoch in range(1, 900):
        if (coor > hcoor): hcoor = coor
        print(str(epoch) + " th,", coor, ",highest : ", hcoor)
        for dat in trainset:
            train(dat)
        coor = test(testset)


start()


# def printarr(x):
#     # newData(x,[],[])
#     t1=newtensor(x)
#     for epoch in range(1, 200):
#         print(t1)
#     print(type(x))
# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long) # edge connections
# x = torch.tensor([[-1], [0], [1],[1],[1]], dtype=torch.float) # node features
#
# data = Data(x=x, edge_index=edge_index.t().contiguous(),y=torch.tensor([0,1]))

# print("pred", pred, ypredI, "yreal", yrealI, "boo", isEq)
# print("corr ",isEq)
# while(1):{}
# print("pred : ",pred,"real : " ,data.y)
# correct += pred.eq(yreal).sum().item()

# def newtensor(x):return

# x=torch.tensor([[1,2],[1,2]])
# y=torch.tensor([0.1,0])
# l=F.nll_loss(x,y)
# print(x.shape,y.shape,",",l)

# def tstest():
#     inputs_tensor = torch.FloatTensor([
#         [10, 2, 1, -2, -3],
#         [-1, -6, -0, -3, -5],
#         [-5, 4, 8, 2, 1]
#     ])
#     print(inputs_tensor.shape)
#
#     r = F.binary_cross_entropy(F.sigmoid(nt([10.0, 10.0])), nt([0.0, 1.0]))
#     print(r)

import torch
from torch_geometric.data import Data

# https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html
# edge list
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1,1], [0,1], [1,1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
print(
    data.num_features)

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

def nt(x): return torch.tensor(x)
def newtensor(x): return torch.tensor(x)
def fmap(f, xs): return list(map(f, xs))
flatten = lambda l: [item for sublist in l for item in sublist]

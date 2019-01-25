import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops

# from ..inits import reset


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GINConv(torch.nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (nn.Sequential): Neural network :math:`h_{\mathbf{\Theta}}`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool optional): If set to :obj:`True`, :math:`\epsilon` will
            be a trainable parameter. (default: :obj:`False`)
    """

    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.mlp = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x_in, edge_index):
        """"""
        x_in = x_in.unsqueeze(-1) if x_in.dim() == 1 else x_in
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index # i,j  th index for nodes
        # print("gin forward",edge_index.shape ,row.shape)

        sums = (1 + self.eps) * x_in + scatter_add(x_in[col], row, dim=0, dim_size=x_in.size(0)) # eq 4.1
        out = self.mlp(sums)
        return out

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mlp)

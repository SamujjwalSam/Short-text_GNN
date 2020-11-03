import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn import init

from dgl.base import DGLError
from dgl.utils import expand_as_pair


class GraphConv(nn.Module):
    r"""
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, eweight):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
            Note that in the special case of graph convolutional networks, if a pair of
            tensors is given, the latter element will not participate in computation.
        eweight : torch.Tensor of shape (E, 1)
            Values associated with the edges in the adjacency matrix.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                feat_src = th.matmul(feat_src, self.weight)
                graph.srcdata['h'] = feat_src
                graph.edata['w'] = eweight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                 fn.sum('m', 'h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.edata['w'] = eweight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                 fn.sum('m', 'h'))
                rst = graph.dstdata['h']
                rst = th.matmul(rst, self.weight)

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


## GCN with precomputed normalized adjacency matrix and edge features:
# The norm can be computed as follows
def normalize_adj(g):
    in_degs = g.in_degrees().float()
    in_norm = th.pow(in_degs, -0.5).unsqueeze(-1)
    out_degs = g.out_degrees().float()
    out_norm = th.pow(out_degs, -0.5).unsqueeze(-1)
    g.ndata['in_norm'] = in_norm
    g.ndata['out_norm'] = out_norm
    g.apply_edges(fn.u_mul_v('in_norm', 'out_norm', 'eweight'))

    return g


class GNN_Layer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(GNN_Layer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': F.relu(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 2)))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class GCN(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNN_Layer(ndim_in, edim, 50, activation))
        self.layers.append(GNN_Layer(50, edim, 25, activation))
        self.layers.append(GNN_Layer(25, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


if __name__ == '__main__':
    model = GCN(3, 1, 3, F.relu, 0.5)
    g = dgl.DGLGraph([[0, 2], [2, 3]])
    nfeats = torch.randn((g.number_of_nodes(), 3, 3))
    efeats = torch.randn((g.number_of_edges(), 3, 3))
    model(g, nfeats, efeats)

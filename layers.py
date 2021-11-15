import math
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        """
               :param in_features:     size of the input per node
               :param out_features:    size of the output per node
               :param bias:            whether to add a learnable bias before the activation
               :param device:          device used for computation
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class gcnmask(Module):

    def __init__(self, add_all, in_features, out_features, bias=False):
        super(gcnmask, self).__init__()
        self.in_features = in_features
        self.Sig = nn.Sigmoid()
        self.out_features = out_features
        self.add_all = add_all
        self.drop_rate = 0.6
        self.weight_0 = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = []
        self.weights_mask0 = Parameter(torch.FloatTensor(2 * in_features, in_features))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_0.size(1))
        self.weight_0.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        input_new = []
        for i in range(len(self.add_all)):
            index = torch.tensor([[i] * input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i] * input.shape[1]]))
            aa = aa.detach().numpy()
            aa_tile = np.tile(aa, [len(self.add_all[i]), 1])  # expand central
            aa = torch.tensor(aa)
            aa_tile = torch.tensor(aa_tile)
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i] * input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input, 0, torch.tensor(bb_nei_index2))
            cen_nei = torch.cat([aa_tile, bb_nei], 1)
            mask0 = torch.mm(cen_nei, self.weights_mask0)
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.drop_rate)

            self.mask.append(mask0)

            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True)  # hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator
            input_new.append(new_cen_nei)

        input_new = torch.stack(input_new)
        input_new = torch.squeeze(input_new)
        support = torch.mm(input_new, self.weight_0)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttention(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        # self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(
        #     torch.FloatTensor), gain=np.sqrt(2.0)),
        #                       requires_grad=True)
        # self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
        #     torch.FloatTensor), gain=np.sqrt(2.0)),
        #                        requires_grad=True)
        # self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
        #     torch.FloatTensor), gain=np.sqrt(2.0)),
        #                        requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
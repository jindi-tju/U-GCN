import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttention, gcnmask
from torch.nn.parameter import Parameter
import torch
import math


class GCN(nn.Module):
    def __init__(self, add_all, nfeat, nhid, out, dropout):
        print(nhid)
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        # self.gc2 = gcnmask(add_all, nhid, out)
        self.dropout = dropout

    def _mask(self):
        return self.mask

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, dropout, alpha, nheads):
        super(SFGCN, self).__init__()

        # use GCN or GAT
        self.SGAT1 = GAT(nfeat, nhid1, nhid2, dropout, alpha, nheads)
        self.SGAT2 = GAT(nfeat, nhid1, nhid2, dropout, alpha, nheads)
        self.SGAT3 = GAT(nfeat, nhid1, nhid2, dropout, alpha, nheads)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, sadj2, fadj):
        emb1 = self.SGAT1(x, sadj) # Special_GAT out1 -- sadj structure graph
        emb2 = self.SGAT2(x, sadj2) # Special_GAT out2 -- sadj2 feature graph
        emb3 = self.SGAT3(x, fadj)  # Special_GAT out3 -- fadj feature graph
        emb = torch.stack([emb1, emb2, emb3], dim=1)
        emb, att = self.attention(emb)
        return emb, att, emb1, emb2, emb3
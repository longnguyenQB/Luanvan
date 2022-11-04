import torch
import math
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from models.gnn.layers import GraphConvolution

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, concat=True):
        
        super(GATLayer, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)).cuda()
        self.a = Parameter(torch.FloatTensor(2*out_features,1 )).cuda()
        self.leakyrelu = nn.LeakyReLU()
        
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
        """_summary_

        Args:
            input (tensor): 
            adj (sparse_matrix): 

        Returns:
            tensor: Output of 1 head attention
        """        

        h = torch.spmm(input, self.weight).to_sparse()
        if self.bias is not None:
            h = h + self.bias.cuda()


        leakyrelu = nn.LeakyReLU(0.2)  # LeakyReLU

        Wh1 = torch.sparse.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.sparse.mm(h, self.a[self.out_features:, :])

        e = Wh1 + Wh2.T
        # e: n_node x n_node
        e = leakyrelu(e)

        # Masked Attention

        attention =  e.sparse_mask(adj.coalesce())
        attention = torch.sparse.softmax(attention, dim=1)
        
        attention = F.dropout(attention, 0.25)

        h = torch.sparse.mm(attention,h)
  
        return h
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



#_____________________    Sparse version of GAT   _______________    
  
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        
        # set tạm giá trị drop và relu:
        self.dropout = nn.Dropout(0.25)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'



        N = input.size()[0]
        edge = adj.coalesce().indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()
                
        


        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E


        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
"""Sparse version of GAT."""

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads):
        """Sparse version of GAT."""
        super().__init__()

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphConvolution(nhid * nheads, 
                                             72)

    def forward(self, x, adj):
        x = F.dropout(x, 0.6, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, 0.6, training=self.training)
        x = F.leaky_relu(self.out_att(x, adj))
        print(x.shape,x)
        print('______________')
        return x
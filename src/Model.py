import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge
from src.Decoupling_matrix_aggregation import construct_adj

# 
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重矩阵
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #
        if bias:
            # 偏置项
            self.bias = Parameter(torch.FloatTensor(out_features)) #
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass

        # 输入特征与权重相乘
        support = torch.mm(input, self.weight)

        # 图卷积操作，使用邻接矩阵与支持矩阵相乘
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    # __repr__ 方法用于返回对象的字符串表示，可以根据需要进行实现
    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 
class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # 第一层图卷积层，输入特征维度为 nfeat，输出特征维度为 nhid
        self.gc1 = GraphConvolution(nfeat, nhid)

        # 第二层图卷积层，输入特征维度为 nhid，输出特征维度为 nclass
        self.gc2 = GraphConvolution(nhid, nclass)

        # Dropout防止过拟合
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):

        # 第一层图卷积操作
        x = self.gc1(x, adj)

        # 使用 ReLU 激活函数
        if use_relu:
            x = F.relu(x)
        # Dropout 操作
        x = F.dropout(x, self.dropout, training=self.training)

        # 第二层图卷积操作
        x = self.gc2(x, adj)
        return x



class MHGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(MHGCN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        # 第一层异构图卷积层，输入特征维度为 nfeat，输出特征维度为 out
        self.gc1 = GraphConvolution(nfeat, out)

        # 第二层异构图卷积层，输入特征维度为 out，输出特征维度为 out
        self.gc2 = GraphConvolution(out, out)

        # 其他可能的多层异构图卷积层
        # self.gc3 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc4 = GraphConvolution(out, out)
        # self.gc5 = GraphConvolution(out, out)
        # Dropout防止过拟合
        self.dropout = dropout

        """
        Set the trainable weight of adjacency matrix aggregation
        """

        # 定义可训练的邻接矩阵权重参数，以及结构信息权重参数
        # Retail_Rocket
        self.weight_b = torch.nn.Parameter(torch.FloatStorage(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a = 0, b = 0.1)
        self. struct_weight = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        torch.nn.init.uniform_(self.struct_weight, a = 0, b = 0.1)

        # # Alibaba_small
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(7, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(7), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)


        # # Alibaba_large
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(7, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(7), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

        # # Alibaba
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(10, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(15), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)
        # DBLP
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

        # #dblp_small
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)
        
        
        # #imdb_small
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

        # Aminer
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=1)

        # IMDB
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(3), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

    def forward(self, feature, A,encode,use_relu=True):

        """
        前向传播函数

        参数：
        - feature: 输入特征
        - A: 邻接矩阵
        - encode: 结构信息
        - use_relu: 是否使用 ReLU 激活函数

        返回：
        - result: 输出结果
        - (U1 + U2) / 2: 第一层和第二层 GCN 输出的平均值
        - U4: 结构信息对应的 GCN 输出
        """

        # 使用邻接矩阵权重合并邻接矩阵
        final_A = adj_matrix_weight_merge(A, self.weight_b)
        # final_A2 = adj_matrix_weight_merge(A, self.weight_b2)
        # final_A=final_A+torch.eye(final_A.size()[0])

        # final_A2 = adj_matrix_weight_merge(A, self.weight_b2)
        # final_A2=final_A2+torch.eye(final_A2.size()[0])

        # 尝试将输入特征转换为 PyTorch 张量
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # Output of single-layer GCN
        # GCN 第一层和第二层的输出
        U1 = self.gc1(feature, final_A)

        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)
        # return (U1+U2)/2, (U1+U2)/2, (U1+U2)/2

        # 构建结构信息对应的邻接矩阵
        struct_adj=construct_adj(encode,self.struct_weight)
        print(self.struct_weight)
        U3 = self.gc1(feature, struct_adj)
        U4 = self.gc2(U3, struct_adj)
        
        # result=(U1+U2+U4)/2
        # 输出结构信息对应的 GCN 层的结果
        result=((U1+U2)/2+U4)/2
        return result,(U1+U2)/2, U4

        # Output of single-layer GCN
        # U1 = self.gc1(feature, final_A)
        # # Output of two-layer GCN
        # U2 = self.gc2(U1, final_A)
        # # return (U1+U2)/2
        # # struct_adj=construct_adj(encode,self.struct_weight)
        # # print(self.struct_weight)
        # U3 = self.gc1(feature, new_adj)
        # U4 = self.gc2(U3, new_adj)
        # result=((U1+U2)/2+U4)/2
        # #
        # return result, (U1+U2)/2, U4


        # # # Output of single-layer GCN
        # U1 = self.gc1(feature, final_A)
        # # Output of two-layer GCN
        # U2 = self.gc2(U1, final_A)
        # U3 = torch.tanh(self.gc1(feature, new_adj))
        # U4 = torch.tanh(self.gc2(U3, new_adj))
        # return (U2+U4)/2


        # # # Output of single-layer GCN
        # U1 = self.gc1(feature, new_adj)
        # # Output of two-layer GCN
        # U2 = self.gc2(U1, new_adj)
        # return U2,U2,U2
        #
        # # U3 = self.gc3(U2, final_A)
        # # U4 = self.gc4(U2, final_A)
        # # U5 = self.gc5(U2, final_A)
        #
        # # Average pooling
        #
        # return U2

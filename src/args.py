import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    
    # 禁用CUDA训练
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # 训练轮数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    # 初始学习率
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    # 权重衰减（参数的L2损失）
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    # 隐藏单元的数量
    parser.add_argument('--hidden', type=int, default=384,
                        help='Number of hidden units.')
    # 输出的数量
    parser.add_argument('--out', type=int, default=200,
                        help='Number of out.')
    # Dropout率
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    # 使用的数据集
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    # 使用的模型
    parser.add_argument('--model', type=str, default="MHGCN",
                        help='model to use.')
    # 特征类型
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    # 邻接矩阵归一化方法
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    # 近似的阶数
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    # 每个节点的数量，以便平衡
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    # 实验类型
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    # 使用调整后的超参
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
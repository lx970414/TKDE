import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc
import torch.nn.functional as F
from info_nce import InfoNCE
# from src.logreg import LogReg
def load_training_data(f_name):
    """
    加载训练集
    """
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    # 加载测试集
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


def get_score(local_model, node1, node2):
    """
    Calculate embedding similarity
    """
    """
    计算embeds的相似度得分
    """
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        return np.dot(vector1, vector2)
        # return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass


def link_prediction_evaluate(model, true_edges, false_edges):
    """
    Link prediction process
    """
    """
    链路预测
    """

    true_list = list()
    prediction_list = list()
    true_num = 0

    # Calculate the similarity score of positive sample embedding
    # # 计算正样本嵌入的相似性分数
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    # Calculate the the similarity score of negative sample embedding
    # # 计算负样本嵌入的相似性分数
    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    # Determine the positive and negative sample threshold
    # # 确定正负样本的阈值
    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    # Compare the similarity score with the threshold to predict whether the connection exists
    # 将相似性分数与阈值比较，预测链接是否存在
    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def predict_model(model, file_name, feature, A,encode, eval_type, node_matching):
    """
    Link prediction training proces
    """
    """
    链路预测训练过程
    """

    training_data_by_type = load_training_data(file_name + '/train.txt')
    train_true_data_by_edge, train_false_data_by_edge = load_testing_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    network_data = training_data_by_type
    edge_types = list(network_data.keys())  # ['1', '2', '3', '4', 'Base']
    edge_type_count = len(edge_types) - 1
    # edge_type_count = len(eval_type) - 1s

    device = torch.device('cpu')
    contrast = InfoNCE()
    aucs, f1s, prs = [], [], []
    validaucs, validf1s, validprs = [], [], []


    # 169-204行这段代码表示以下操作：
    # 1. 外层循环 for _ in range(1):，实际上只进行一次，因为_不会在循环体内被使用。
    # 2. 内层循环 for iter_ in range(100):，总共进行100次迭代。
    # 3. 将模型（`model`）移动到设备（`device`）上，并定义一个 Adam 优化器（`opt`）。
    # 4. 使用模型进行前向传播，得到嵌入向量 `emb`、`near_embeds` 和 `far_embeds`。
    # 5. 初始化四个空列表，用于存储正样本和负样本的嵌入向量的两个分量。
    # 6. 遍历每个边类型，获取该类型的正样本和负样本。
    # 7. 对于每个正样本，将其两个节点的嵌入向量分别加入对应的列表中。
    # 8. 对于每个负样本，同样将其两个节点的嵌入向量分别加入对应的列表中。
    # 9. 将这四个列表的嵌入向量连接为张量，并进行形状调整，使其成为二维张量。
    # 10. 计算正样本的相似性矩阵 `T1` 和负样本的相似性矩阵 `T2`。
    # 11. 从 `T1` 和 `T2` 中获取对角线元素，分别存储在 `pos_out` 和 `neg_out` 中。
    # 这段代码的目的是进行模型的训练迭代，其中包括模型的前向传播、获取正负样本的嵌入向量、计算相似性矩阵和对角线元素等操作
    for _ in range(1):
        for iter_ in range(100):
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            emb,near_embeds,far_embeds = model(feature, A,encode)

            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    true_edges = train_true_data_by_edge[edge_types[i]]
                    false_edges = train_false_data_by_edge[edge_types[i]]

                for edge in true_edges:
                    # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                    emb_true_first.append(emb[int(edge[0])])
                    emb_true_second.append(emb[int(edge[1])])

                for edge in false_edges:
                    # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                    emb_false_first.append(emb[int(edge[0])])
                    emb_false_second.append(emb[int(edge[1])])

            emb_true_first = torch.cat(emb_true_first).reshape(-1, 200)
            emb_true_second = torch.cat(emb_true_second).reshape(-1, 200)
            emb_false_first = torch.cat(emb_false_first).reshape(-1, 200)
            emb_false_second = torch.cat(emb_false_second ).reshape(-1, 200)

            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)
            # loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
            
            
            # 这段代码表示了模型的训练过程，包括了以下几个步骤：
            # 1. 计算损失函数 `loss`，该损失函数包括两部分：正样本的对数sigmoid损失和负样本的对数sigmoid损失，同时加上一个用于控制两个嵌入向量之间相似性的对比损失（通过 `contrast` 函数计算）。这里的 0.0001 是对比损失的权重，可以根据具体情况进行调整;
            # 2. 将 loss 标记为需要梯度计算;
            # 3. 将梯度清零;
            # 4. 进行反向传播，计算梯度;
            # 5. 更新模型参数;
            # 6. 使用训练好的模型进行测试，得到嵌入向量 `td`;
            # 7. 将嵌入向量转换为NumPy数组，并构建 `final_model` 字典，将节点与其对应的嵌入向量关联起来。根据 `node_matching` 参数，决定是否匹配节点标识;
            # 8. 对每种边类型，分别计算训练集、验证集和测试集的AUC、F1和PR值;
            # 9. 打印当前迭代的损失和权重;
            # 10. 打印训练集、验证集和测试集的AUC和PR值;
            # 11. 记录每次迭代的验证集AUC值，用于后续选取最佳迭代;
            # 12. 记录每次迭代的验证集F1值和PR值;
            # 13. 记录每次迭代的测试集AUC、F1和PR值;
            # 14. 从记录的验证集AUC中找到最大值对应的迭代次数，以及最大验证集F1和PR值对应的迭代次数;
            # 15. 返回最大验证集AUC对应的测试集AUC和PR值。
            
            loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))+ 0.0001 * contrast(near_embeds, far_embeds)
            loss = loss.requires_grad_()

            opt.zero_grad()
            loss.backward()
            opt.step()

            td,near_embeds,far_embeds = model(feature, A,encode)
            td=td.detach().numpy()
            final_model = {}
            try:
                if node_matching == True:
                    for i in range(0, len(td)):
                        final_model[str(int(td[i][0]))] = td[i][1:]
                else:
                    for i in range(0, len(td)):
                        final_model[str(i)] = td[i]
            except:
                td = td.tocsr()
                if node_matching == True:
                    for i in range(0, td.shape[0]):
                        final_model[str(int(td[i][0]))] = td[i][1:]
                else:
                    for i in range(0, td.shape[0]):
                        final_model[str(i)] = td[i]
            train_aucs, train_f1s, train_prs = [], [], []
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    train_auc, train_f1, train_pr = link_prediction_evaluate(final_model,
                                                                              train_true_data_by_edge[edge_types[i]],
                                                                              train_false_data_by_edge[edge_types[i]])
                    train_aucs.append(train_auc)
                    train_f1s.append(train_f1)
                    train_prs.append(train_pr)


                    valid_auc, valid_f1, valid_pr = link_prediction_evaluate(final_model,
                                                                              valid_true_data_by_edge[edge_types[i]],
                                                                              valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(valid_auc)
                    valid_f1s.append(valid_f1)
                    valid_prs.append(valid_pr)

                    test_auc, test_f1, test_pr = link_prediction_evaluate(final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_prs.append(test_pr)

            print("{}\t{:.4f}\tweight_b:{}".format(iter_ + 1, loss.item(), model.weight_b))
            print("train_auc:{:.4f}\ttrain_pr:{:.4f}".format(np.mean(train_aucs),
                                                                              np.mean(train_prs)))
            print("valid_auc:{:.4f}\t\tvalid_pr:{:.4f}".format(np.mean(valid_aucs),
                                                                              np.mean(valid_prs)))
            print("test_auc:{:.4f}\ttest_pr:{:.4f}".format(np.mean(test_aucs),
                                                                           np.mean(test_prs)))
            validaucs.append(np.mean(valid_aucs))
            validf1s.append(np.mean(valid_f1s))
            validprs.append(np.mean(valid_prs))


            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            prs.append(np.mean(test_prs))

    max_iter_aucs = validaucs.index(max(validaucs))
    max_iter_f1s = validf1s.index(max(validf1s))
    max_iter_prs = validprs.index(max(validprs))

    return aucs[max_iter_aucs],  prs[max_iter_prs]

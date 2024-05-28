import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import threading
import os
from time import time
from scipy.sparse import coo_matrix
from utils import test
from early_stopping import early_stopping

class RC(nn.Module):
    def __init__(self, data_config, pretrain_data, pretrain_data2, pretrain_data3):
        super(RC, self).__init__()

        # 参数设置
        self.model_type = 'RC'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data 
        self.pretrain_data2 = pretrain_data2
        self.pretrain_data3 = pretrain_data3

        self.n_users = data_config['n_users'] 
        self.n_items = data_config['n_items'] 
        self.n_fold = 100 
        self.norm_adj = data_config['norm_adj'] 
        self.norm_adj2 = data_config['norm_adj2']
        self.norm_adj3 = data_config['norm_adj3'] 

        self.n_nonzero_elems = self.norm_adj.nnz
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size2) 
        self.weight_size2 = eval(args.layer_size3) 
        self.weight_size3 = eval(args.layer_size4) 

        self.n_layers = len(self.weight_size)  
        self.n_layers2 = len(self.weight_size2) 
        self.n_layers3 = len(self.weight_size3) 

        self.regs = eval(args.regs) 
        self.decay = self.regs[0]
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        
        # 创建模型参数
        self.weights_one = self._init_weights() 
        
        # 输入数据和Dropout的占位符
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = nn.Dropout(p=args.dropout)
        self.mess_dropout = nn.Dropout(p=args.dropout)
        
        # 损失占位符
        self.train_loss = None
        self.train_mf_loss = None
        self.train_emb_loss = None
        self.train_reg_loss = None
        self.train_rec_first = None
        self.train_rec_last = None
        self.train_ndcg_first = None
        self.train_ndcg_last = None
        self.test_loss = None
        self.test_mf_loss = None
        self.test_emb_loss = None
        self.test_reg_loss = None
        self.test_rec_first = None
        self.test_rec_last = None
        self.test_ndcg_first = None
        self.test_ndcg_last = None

        # 创建嵌入
        self.ua_embeddings1, self.ia_embeddings1 = self._create_rc_embed3()
        self.ua_embeddings2, self.ia_embeddings2 = self._create_rc_embed2(self.ua_embeddings1, self.ia_embeddings1)
        self.ua_embeddings3, self.ia_embeddings3 = self._create_rc_embed(self.ua_embeddings2, self.ia_embeddings2)

        self.ua_embeddings = self.ua_embeddings1 + self.ua_embeddings2 + self.ua_embeddings3
        self.ia_embeddings = self.ia_embeddings1 + self.ia_embeddings2 + self.ia_embeddings3
        
    def create_model_str(self): 
        log_dir = '/' + self.alg_type+'/layers_'+str(self.n_layers)+'/dim_'+str(self.emb_dim) 
        log_dir+='/'+args.dataset+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def _init_weights(self): 
        all_weights_one = nn.ParameterDict()
        if self.pretrain_data is None: 
            all_weights_one['user_embedding1'] = nn.Parameter(torch.randn(self.n_users, self.emb_dim))
            all_weights_one['item_embedding1'] = nn.Parameter(torch.randn(self.n_items, self.emb_dim))
            all_weights_one['user_embedding2'] = nn.Parameter(torch.randn(self.n_users, self.emb_dim))
            all_weights_one['item_embedding2'] = nn.Parameter(torch.randn(self.n_items, self.emb_dim))
            all_weights_one['user_embedding3'] = nn.Parameter(torch.randn(self.n_users, self.emb_dim))
            all_weights_one['item_embedding3'] = nn.Parameter(torch.randn(self.n_items, self.emb_dim))
        else:
            all_weights_one['user_embedding1'] = nn.Parameter(torch.FloatTensor(self.pretrain_data3['user_embed']))
            all_weights_one['item_embedding1'] = nn.Parameter(torch.FloatTensor(self.pretrain_data3['item_embed']))
            all_weights_one['user_embedding2'] = nn.Parameter(torch.FloatTensor(self.pretrain_data3['user_embed']))
            all_weights_one['item_embedding2'] = nn.Parameter(torch.FloatTensor(self.pretrain_data3['item_embed']))
            all_weights_one['user_embedding3'] = nn.Parameter(torch.FloatTensor(self.pretrain_data3['user_embed']))
            all_weights_one['item_embedding3'] = nn.Parameter(torch.FloatTensor(self.pretrain_data3['item_embed']))

        all_weights_one['W_u1'] = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))
        all_weights_one['W_u2'] = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))
        all_weights_one['W_i1'] = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))
        all_weights_one['W_i2'] = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))

        return all_weights_one

    def _split_A_hat(self, X): 
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = (i_fold + 1) * fold_len if i_fold < self.n_fold - 1 else self.n_users + self.n_items

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))

        return A_fold_hat

    def _split_A_hat_node_dropout(self, X): 
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = (i_fold + 1) * fold_len if i_fold < self.n_fold - 1 else self.n_users + self.n_items

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].nnz
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout, n_nonzero_temp))

        return A_fold_hat

    def _create_rc_embed(self, user_embedding, item_embedding):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings1 = torch.cat([user_embedding, item_embedding], dim=0)
        all_embeddings = [ego_embeddings1]
        
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings1))

            side_embeddings = torch.cat(temp_embed, dim=0)
            ego_embeddings1 = side_embeddings
            all_embeddings.append(ego_embeddings1)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def _create_rc_embed2(self, user_embedding, item_embedding):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj2)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj2)
        
        ego_embeddings2 = torch.cat([user_embedding, item_embedding], dim=0)
        all_embeddings = [ego_embeddings2]
        
        for k in range(self.n_layers2):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings2))

            side_embeddings2 = torch.cat(temp_embed, dim=0)
            ego_embeddings2 = side_embeddings2
            all_embeddings.append(ego_embeddings2)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings2, i_g_embeddings2 = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings2, i_g_embeddings2        
    
    def _create_rc_embed3(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj3)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj3)
        
        ego_embeddings3 = torch.cat([self.weights_one['user_embedding3'], self.weights_one['item_embedding3']], dim=0)
        all_embeddings = [ego_embeddings3]
        for k in range(self.n_layers3):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings3))

            side_embeddings3 = torch.cat(temp_embed, dim=0)
            ego_embeddings3 = side_embeddings3
            all_embeddings.append(ego_embeddings3)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings3, i_g_embeddings3 = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings3, i_g_embeddings3        

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = torch.norm(self.u_g_embeddings_pre) + torch.norm(self.pos_i_g_embeddings_pre) + torch.norm(self.neg_i_g_embeddings_pre) + torch.norm(self.weights_one['W_u1']) + torch.norm(self.weights_one['W_u2']) + torch.norm(self.weights_one['W_i1']) + torch.norm(self.weights_one['W_i2'])
        regularizer = regularizer / self.batch_size
        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = torch.tensor(0.0, dtype=torch.float32)
        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Sparse tensors的Dropout。
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += torch.rand(noise_shape)
        dropout_mask = torch.floor(random_tensor).bool()
        pre_out = X._values()[dropout_mask]

        return pre_out * torch.div(1., keep_prob)




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        with torch.no_grad():
            self.data = data_generator.sample()

class sample_thread_test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        with torch.no_grad():
            self.data = data_generator.sample_test() 

class train_thread(threading.Thread): 
    def __init__(self, model, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sample = sample
    def run(self):
        users, pos_items, neg_items = self.sample.data
        self.model.optimizer.zero_grad()
        mf_loss, emb_loss, reg_loss = self.model(users, pos_items, neg_items)
        total_loss = mf_loss + emb_loss + reg_loss
        total_loss.backward()
        self.model.optimizer.step()

class train_thread_test(threading.Thread): 
    def __init__(self, model, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sample = sample
    def run(self):
        users, pos_items, neg_items = self.sample.data
        with torch.no_grad():
            mf_loss, emb_loss, _ = self.model(users, pos_items, neg_items)
        total_loss = mf_loss + emb_loss

if __name__ == '__main__':
    torch.manual_seed(42)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    f0 = time() 
    config = dict() 
    n_item = max(data_generator.n_items, data_generator2.n_items, data_generator3.n_items)  
    config['n_users'] = data_generator.n_users 
    config['n_items'] = n_item 

    # Generate adjacency matrices
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat() 
    plain_adj2, norm_adj2, mean_adj2, pre_adj2 = data_generator2.get_adj_mat()
    plain_adj3, norm_adj3, mean_adj3, pre_adj3 = data_generator3.get_adj_mat() 

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix') 
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix') 
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        config['norm_adj2'] = pre_adj2
        config['norm_adj3'] = pre_adj3 
        print('use the pre adjacency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    pretrain_data, pretrain_data2, pretrain_data3 = load_pretrained_data()

    model = RC(data_config=config, pretrain_data=pretrain_data, pretrain_data2=pretrain_data2, pretrain_data3=pretrain_data3)
    model.to(device)
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.save_flag == 1:  
        model_type = 'RC'
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights_one/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)])) 
        ensureDir(weights_save_path)
        # Save model weights here if needed

    run_time = 1
    while True:
        if os.path.exists(tensorboard_model_path + model.log_dir + '/run_' + str(run_time)):
            run_time += 1
        else:
            break

    train_writer = SummaryWriter(tensorboard_model_path + model.log_dir + '/run_' + str(run_time)) 

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], [] 
    stopping_step = 0
    should_stop = False

    for epoch in range(1, args.epoch + 1): 
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0. 
        n_batch = data_generator.n_train // args.batch_size + 1 
        loss_test, mf_loss_test, emb_loss_test, reg_loss_test = 0., 0., 0., 0.

        sample_last = sample_thread() 
        sample_last.start() 
        sample_last.join() 

        for idx in range(n_batch): 
            train_cur = train_thread(model, sample_last) 
            sample_next = sample_thread() 
            
            train_cur.start() 
            sample_next.start()
            
            sample_next.join() 
            train_cur.join() 
            
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch 
            
            sample_last = sample_next 

        summary_train_loss = torch.tensor(loss).unsqueeze(0).to(device)
        summary_train_loss = torch.tensor(mf_loss).unsqueeze(0).to(device)
        summary_train_loss = torch.tensor(emb_loss).unsqueeze(0).to(device)
        summary_train_loss = torch.tensor(reg_loss).unsqueeze(0).to(device)
        train_writer.add_scalar('train/loss', summary_train_loss, epoch)
        train_writer.add_scalar('train/mf_loss', summary_mf_loss, epoch)
        train_writer.add_scalar('train/emb_loss', summary_emb_loss, epoch)
        train_writer.add_scalar('train/reg_loss', summary_reg_loss, epoch)

        if np.isnan(loss):
            print('ERROR: loss is nan.')
            sys.exit()
        
        if (epoch % 5) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        
        users_to_test = list(data_generator.train_items.keys()) 
        ret = test(model, users_to_test, drop_flag=True, train_set_flag=1) 
        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s]' % \
                   (epoch, loss, mf_loss, emb_loss, reg_loss, 
                    ', '.join(['%.5f' % r for r in ret['recall']]),
                    ', '.join(['%.5f' % r for r in ret['precision']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]))
        print(perf_str) 
        summary_train_acc = torch.tensor(ret['recall'][0]).to(device)
        train_writer.add_scalar('train/recall_first', summary_train_acc, epoch)
        summary_train_acc = torch.tensor(ret['recall'][-1]).to(device)
        train_writer.add_scalar('train/recall_last', summary_train_acc, epoch)
        summary_train_acc = torch.tensor(ret['ndcg'][0]).to(device)
        train_writer.add_scalar('train/ndcg_first', summary_train_acc, epoch)
        summary_train_acc = torch.tensor(ret['ndcg'][-1]).to(device)
        train_writer.add_scalar('train/ndcg_last', summary_train_acc, epoch)

class SampleThreadTest(threading.Thread):
    def __init__(self, data_generator):
        super(SampleThreadTest, self).__init__()
        self.data_generator = data_generator

    def run(self):
        self.data = self.data_generator.sample_test()

class TrainThreadTest(threading.Thread):
    def __init__(self, model, sample_last):
        super(TrainThreadTest, self).__init__()
        self.model = model
        self.sample_last = sample_last

    def run(self):
        users, pos_items, neg_items = self.sample_last.data
        with torch.no_grad():
            self.data = self.model(users, pos_items, neg_items)

if __name__ == '__main__':
    t0 = time()

    loss_loger, rec_loger, pre_loger, ndcg_loger = [], [], [], []
    stopping_step = 0
    should_stop = False

    train_writer = SummaryWriter('tensorboard/' + model.log_dir)

    for epoch in range(1, args.epoch + 1):
        t1 = time()

        loss_test, mf_loss_test, emb_loss_test, reg_loss_test = 0., 0., 0., 0.

        sample_last = SampleThreadTest(data_generator)
        sample_last.start()
        sample_last.join()

        for idx in range(n_batch):
            train_cur = TrainThreadTest(model, sample_last)
            sample_next = SampleThreadTest(data_generator)

            train_cur.start()
            sample_next.start()

            sample_next.join()
            train_cur.join()

            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test = train_cur.data

            users, pos_items, neg_items = sample_last.data
            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            emb_loss_test += batch_emb_loss_test / n_batch

            sample_last = sample_next

        summary_test_loss = torch.tensor(loss_test)
        train_writer.add_scalar('Test Loss', summary_test_loss, epoch // 5)
        t2 = time()

        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=True)
        summary_test_acc = torch.tensor(ret['recall'][0])
        train_writer.add_scalar('Test Accuracy', summary_test_acc, epoch // 5)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = f'Epoch {epoch} [{t2 - t1:.1f}s + {t3 - t2:.1f}s]: ' \
                       f'test==[{loss_test:.5f}={mf_loss_test:.5f} + {emb_loss_test:.5f} + {reg_loss_test:.5f}], ' \
                       f'recall=[{", ".join([str(r) for r in ret["recall"]])}], ' \
                       f'precision=[{", ".join([str(p) for p in ret["precision"]])}], ' \
                       f'ndcg=[{", ".join([str(n) for n in ret["ndcg"]])}]'
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        if should_stop:
            break

        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), weights_save_path + '/weights')
            print('save the weights in path:', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = f'Best Iter=[{idx}]@[{time() - t0:.1f}]\trecall=[' \
                 f'{" ".join([str(r) for r in recs[idx]])}], precision=[' \
                 f'{" ".join([str(p) for p in pres[idx]])}], ndcg=[' \
                 f'{" ".join([str(n) for n in ndcgs[idx]])}]'
    print(final_perf)

    save_path = f'{args.proj_path}/output/{args.dataset}/{model.model_type}.result2'
    ensureDir(save_path)
    with open(save_path, 'a') as f:
        f.write(
            f'embed_size={args.embed_size}, lr={args.lr}, layer_size={args.layer_size}, node_dropout={args.node_dropout}, ' \
            f'mess_dropout={args.mess_dropout}, regs={args.regs}, adj_type={args.adj_type}\n\t{final_perf}\n')


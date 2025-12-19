import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from .train_func import *

def init_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight, gain=1.414)


### two acyclic constraint methods
"""
    SCCPowerIteration and PowerIterationGradient is two acyclic constraint methods, 
    refer to https://github.com/azizilab/sdcd   
"""
class SCCPowerIteration(nn.Module):
    def __init__(self, init_adj_mtx, update_scc_freq=1000):
        super().__init__()
        self.d = init_adj_mtx.shape[0]
        self.update_scc_freq = update_scc_freq

        self.device = init_adj_mtx.device

        self.scc_list = None
        self.update_scc(init_adj_mtx)

        self.register_buffer("v", None)
        self.register_buffer("vt", None)
        self.initialize_eigenvectors(init_adj_mtx)

        self.n_updates = 1

    def initialize_eigenvectors(self, adj_mtx):
        # self.v, self.vt = torch.rand(size=(2, self.d), device=self.device)
        self.v, self.vt = torch.ones(size=(2, self.d), device=self.device)
        self.v = normalize(self.v)
        self.vt = normalize(self.vt)
        return self.power_iteration(adj_mtx, 5)

    def update_scc(self, adj_mtx):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=scipy.sparse.coo_matrix(adj_mtx.cpu().detach().numpy()),
            directed=True,
            return_labels=True,
            connection="strong",
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)
        # print(len(self.scc_list))

    def power_iteration(self, adj_mtx, n_iter=5):
        matrix = adj_mtx**2
        for scc in self.scc_list:
            if len(scc) == self.d:
                sub_matrix = matrix
                v = self.v
                vt = self.vt
                for i in range(n_iter):
                    v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                    vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())

                self.v = v
                self.vt = vt

            else:
                sub_matrix = matrix[scc][:, scc]
                v = self.v[scc]
                vt = self.vt[scc]
                for i in range(n_iter):
                    v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                    vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())

                self.v[scc] = v
                self.vt[scc] = vt

        return matrix

    def compute_gradient(self, adj_mtx):
        if (self.n_updates % self.update_scc_freq) == 0:
            # print(self.n_updates)
            self.update_scc(adj_mtx)
            self.initialize_eigenvectors(adj_mtx)

        # matrix = self.power_iteration(4)
        matrix = self.initialize_eigenvectors(adj_mtx)

        gradient = torch.zeros(size=(self.d, self.d), device=self.device)
        for scc in self.scc_list:
            if len(scc) == self.d:
                v = self.v
                vt = self.vt
                gradient = torch.outer(vt, v) / torch.inner(vt, v)
            else:
                v = self.v[scc]
                vt = self.vt[scc]
                gradient[scc][:, scc] = torch.outer(vt, v) / torch.inner(vt, v)

        gradient += 100 * torch.eye(self.d, device=self.device)
        # gradient += matrix.T
        self.n_updates += 1

        return gradient, matrix


class PowerIterationGradient(nn.Module):
    ## 幂迭代法计算矩阵特征向量的梯度，逼近主特征值
    def __init__(self, init_adj_mtx, n_iter=5):
        super().__init__()
        self.d = init_adj_mtx.shape[0]
        self.n_iter = n_iter

        self.device = init_adj_mtx.device

        self.register_buffer("u", None)     # 用于估计矩阵的左特征向量和右特征向量
        self.register_buffer("v", None)

        self.init_eigenvect(init_adj_mtx)

    def init_eigenvect(self, adj_mtx):
        self.u, self.v = torch.ones(size=(2, self.d), device=self.device)
        # self.u, self.v = torch.rand(size=(2, self.d), device=self.device)
        self.u = normalize(self.u)
        self.v = normalize(self.v)
        self.iterate(adj_mtx, self.n_iter)

    def one_iteration(self, A):
        """One iteration of power method"""
        self.u = normalize(A.T @ self.u)
        self.v = normalize(A @ self.v)

    def iterate(self, adj_mtx, n=2):
        with torch.no_grad():
            A = adj_mtx + 1e-6
            for _ in range(n):
                self.one_iteration(A)

    def compute_gradient(self, adj_mtx):
        """Gradient eigenvalue"""
        A = adj_mtx  # **2
        # fixed penalty
        self.iterate(A, self.n_iter)
        # self.init_eigenvect(adj_mtx)
        grad = self.u[:, None] @ self.v[None] / (self.u.dot(self.v) + 1e-6)
        # grad += torch.eye(self.d)
        # grad += A.T
        return grad, A


#### DAGAST modules
class NeighborAttentionLayer(nn.Module):
    def __init__(self, in_features, emb_features, n_neighbor, dk_re, dropout, leakyalpha, concat=True):
        super(NeighborAttentionLayer, self).__init__()

        self.in_features = in_features 
        self.emb_features = emb_features 
        self.n_neighbor = n_neighbor 
        self.dropout = dropout 
        self.alpha = leakyalpha 
        self.concat = concat 

        self.Wh_linear = nn.Linear(1, dk_re)
        self.scaling_factor = math.sqrt(dk_re)

        # attention : gene regulation 
        self.re_Q = nn.Linear(dk_re, dk_re)
        self.re_K = nn.Linear(dk_re, dk_re)

        # attention : cell-cell gene interactions
        self.a_gene_cc = nn.Parameter(torch.empty(size=(dk_re, n_neighbor)))        
        nn.init.xavier_uniform_(self.a_gene_cc.data, gain=1.414)                        # xavier
        # self.cc_K = nn.Linear(n_neighbor, dk_re)

        # attention : cell-cell interactions
        self.W_cell_cc = nn.Parameter(torch.empty(size=(2*in_features, emb_features)))   # cell
        nn.init.xavier_uniform_(self.W_cell_cc.data, gain=1.414)                       # xavier
        self.a_cell_cc = nn.Parameter(torch.empty(size=(2*emb_features, 1)))
        nn.init.xavier_uniform_(self.a_cell_cc.data, gain=1.414)                        # xavier


        self.att_gene_re = None
        self.att_gene_cc = None
        self.att_cell = None

    def cal_attention_re(self, Wh):
        
        Q_re = self.re_Q(Wh)
        K_re = self.re_K(Wh)

        self.att_gene_re = torch.bmm(Q_re, K_re.transpose(1, 2)) / self.scaling_factor    # e * m * 1    e * 1 * m
        self.att_gene_re = F.softmax(self.att_gene_re, dim=-1)              # e * m * m
        self.att_gene_re = F.dropout(self.att_gene_re, self.dropout, training=self.training)


    def cal_attention_cc(self, h, kadj, Wh):
        
        Q_cc = self.re_Q(Wh)

        K_cc = torch.matmul(self.a_gene_cc, h[kadj, :])        # N * dk * m
        # K_cc = self.cc_K(h[kadj, :].transpose(2, 1))

        self.att_gene_cc = torch.bmm(Q_cc, K_cc) / self.scaling_factor    # N * m * m   
        self.att_gene_cc = F.softmax(self.att_gene_cc, dim=-1)            # N * m * m
        self.att_gene_cc = F.dropout(self.att_gene_cc, self.dropout, training=self.training)


    def cal_attention_cell(self, h, kadj):

        Wh_cc = torch.mm(h, self.W_cell_cc)          # h.shape: (N, in_features), Wh.shape: (N, out_features)

        n_neighb = kadj.shape[1]
        Wh1 = torch.matmul(Wh_cc, self.a_cell_cc[: self.emb_features, :])
        Wh2 = torch.matmul(Wh_cc, self.a_cell_cc[self.emb_features :, :])

        self.att_cell = Wh1[kadj, :].squeeze(-1) + Wh2.expand(-1, n_neighb)     # N * k
        self.att_cell = F.leaky_relu(self.att_cell, negative_slope=self.alpha)
        # self.att_cell = F.gelu(self.att_cell)
        self.att_cell = F.softmax(self.att_cell, dim=-1)        # N * k
        self.att_cell = F.dropout(self.att_cell, self.dropout, training=self.training)


    def forward(self, h, kadj):

        Wh = F.relu(self.Wh_linear(h.unsqueeze(-1)))        #   N * m * dk

        # attention : re gene regulation 
        self.cal_attention_re(Wh)
        h_re = torch.bmm(self.att_gene_re, h.unsqueeze(-1)).squeeze(-1)       # N * m
        h_re = h_re + h

        # attention : cc gene regulation
        self.cal_attention_cc(h, kadj, Wh)
        h_cc  = torch.bmm(self.att_gene_cc, h.unsqueeze(-1)).squeeze(-1)                    # N * m
        h_cc = h_cc + h

        h_all = torch.cat([h_re, h_cc], dim=-1)

        # attention : cell-cell interactions
        self.cal_attention_cell(h_all, kadj)             # [N, N, 1] => [N, N]  
        h_all = torch.bmm(self.att_cell.unsqueeze(1), h_all[kadj, :]).squeeze(1) + h_all         # N * m

        if self.concat:
            return F.elu(h_all)
        else:
            return h_all
        
    def get_attention(self):
        print("Single head attention.")
        return self.att_gene_re.detach().cpu().numpy(), self.att_gene_cc.detach().cpu().numpy(), self.att_cell.detach().cpu().numpy()


# class MultiHeadGAT_Niche(nn.Module):
#     def __init__(self, celltypeidx, nfeat, nemb, nhid, dropout, leakyalpha, nheads):
#         """Dense version of GAT."""
#         super(MultiHeadGAT_Niche, self).__init__()
#         self.dropout = dropout
#         self.nheads = nheads

#         self.attentions = [NeighborAttentionLayer(celltypeidx, nfeat, nemb, nhid, dropout, leakyalpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         # self.out_att = NeighborAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         return x
    
#     def get_attention(self):
#         print(f"{self.nheads} head attention.")
#         att_gene_re, att_gene_cc, att_cell = [], [], []
#         for conv in self.attentions:
#             a_gene_re, a_gene_cc, a_cell = conv.get_attention()
#             att_gene_re.append(a_gene_re)
#             att_gene_cc.append(a_gene_cc)
#             att_cell.append(a_cell)

#         return att_gene_re, att_gene_cc, att_cell


class GATmodule_Niche(nn.Module):
    def __init__(
        self, in_channels, emb_channels, n_neighbor, dk_re, 
        num_heads=1, dropout=0.5, leakyalpha=0.1, 
        use_residual=False, resalpha=0.5,
        use_bn=True, bntype='LayerNorm', 
        use_act=True, 
    ):
        super(GATmodule_Niche, self).__init__()

        # if num_heads == 1:
        self.convs = NeighborAttentionLayer(in_channels, emb_channels, n_neighbor, dk_re, dropout, leakyalpha, concat=False)
        # else:
        #     self.convs = MultiHeadGAT_Niche(in_channels, emb_channels, n_neighbor, dropout, leakyalpha, num_heads)

        # n_tmp = 2 if use_residual else 1
        # if bntype == 'LayerNorm':
        #     self.bns = nn.LayerNorm(in_channels + emb_channels)
        # else:
        #     self.bns = nn.BatchNorm1d(in_channels + emb_channels)

        if bntype == 'LayerNorm':
            self.bns = nn.LayerNorm(in_channels * num_heads * 2)
        else:
            self.bns = nn.BatchNorm1d(in_channels * num_heads * 2)

        self.dropout = dropout
        self.activation = nn.LeakyReLU(leakyalpha)
        # self.activation = nn.ReLU()
        # self.use_residual = use_residual
        # self.resalpha = resalpha
        self.use_bn = use_bn
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, adj):
        x = self.convs(x, adj)
        if self.use_act:
            x = self.activation(x)
        if self.use_bn:
            x = self.bns(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def get_attention(self):
        return self.convs.get_attention()


#### DAGAST models
class DAGAST(nn.Module):
    def __init__(self, args, adj, indices_use):
        super(DAGAST, self).__init__()

        num_input = args["num_input"]
        num_emb = args["num_emb"]
        dk_re = args["dk_re"]
        nheads = args["nheads"]
        droprate = args["droprate"]
        leakyalpha = args["leakyalpha"]
        resalpha = args["resalpha"]
        bntype = args["bntype"]
        device = args["device"]
        info_type = args["info_type"]

        self.num_emb = num_emb
        self.leakyalpha = leakyalpha

        self.adjmat = torch.tensor(adj, dtype=torch.long).to(device)
        self.flag = torch.tensor(indices_use, dtype=torch.long).to(device)
        self.emb = None

        self.NeighborAttention = GATmodule_Niche(
            num_input, num_emb, adj.shape[1], dk_re, 
            num_heads=nheads, dropout=droprate, leakyalpha=leakyalpha, 
            use_residual=False, resalpha=resalpha, 
            bntype=bntype
        )

        self.Embedding = nn.Sequential(
            nn.Linear(num_input * nheads * 2, num_emb),
            nn.BatchNorm1d(num_emb),
            nn.LeakyReLU(negative_slope=leakyalpha),

            nn.Linear(num_emb, num_emb),
            nn.BatchNorm1d(num_emb),
            nn.LeakyReLU(negative_slope=leakyalpha),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(num_emb, num_input),
            nn.BatchNorm1d(num_input),
            nn.LeakyReLU(negative_slope=leakyalpha),
            nn.Linear(num_input, num_input),
            nn.LeakyReLU(negative_slope=leakyalpha),
        )

        if info_type == 'nonlinear':
            self.inference = nn.Sequential(
                nn.Linear(num_emb, num_emb),
                nn.BatchNorm1d(num_emb),
                nn.LeakyReLU(negative_slope=leakyalpha),
            )
        else:
            self.inference = nn.Linear(num_emb, num_emb)

        self.Trajectory = None
        self.iter_num = args["iter_num"]
        self.iter_type = args["iter_type"]
        self.sig = torch.nn.Sigmoid()
    

    def forward(self, G, iftrj=False):
        self.emb = self.NeighborAttention(G, self.adjmat)  # feature extraction
        self.emb = self.Embedding(self.emb)
        G = self.Decoder(self.emb[self.flag])       # expression recreate

        emb_ = self.emb[self.flag]  
        emb_1 = self.inference(emb_)
        self.trj_matrix = torch.mm(emb_, emb_1.T)
        self.trj_matrix = self.sig(self.trj_matrix)

        if self.Trajectory == None:     # 懒加载
            if self.iter_type == "SCC":
                self.Trajectory = SCCPowerIteration(self.trj_matrix, update_scc_freq=self.iter_num)
            else:
                self.Trajectory = PowerIterationGradient(self.trj_matrix, n_iter=self.iter_num)

        if iftrj:       
            grad, A = self.Trajectory.compute_gradient(self.trj_matrix)     # 计算 
            # with torch.no_grad():
            #     grad = grad - A * (grad * A).sum() / ((A**2).sum() + 1e-6) / 2
            # grad = grad + torch.eye(grad.shape[0]).to(device=grad.device)
            h_val = (grad * A).sum()

            return self.emb, G, h_val, grad, A
        else:
            return self.emb, G


    def get_encoder_attention(self):
        return self.NeighborAttention.get_attention()


    def get_emb(self, isall=True):
        if isall:
            return self.emb.detach().cpu().numpy()
        return self.emb[self.flag].detach().cpu().numpy()


#### Trainer models
class DAGAST_Trainer(nn.Module):
    def __init__(self, args, st_data, st_data_use):
        super(DAGAST_Trainer, self).__init__()

        self.args = args
        self.st_data = st_data
        self.st_data_use = st_data_use

        n_neighbors = args["n_neighbors"]
        n_type = args["neighbor_type"]
        if n_type == "extern":
            n_externs = args["n_externs"]
            self.kNNGraph_use, self.indices_use, self.st_data_sel = get_neighbor(st_data, st_data_use, n_neighbors=n_neighbors, n_externs=n_externs, ntype="extern")
        else:
            self.kNNGraph_use, self.indices_use = get_neighbor(st_data, st_data_use, n_neighbors=n_neighbors, ntype="noextern")

        self.exp_input = None
        self.indices_torch = None
        self.startflag_torch = None
        self.model = None
        self.optimizer = None
        self.scheduler = None


    def init_train(self):
        ## data
        if self.args["neighbor_type"] == "extern":
            self.exp_input = np_to_torch(self.st_data_sel.X, self.args["device"], requires_grad=False)
        else:
            self.exp_input = np_to_torch(self.st_data_use.X, self.args["device"], requires_grad=False)
        self.indices_torch = np_to_torch(self.indices_use, self.args["device"], dtype=torch.long, requires_grad=False)

        ## model
        self.model = DAGAST(self.args, self.kNNGraph_use, self.indices_use)
        self.model.to(self.args["device"])


    def train_stage1(self, save_path):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["lr"], weight_decay=1e-4)
        if self.scheduler is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler, gamma=0.1)

        print('stage1 training on:', self.args["device"])  
        with tqdm(total=self.args["num_epoch1"]) as t:  
            self.model.train()
            for i in range(self.args["num_epoch1"]):
                emb, exp_predict = self.model(self.exp_input)
                optimizer.zero_grad()
                mse_loss, emb_TV = MyLoss_pre(self.exp_input, exp_predict, emb, self.model.adjmat, self.indices_torch)
                total_loss = mse_loss * self.args["alpha"] + emb_TV * self.args["beta"]
                total_loss.backward()
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()    
                if i % self.args["update_interval"] == 0:
                    t.update(self.args["update_interval"])
                    t.set_postfix({'MSEloss' : mse_loss.item(), 'emb_TV' : emb_TV.item()})
                if emb_TV < 0.01 or mse_loss * self.args["cutof"] > emb_TV:
                    break

        torch.save(self.model, save_path)


    def set_start_region(self, flag, start_flag=None):
        if start_flag == None:
            self.st_data_use.obs['start_cluster'] = 0
            self.st_data_use.obs.loc[flag, 'start_cluster'] = 1
            start_flag = self.st_data_use.obs['start_cluster'].values    # 轨迹起点

        self.startflag_torch = np_to_torch(start_flag, self.args["device"], dtype=torch.long, requires_grad=False)

    def train_stage2(self, save_path, sample_name):
        torch.cuda.empty_cache()     

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["lr"]*0.1, weight_decay=1e-4)
        print('stage2 training on:', self.args["device"])  
        with tqdm(total=self.args["num_epoch2"]) as t:  
            self.model.train()
            for i in range(self.args["num_epoch2"]):
                emb, exp_predict, h_val, grad, trj = self.model(self.exp_input, iftrj=True)
                optimizer.zero_grad()

                mse_loss, emb_TV = MyLoss_post(self.exp_input, exp_predict, emb, h_val, self.model.adjmat, self.indices_torch)
                total_loss = mse_loss * self.args["alpha"] + emb_TV * self.args["beta"] + h_val * self.args["theta1"]

                if self.startflag_torch is not None:
                    startloss = torch.norm(trj[:, self.startflag_torch==1].sum(0), p=1)
                    total_loss = total_loss + startloss * self.args["theta2"]

                total_loss.backward()
                optimizer.step()

                if i % self.args["update_interval"] == 0:
                    t.update(self.args["update_interval"])
                    t.set_postfix({'MSEloss' : mse_loss.item(), 'emb_TV' : emb_TV.item(), 'h_val' : h_val.item(), 's_val' : startloss.item()})

        self.model.eval()
        emb, exp_predict_stage2, h_val, grad, trj = self.model(self.exp_input, iftrj=True)
        trj = trj.detach().cpu().numpy()
        emb = emb[self.indices_torch].detach().cpu().numpy()
        self.st_data_use.obsm['emb'] = emb
        self.st_data_use.obsm['trans'] = trj

        torch.save(self.model, f"{save_path}/model_{sample_name}.pkl")
        np.save(f"{save_path}/trj_{sample_name}.npy", trj)
        np.save(f"{save_path}/emb_{sample_name}.npy", emb)

    
    def get_Trajectory_Ptime(self, knn=30, grid_num=50, smooth=0.5, density=0.7):
        self.st_data_use = get_ptime(self.st_data_use)
        self.st_data_use.uns["E_grid"], self.st_data_use.uns["V_grid"] = get_velocity(
            self.st_data_use, basis="spatial", n_neigh_pos=knn, grid_num=grid_num, smooth=smooth, density=density)



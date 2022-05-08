#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:02:34 2022

https://github.com/WOW5678/CompNet
@inproceedings{wang2019CompNet,
  title="{Order-free Medicine Combination Prediction With Graph Convolutional Reinforcement Learning}",
  author={Shanshan Wang and Pengjie Ren and Zhumin Chen and Zhaochun Ren and Jun Ma and Maarten de Rijke},
  Booktitle={{CIKM} 2019},
  year={2019}
}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.nn.parameter import Parameter
import time

from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)


class CNN(nn.Module):
    def __init__(self,vocab_size,emb_size,num_channels,dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(embedding_dim=emb_size,num_embeddings=vocab_size)
        self.conv=nn.Sequential(
            nn.Conv1d(           #input shape (1, 28, 28)
                in_channels=emb_size,   #input height
                out_channels=num_channels, # n_filters
                kernel_size=3,   # filter size
                stride=1,        # filter movement/step
                padding = 'same'
            ),
            nn.Tanh(),
            #nn.MaxPool2d(kernel_size=2),# choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding = 'same'# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.Tanh(),
            # output shape (16, 28, 28)
            #nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding = 'same'# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
        )
        self.dropout=dropout
        self.out=nn.Linear(num_channels, emb_size,bias=True)
        nn.init.kaiming_normal_(self.out.weight)
        
    def forward(self,x):
        #print('x:',type(x))
        x_emb=self.embedding(x).unsqueeze(0).permute(0,2,1)
        #print('x_emb:',x_emb.shape) #[1, 100, 30]
        x = self.conv(x_emb)
        # average and remove the extra dimension
        remaining_size = x.size(dim=2)
        features = F.max_pool1d(x, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self.dropout)
        output = torch.sigmoid(self.out(features))
        return output


class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN, self).__init__()
        self.W=nn.Parameter(torch.FloatTensor(state_size,state_size))
        nn.init.xavier_uniform_(self.W.data)
        self.U=nn.Parameter(torch.FloatTensor(state_size,state_size))
        nn.init.xavier_uniform_(self.U.data)
        self.fc1=nn.Linear(state_size,512)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2=nn.Linear(512,action_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.learn_step_counter=0
    def forward(self, x, h):
        state= torch.sigmoid(torch.mm(self.W,x.t())+torch.mm(self.U,h.t())).t()
        fc1=F.relu(self.fc1(state))
        output= self.fc2(fc1)
        return output, state

class Agent(object):
    def __init__(self,n_proc, n_diag, n_meds, ehr_adj, emb_dim = 100, filters = 128):
        super(Agent, self).__init__()
        self.np = n_proc
        self.nd = n_diag
        self.nm = n_meds
        self.n_act = self.nm+1
        self.null_token = torch.tensor(self.nm).type(torch.long).to('cuda')
        #self.gamma = 0.9 
        #self.epsilon = 0.9
        #self.epsilon_min = 0.05
        #self.epsilon_decay = 0.995
        self.cnn_diagnosis = CNN(self.nd, emb_dim, filters, 0.5).to('cuda')
        self.cnn_procedure = CNN(self.nd, emb_dim, filters, 0.5).to('cuda')
        self.rgcn= knowledgeGraphEmbed(self.nm, emb_dim, ehr_adj).to('cuda')
        self.model =DQN(emb_dim,self.n_act).to('cuda')
        self.target_model =DQN(emb_dim, self.n_act).to('cuda')
        self.model_params = list(self.cnn_diagnosis.parameters())+\
            list(self.cnn_procedure.parameters()) +list(self.rgcn.parameters())+\
                list(self.model.parameters())
        self.loss=nn.MSELoss()
        self.update_target_model()
        
    def set_to_train(self):
        self.cnn_diagnosis.train()
        self.cnn_procedure.train()
        self.rgcn.train()
        self.model.train()
        
    def set_to_eval(self):
        self.cnn_diagnosis.eval()
        self.cnn_procedure.eval()
        self.rgcn.eval()
        self.model.eval()        
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def load_params(self, path):
        #reload the params
        trainedModel=torch.load(path)
        self.cnn_diagnosis.load_state_dict(trainedModel.cnn_diagnosis.state_dict())
        self.cnn_procedure.load_state_dict(trainedModel.cnn_procedure.state_dict())
        self.rgcn.load_state_dict(trainedModel.rgcn.state_dict())
        self.model.load_state_dict(trainedModel.model.state_dict())
        self.target_model.load_state_dict(trainedModel.target_model.state_dict())
        
    def patient_state(self, p, d):
        diagnosis_f = self.cnn_diagnosis(d)
        procedure_f=self.cnn_procedure(p)
        hsr =torch.cat((diagnosis_f,procedure_f),0)
        return hsr # f.shape:(2,100)

    def treatment(self, state, meds = [], target_model = False):
        m = torch.zeros((1, self.nm)).to('cuda')
        if len(meds)>0:
            m[0,meds]=1
        m = torch.matmul(m, self.rgcn())
        a = torch.softmax(torch.mm(state, m.t()),dim=0)
        f_ = torch.mm(a.t(), state)
        x = f_+ m
        if not target_model:
            a, _ = self.model(x, m)
        else:
            a, _ = self.target_model(x, m)
        return a #x shape(1,100)

    def next_med(self, act, used_meds=[]):
        #used_meds = set(used_meds)
        s = act[0].argsort()
        cntr = 1
        while s[-cntr] in used_meds:
            cntr+=1
        return s[-cntr]
    
    def qval(self, p, d, m):
        state = self.patient_state(p, d)
        return self.treatment(state, m)
    
def train_compnet(train_DL, val_DL, model, epochs, lr = .00002, gamma = 1,
                  target_update_iter = 16, prop_train = .1, epsilon = .2, teacher_force=.5):
    #opt = torch.optim.Adam(model.model_params,
    #         lr=lr,betas=(0.9,0.999),weight_decay=5.0)
    opt = torch.optim.Adam(model.model_params, lr=lr)
    #targ_opt = torch.optim.Adam(model.target_model.parameters(), lr=lr)
    history = {k: [] for k in ['jaccard','recall','precision','f1','loss']}
    best_ja = 0
    for epoch in range(epochs):
        start_time = time.time()
        model.set_to_eval()
        cntr = 0
        n_batch = len(train_DL)
        loss_record = []
        memory = replay_buffer(max_size = 2**16, keys = ['p', 'd', 'm', 'next', 'r', 'meds_left'])
        for p_, d_, m_, mh_, np_, nd_, nm_, nv_ in train_DL:
            cntr+=1
            for i in range(len(p_)):                
                if torch.rand(1) > prop_train: continue
                p= torch.cat(p_[i], dim = 0)
                d = torch.cat(d_[i], dim = 0)
                target_meds = m_[i][-1][torch.randperm(len(m_[i][-1]))]
                steps = len(target_meds)+1
                patient_state = model.patient_state(p,d)
                meds = []
                for step in range(steps):
                    r = torch.rand(1)
                    if r <= teacher_force:
                        if len(target_meds)==0:
                            next_med = model.null_token
                        else:
                            next_med = target_meds[-1]                        
                    elif r <= teacher_force + epsilon:
                        next_med = torch.randint(high = model.nm+1, 
                             size = ()).type(torch.long).to('cuda')
                        while next_med in meds:
                            next_med = torch.randint(high = model.nm+1, 
                             size = ()).type(torch.long).to('cuda')
                    else:
                        action = model.treatment(patient_state, meds)
                        next_med = model.next_med(action, meds)
                    if next_med == model.null_token:
                        if len(target_meds)==0:
                            reward = 2
                        else:
                            reward = -2                           
                    elif len(target_meds)==0:
                        reward = -1
                    elif next_med in target_meds:
                        reward = 1
                    else:
                        reward = -1
                    memory.store_transition({'p':p, 'd':d, 'm':meds,'next':next_med,
                                             'r':reward})
                    if len(target_meds)>0:
                        next_med = target_meds[-1]
                        meds.append(next_med)
                        target_meds = target_meds[target_meds != next_med]
                    
            print(f"\rFilling Mem Buffer {epoch+1}/{epochs}: "+
              f"[{'*'*int(cntr/n_batch*50)+' '*(50-int(cntr/n_batch*50))}]", 
              end = '\r')
        #print(f"{' '*150}\r", end = '\r')
        print('')

        model.set_to_train()
        mem_shuffle = np.arange(memory.mem_cntr).astype(int)
        np.random.shuffle(mem_shuffle)
        cntr = 0
        for i in mem_shuffle:
            cntr+=1
            mdict = memory._get(i)
            state = model.patient_state(mdict['p'], mdict['d'])
            act_vals = model.treatment(state, mdict['m'])
            q_val = act_vals[0,mdict['next']]
            if mdict['next'] != model.null_token:
                new_meds = list(mdict['m'])
                new_meds.append(mdict['next'])
                next_act_val = model.treatment(state, new_meds, target_model = True)
                next_val = next_act_val.max()
            else:
                next_val = torch.tensor(0).to('cuda').type(torch.float)
            q_target = (mdict['r'] + next_val * gamma)
            #if q_val.shape != q_target.shape: print(i)
            #print(q_target.shape)
            #print(q_val.shape)
            loss = model.loss(q_val, q_target)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_record.append(loss.item())
            if cntr%target_update_iter==0:
                model.update_target_model()
            if cntr%64==0:
                print(f"\rReplaying Buffer {epoch+1}/{epochs}: "+
                  f"[{'*'*int(cntr/memory.mem_cntr*50)+' '*(50-int(cntr/memory.mem_cntr*50))}]", 
                  end = '\r')
        #print(f"{' '*150}\r", end = '\r')
        print('')
        js, rs, ps, f1 = eval_model(val_DL, model, return_xy=False)
        history['loss'].append(loss_record)
        history['jaccard'].append(js)
        history['recall'].append(rs)
        history['precision'].append(ps)
        history['f1'].append(f1)
        
        if js > best_ja:
            torch.save(model, open('COMPNet.model','wb'))
            best_ja = js
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f'Epoch: {epoch+1}, Loss: {str(np.mean(loss_record).round(2))}, '+
              f'Epoch Time: {str(np.round(elapsed_time,2))}, '+
              f'Appro Left Time: {str(np.round(elapsed_time * (epochs - epoch - 1), 2))}')
    pickle.dump(history, open('history.pkl', 'wb'))
    print('Completed!')

def eval_model(dataLoader, model, return_xy = True, max_len = 24):
    pmeds = []
    ymeds = []
    model.set_to_eval()
    n_batch=len(dataLoader)
    cntr=0
    for p, d, m, mh, n_p, n_d, n_m, n_v in dataLoader:
        cntr+=1
        b_size = len(p)
        x = torch.zeros((b_size, model.nm)).to('cuda')
        for b in range(b_size):
            meds = []
            state = model.patient_state(torch.cat(p[b]), torch.cat(d[b]))
            step = 0
            while True:
                act = model.treatment(state, meds)
                next_med = model.next_med(act, meds)
                if next_med == model.null_token:  break
                meds.append(next_med)
                step+=1
                if step > max_len:  break
            x[b, meds] = 1
        pmeds.append(x.to('cpu').numpy())
        ymeds.append(mh.to('cpu').numpy())
        print(f"\rBatch {cntr}/{n_batch}: "+
              f"[{'*'*int(cntr/n_batch*50)+' '*(50-int(cntr/n_batch*50))}]",
              end = '\r')
    print(f"{' '*150}\r", end = '\r')
    x = np.concatenate(pmeds, axis=0)
    y = np.concatenate(ymeds, axis=0)
    js = jaccard_score(y, x, average='samples')
    rs = recall_score(y, x, average='samples')
    ps = precision_score(y, x, average='samples', zero_division = 0)
    f1 = f1_score(y, x, average='samples')
    print(f'Jaccard: [{str(js.round(4))}]   Recall: [{str(rs.round(4))}]   '+
    f'Precision: [{str(ps.round(4))}]   F1: [{str(f1.round(4))}]')
    if return_xy:
        return x, y
    else:
            return js, rs, ps, f1
    
      
    #####################v#######v###################################
    #####################v#######v###################################
    #####################v#######v###################################

    
    

'''
Note, the below code is to compare all models on an "apples to apples" basis.
This is borrowed from the GAMEnet work, below.  I use only the first visits
to design the adjacency matrices, as that eliminates data-peeping
'''
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = (1. / np.sqrt(self.weight.size(1)))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class knowledgeGraphEmbed(nn.Module):
    def __init__(self, tokens, dim, adj, bias = False):
        super(knowledgeGraphEmbed, self).__init__()
        self.tokens = tokens
        self.dim = dim
        adj = self.normalize(adj)
        self.adj = torch.FloatTensor(adj).to('cuda')
        self.x = torch.eye(tokens).to('cuda')
        self.dropout = nn.Dropout(p=0.3)
        self.weight = Parameter(torch.FloatTensor(tokens, dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self):
        e = torch.mm(self.x, self.weight)
        a = torch.mm(self.adj, e)
        if self.bias is not None:
            return a + self.bias
        else: return e + a / 2
        
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = (rowsum**-1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def reset_parameters(self):
        stdv = (1. / np.sqrt(self.weight.size(1)))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
               
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        self.adj = torch.FloatTensor(adj).to('cuda')
        self.x = torch.eye(voc_size).to('cuda')

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = (rowsum**-1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
# weighted ehr adj 
def get_weighted_ehr_adj(records, n_meds, only_first_visit = False):
    ehr_adj = np.zeros((n_meds, n_meds))
    if only_first_visit:
        for patient in records:
            med_set = patient[0][2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<i:
                        continue
                    ehr_adj[med_i, med_j] = ehr_adj[med_i, med_j] + 1
                    ehr_adj[med_j, med_i] = ehr_adj[med_j, med_i] + 1
    else:
        for patient in records:
            for adm in patient:
                med_set = adm[2]
                for i, med_i in enumerate(med_set):
                    for j, med_j in enumerate(med_set):
                        if j<i:
                            continue
                        ehr_adj[med_i, med_j] = ehr_adj[med_i, med_j]+1
                        ehr_adj[med_j, med_i] = ehr_adj[med_i, med_j]+1
    return ehr_adj

class replay_buffer:
    def __init__(self, max_size= int(2**11), keys = ['states','holdings','time','rewards']):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.keys = keys
        self.n = len(keys)
        self.buffer = {k:[[] for i in range(self.mem_size)] for k in keys}
    
    # FIFO ordering...
    def store_transition(self, items):
        index = self.mem_cntr % self.mem_size
        for k, v in items.items():
            self.buffer[k][index] = v
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)    
        return {k: np.array([self.buffer[k][b] for b in batch]) for k in self.keys}
    
    def _get(self, idx = None):
        if idx is None:
            max_mem = min(self.mem_cntr, self.mem_size)
            idx = np.random.choice(max_mem, 1)[0]
        #else:
        #    assert isinstance(idx, int)

        return {k: self.buffer[k][idx] for k in self.keys}


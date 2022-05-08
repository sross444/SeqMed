#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: 
https://github.com/sjy1203/GAMENet
@article{shang2018gamenet,
  title="{GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination}",
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  journal={arXiv preprint arXiv:1809.01852},
  year={2018}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import time
import pickle

from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)


class GAMENet(nn.Module):
    #def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
    '''
    Here I NON-use the ddi graph knowledge.  It isn't a target for seqMed,
    so it is not used in training gamenet (for apples to apples comparison)
    '''
    def __init__(self, vocab_size, ehr_adj, emb_dim=100, device=torch.device('cuda:0')):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.nm = vocab_size[2]
        self.vocab_size = vocab_size
        self.device = device
        #self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        #self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        #self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        #self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

        self.to(self.device)
    '''
    One minor change adjusts the inputs from one list of visits into
    3 separate lists (of patients x visist) for each of 
    procedures, diagnoses and meds (p, d, m)
    '''
    # input (adm, 3, codes) --> changed to p, d, m: list of each.
    def forward(self, p, d, m):
        # generate medical embeddings and queries
        p_seq = []
        d_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for i in range(len(p)):
            p_seq.append(mean_embedding(self.dropout(
                self.embeddings[0](p[i].unsqueeze(0)))))
            d_seq.append(mean_embedding(self.dropout(
                self.embeddings[1](d[i].unsqueeze(0)))))
        p_seq = torch.cat(p_seq, dim = 1)
        d_seq = torch.cat(d_seq, dim = 1)

        o1, h1 = self.encoders[0](
            p_seq
        )
        o2, h2 = self.encoders[1](
            d_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn()

        if len(p) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = torch.zeros((len(p)-1, self.vocab_size[2])).to(self.device)
            for idx, m_idx in enumerate(m):
                if idx == len(p)-1:
                    break
                history_values[idx, m_idx] = 1
            # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(p) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()),dim=-1) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

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
               
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

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
        r_inv = np.power(rowsum, -1).flatten()
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
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    else:
        for patient in records:
            for adm in patient:
                med_set = adm[2]
                for i, med_i in enumerate(med_set):
                    for j, med_j in enumerate(med_set):
                        if j<=i:
                            continue
                        ehr_adj[med_i, med_j] = 1
                        ehr_adj[med_j, med_i] = 1
    return ehr_adj


def train_gamenet(train_DL, validate_DL, model, epochs = 40, lr = .0002):
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    history = {k: [] for k in ['jaccard','recall','precision','f1','loss']}
    best_ja = 0
    for epoch in range(epochs):
        loss_record = []
        start_time = time.time()
        model.train()
        cntr = 0
        n_batch = len(train_DL)
        model.train()
        for p_, d_, m_, mh_, np_, nd_, nm_, nv_ in train_DL:
            cntr+=1
            for i in range(len(p_)):
                p= p_[i]; d = d_[i]; loss1_target = mh_[i].unsqueeze(0); m = m_[i]
                loss3_target = np.full((1, model.vocab_size[2]), -1)
                for idx, item in enumerate(m[-1]):
                    loss3_target[0][idx] = item
                target_output = model(p, d, m)
                loss1 = F.binary_cross_entropy_with_logits(target_output, loss1_target)
                loss3 = F.multilabel_margin_loss(torch.sigmoid(target_output), torch.LongTensor(loss3_target).to('cuda'))

                loss = 0.9 * loss1 + 0.01 * loss3
                opt.zero_grad()
                #loss.backward(retain_graph=True)
                loss.backward()
                opt.step()

                loss_record.append(loss.item())
            
            print(f"\rEpoch {epoch+1}/{epochs}: "+
                  f"[{'*'*int(cntr/n_batch*50)+' '*(50-int(cntr/n_batch*50))}]", 
                  end = '\r')
        print(f"{' '*150}\r", end = '\r')
      
        js, rs, ps, f1 = eval_model(validate_DL, model, return_xy=False)
        history['loss'].append(loss_record)
        history['jaccard'].append(js)
        history['recall'].append(rs)
        history['precision'].append(ps)
        history['f1'].append(f1)
        
        if js > best_ja:
            torch.save(model.state_dict(), open('GAMENet.model','wb'))
            best_ja = js
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f'Epoch: {epoch+1}, Loss: {str(np.mean(loss_record).round(2))}, '+
              f'Epoch Time: {str(np.round(elapsed_time,2))}, '+
              f'Appro Left Time: {str(np.round(elapsed_time * (epochs - epoch - 1), 2))}')
    pickle.dump(history, open('GAMENet_history.pkl', 'wb'))
    print('Completed!')
    
def eval_model(dataLoader, model, return_xy = True):
    pmeds = []
    ymeds = []
    model = model.eval()
    n_batch=len(dataLoader)
    cntr=0
    for p, d, m, mh, n_p, n_d, n_m, n_v in dataLoader:
        cntr+=1
        b_size = len(p)
        x = torch.zeros((b_size, model.nm)).to('cuda')
        for b in range(b_size): 
            x[b] = (torch.sigmoid(model(p[b],d[b],m[b]))>.5).type(torch.long)
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
    

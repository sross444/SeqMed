#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:45:13 2022

@author: steven
"""
import torch
import pickle
import torch.nn as nn
import numpy as np
import time

from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)

class gated_attention(nn.Module):
    def __init__(self, dim, dropout = None):
        super(gated_attention, self).__init__()
        self.queries = torch.nn.Linear(dim, dim)
        self.keys = torch.nn.Linear(dim, dim, bias=False)
        self.weight_reduce = nn.Linear(dim,1, bias = False)
        self.values = torch.nn.Linear(dim, dim)
        self.drop = dropout
        if self.drop:
            self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.lnorm = nn.LayerNorm(dim)
        self.deflator = 1/torch.tensor(dim**.5).to('cuda')
    def forward(self, x, c, mask=None):
        # x, c:  batch, ..., seq_len, dim
        # mask:  batch, ..., seq_x, seq_c
        q = torch.tanh(self.queries(x)).unsqueeze(-2) # b, ..., seq_x, 1, dim
        k = torch.tanh(self.keys(c)).unsqueeze(-3) # b, ... 1, seq_c, dim
        w = self.relu(torch.tanh(self.weight_reduce(q+k))).squeeze(-1) # b, ..., seq_x, seq_c
        if mask is not None:
            w = w * mask
        v = self.values(c) * self.deflator # batch, ..., seq_c, dim
        add = torch.matmul(w, v) # batch, ..., seq_x, dim
        if self.drop:
            add = self.dropout(add)
        return self.lnorm(x + add)

class patient_representation(nn.Module):
    def __init__(self, num_embeddings, dim, layers = 2, dropout = None):
        super(patient_representation, self).__init__()
        self.embed = torch.nn.Embedding(num_embeddings, dim)
        self.nlayers = layers
        self.med_attn = nn.ModuleList([gated_attention(dim, dropout) for _ in range(layers)])
        self.visit_attn = nn.ModuleList([gated_attention(dim, dropout) for _ in range(layers)])
    def forward(self, x, mask = None):
        med_len = x.shape[2]
        e = self.embed(x)
        m = mask.unsqueeze(2).repeat(1,1,med_len,1)
        for l in range(self.nlayers):
            e = self.med_attn[l](e, e, m)
        # e: batch, visit_len, seq_x, dim
        if mask is not None:
            e += (mask.unsqueeze(-1)-1)*1e9
        e = e.max(2)[0]
        m = (mask.sum(2)>0).type(torch.long).unsqueeze(-1) # b, visit, 1
        e = e*m
        q = e[:,-1].unsqueeze(1)# batch, 1, dim
        k = e[:,:-1] # batch, seq_x-1, dim
        m = m[:,:-1].transpose(1,2)
        for l in range(self.nlayers):
            q = self.visit_attn[l](q, k, m)
        return q
    
class state_patient_attention(nn.Module):
    def __init__(self, dim):
        super(state_patient_attention, self).__init__()
        self.patient_linear = nn.Linear(dim, dim, bias = False)
        self.state_linear = nn.Linear(dim, dim)
        self.agg = nn.Linear(dim, 1)
    def forward(self, state, patient):
        p = self.patient_linear(patient) # batch, 2, dim
        state = self.state_linear(state) # batch, 1, dim
        x = torch.tanh(p + state.repeat(1,2,1))
        w = torch.softmax(self.agg(x),dim=1) # batch, 2, 1
        return (patient * w).sum(dim=1) # batch, dim
        
class LEAP(nn.Module):
    def __init__(self,n_proc, n_diag, n_meds, med_count_dict,
                 dim = 64, dff = 256, layers = 4, dropout = .1):
        super(LEAP, self).__init__()
        self.dim = dim
        self.np = n_proc
        self.nd = n_diag
        self.nm = n_meds
        self.nlayers = layers
        self.drop = dropout
        self.mcd = med_count_dict
        if self.drop:
            self.dropout = nn.Dropout(dropout)
        self.start_token = torch.tensor(self.nm+1).type(torch.long).to('cuda')
        self.end_token = torch.tensor(self.nm).type(torch.long).to('cuda')

        self.cnn_procedure = patient_representation(self.np, dim, layers = layers, dropout = dropout).to('cuda')
        self.cnn_diagnosis = patient_representation(self.nd, dim, layers = layers, dropout = dropout).to('cuda')

        # embed all meds, including "start_token"
        # use parameter matrix to allow for "global embedding" step
        self.med_embed =nn.Parameter(torch.FloatTensor(self.nm+2,dim))
        nn.init.xavier_uniform_(self.med_embed.data)
        
        self.med_attn = nn.ModuleList([gated_attention(dim, dropout) for _ in range(layers)])
        
        self.dff = nn.Linear(dim, dff, bias = True)
        self.relu = nn.ReLU()
        self.final = nn.Linear(dff, self.nm+1) # exclude start
        
    def hsr(self, p, d, pm = None, dm = None):
        diagnosis_f = self.cnn_diagnosis(d, dm)
        procedure_f=self.cnn_procedure(p, pm)
        hsr =torch.cat([diagnosis_f,procedure_f],dim=1)
        return hsr # f.shape:(batch, 2, dim)

    def single_step(self, hsr, med_vec = None):
        # hsr: batch, 2, dim
        if med_vec is None:
            # get start_token embedding
            med_vec = self.med_embed[self.start_token].unsqueeze(0).repeat(
                hsr.shape[0],1).unsqueeze(1) # b , 1, dim
        for l in range(self.nlayers):
            hsr = self.med_attn[l](hsr, med_vec)
        x = self.dff(hsr).sum(1)
        #pred = torch.softmax(self.final(self.relu(x)), dim = -1)
        pred = self.final(self.relu(x))
        return pred, hsr
    
    def predict(self, p, d, pm=None, dm=None, T=36):
        hsr = self.hsr(p, d, pm, dm)
        rv = []
        med_vec = None
        for step in range(T):
            pred, hsr = self.single_step(hsr, med_vec)
            rv.append(torch.sigmoid(pred))
            # global embedding pooling...
            med_vec = torch.mm(torch.softmax(pred, dim = 1),
                       self.med_embed[:-1]).unsqueeze(1) # batch, 1, dim
        return torch.stack(rv, dim=1)
    
    def discrete(self, p, d, pm=None, dm=None, T=36):
        preds = self.predict(p,d,pm,dm,T)
        b_size = preds.shape[0]
        still_prescribing = [True for _ in range(b_size)]
        cntr = 0
        rvals = [[] for _ in range(b_size)]
        while any(still_prescribing) and cntr < T:
            s = preds[:,cntr].argsort(dim=1)
            for b in range(b_size):
                if not still_prescribing[b]: continue
                idx_cntr = 1
                while s[b,-idx_cntr] in rvals[b]:
                    idx_cntr+=1
                if s[b,-idx_cntr] == self.end_token:
                    still_prescribing[b] = False
                else:
                    rvals[b].append(s[b,-idx_cntr])
            cntr+=1
        return rvals

def train_leap(train_DL, val_DL, model, epochs = 1, lr = 1e-4, T = 24):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {k: [] for k in ['jaccard','recall','precision','f1','loss']}
    best_ja = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        cntr = 0
        n_batch = len(train_DL)
        loss_record = []
        for p_, d_, m_, mh, n_p, n_d, n_m, n_v in train_DL:
            cntr+=1
            b_size = len(p_)
            p, d, pm, dm = pad(p_, d_, n_p, n_d, n_v)
            
            meds = [j[-1] for j in m_]
            n_meds = [j.shape[0] for j in meds]
            max_meds = max(n_meds)+1
            # !!! Key Element - building targets 
            targs= torch.cat([
                mh.unsqueeze(1).repeat(1,max_meds,1),
                torch.zeros((b_size, max_meds,1)).to('cuda')], dim = 2)
            loss_masks = torch.ones((b_size, max_meds)).to('cuda')
            for b in range(b_size):
                loss_masks[b,(n_meds[b]+1):] = 0
                # random shuffle real meds
                #meds[b][:n_meds[b]] = meds[b][:n_meds[b]][torch.randperm(n_meds[b])]
                # NO - NOT FOR LEAP MODEL, WHERE ORDER IS PRESERVED!
                # NO - SHUFFLE - I choose Least Frequent First
                sort_vals = [model.mcd[j] for j in meds[b].to('cpu').numpy()]
                meds[b][:n_meds[b]] = meds[b][:n_meds[b]][np.argsort(sort_vals)]
                # after each token token is added to set, its target becomes 0
                for s in range(n_meds[b]):
                    targs[b][(s+1):,meds[b][s]]=0
                # finally, predict end token
                targs[b][(s+1),-1] = 1

            preds = model.predict(p, d, pm, dm, T = max_meds)
            loss = nn.BCELoss(reduction='none')(preds,targs) * loss_masks.unsqueeze(-1)
            loss = loss.sum((1,2)).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_record.append(loss.item())
            
            print(f"\rEpoch {epoch+1}/{epochs}: "+
                  f"[{'*'*int(cntr/n_batch*50)+' '*(50-int(cntr/n_batch*50))}]", 
                  end = '\r')
        print(f"{' '*150}\r", end = '\r')
      
        js, rs, ps, f1 = eval_model(val_DL, model, return_xy=False, T = T)
        history['loss'].append(loss_record)
        history['jaccard'].append(js)
        history['recall'].append(rs)
        history['precision'].append(ps)
        history['f1'].append(f1)
        
        if js > best_ja:
            torch.save(model.state_dict(), open('LEAP.model','wb'))
            best_ja = js
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f'Epoch: {epoch+1}, Loss: {str(np.mean(loss_record).round(2))}, '+
              f'Epoch Time: {str(np.round(elapsed_time,2))}, '+
              f'Appro Left Time: {str(np.round(elapsed_time * (epochs - epoch - 1), 2))}')
    pickle.dump(history, open('LEAP_history.pkl', 'wb'))
    print('Completed!')
    
def eval_model(dataLoader, model, T = 36, return_xy = True):
    pmeds = []
    ymeds = []
    model = model.eval()
    n_batch=len(dataLoader)
    cntr=0
    for p_, d_, m_, mh, n_p, n_d, n_m, n_v in dataLoader:
        cntr+=1
        p, d, pm, dm = pad(p_, d_, n_p, n_d, n_v)
        discrete = model.discrete(p,d,pm,dm,T)
        b_size = len(p)
        x = torch.zeros((b_size, model.nm)).to('cuda')
        for b in range(b_size): 
            x[b, discrete[b]]=1
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


def pad(p, d, n_p, n_d, n_v):
    b_size = len(p)
    max_visits = max(n_v)
    max_p = max(n_p)
    max_d = max(n_d)
    
    p_ = torch.zeros((b_size, max_visits, max_p)).type(torch.long).to('cuda')
    pm = p_.clone()
    d_ = torch.zeros((b_size, max_visits, max_d)).type(torch.long).to('cuda')
    dm = d_.clone()
    
    for b in range(b_size):
        first_visit = max_visits-n_v[b]
        for v in range(n_v[b]):
            p_[b,first_visit+v, :len(p[b][v])] = p[b][v]
            pm[b,first_visit+v, :len(p[b][v])] = 1
            d_[b,first_visit+v, :len(d[b][v])] = d[b][v]
            dm[b,first_visit+v, :len(d[b][v])] = 1            
    return p_, d_, pm, dm

def med_count_dict(patients, med_dict):
    cdict = {k:0 for k, v in med_dict.word_dict.items()}
    for p in patients:
        for m in p[0][2]:
            cdict[m]+=1
    return cdict

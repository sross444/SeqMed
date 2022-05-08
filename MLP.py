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
import time

from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)

        
class MLP(nn.Module):
    def __init__(self,n_proc, n_diag, n_meds, dim = 512, dropout = .2):
        super(MLP, self).__init__()
        self.dim = dim
        self.np = n_proc
        self.nd = n_diag
        self.nm = n_meds
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(self.np+self.nd, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, self.nm)
        
    def forward(self, x):
        #b_size = p.shape[0]
        x = torch.tanh(self.l1(x))
        x = self.dropout(self.relu(self.l2(x)))
        return self.l3(x)

def train_MLP(train_DL, val_DL, model, epochs = 1, lr = 1e-3):
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
            pro = torch.zeros((b_size, model.np), device = 'cuda')
            dia = torch.zeros((b_size, model.nd), device = 'cuda')
            for b in range(b_size):
                for v in range(len(p_[b])):
                    pro[b,p_[b][v]]+=1
                    dia[b,d_[b][v]]+=1
            x=torch.cat([pro, dia], dim =1)
            
            margin_target = torch.full((b_size, model.nm), -1).to('cuda').type(torch.long)
            for b in range(b_size):
                for idx, item in enumerate(m_[b][-1]):
                    margin_target[b][idx] = item
            
            preds = model(x)
            bce_loss = F.binary_cross_entropy_with_logits(preds, mh)
            margin_loss = F.multilabel_margin_loss(
                torch.sigmoid(preds), margin_target)
            
            loss = .5 * bce_loss + .5 * margin_loss
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_record.append(loss.detach().to('cpu').numpy())
            
            print(f"\rEpoch {epoch+1}/{epochs}: "+
                  f"[{'*'*int(cntr/n_batch*50)+' '*(50-int(cntr/n_batch*50))}]", 
                  end = '\r')
        print(f"{' '*150}\r", end = '\r')
      
        js, rs, ps, f1 = eval_model(val_DL, model, return_xy=False)
        history['loss'].append(loss_record)
        history['jaccard'].append(js)
        history['recall'].append(rs)
        history['precision'].append(ps)
        history['f1'].append(f1)
        
        if js > best_ja:
            torch.save(model.state_dict(), open('MLP.model','wb'))
            best_ja = js
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f'Epoch: {epoch+1}, Loss: {str(np.mean(loss_record).round(2))}, '+
              f'Epoch Time: {str(np.round(elapsed_time,2))}, '+
              f'Appro Left Time: {str(np.round(elapsed_time * (epochs - epoch - 1), 2))}')
    pickle.dump(history, open('MLP_history.pkl', 'wb'))
    print('Completed!')

    
def eval_model(dataLoader, model, return_xy = True):
    pmeds = []
    ymeds = []
    model = model.eval()
    n_batch=len(dataLoader)
    cntr=0
    for p_, d_, m_, mh, n_p, n_d, n_m, n_v in dataLoader:
        cntr+=1
        b_size = len(p_)
        pro = torch.zeros((b_size, model.np), device = 'cuda')
        dia = torch.zeros((b_size, model.nd), device = 'cuda')
        for b in range(b_size):
            for v in range(len(p_[b])):
                pro[b,p_[b][v]]+=1
                dia[b,d_[b][v]]+=1
        x=torch.cat([pro, dia], dim =1)
        preds = model(x)
        x = (preds > .5).type(torch.int)
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


def pad_seq(x):
    b_size = len(x)
    x = [torch.cat(j) for j in x]
    xs = [j.shape[0] for j in x]
    max_x = max(xs)
    rvals = torch.zeros((b_size, max_x)).type(torch.long).to('cuda')
    mask = torch.zeros((b_size, max_x)).type(torch.long).to('cuda')
    for b in range(b_size):
        rvals[b, :xs[b]] = x[b]
        mask[b, :xs[b]] = 1
    return rvals, mask





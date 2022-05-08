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
        x_emb=self.embedding(x).permute(0,2,1)
        #print('x_emb:',x_emb.shape) #[1, 100, 30]
        x = self.conv(x_emb)
        # average and remove the extra dimension
        remaining_size = x.size(dim=2)
        features = F.max_pool1d(x, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self.dropout)
        output = torch.sigmoid(self.out(features))
        return output

    
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
        
class SeqMed(nn.Module):
    def __init__(self,n_proc, n_diag, n_meds, dim = 128):
        super(SeqMed, self).__init__()
        self.dim = dim
        self.np = n_proc
        self.nd = n_diag
        self.nm = n_meds
        self.start_token = torch.tensor(self.nm+1).type(torch.long).to('cuda')
        self.end_token = torch.tensor(self.nm).type(torch.long).to('cuda')

        self.cnn_diagnosis = CNN(self.nd, dim, dim, 0.5).to('cuda')
        self.cnn_procedure = CNN(self.nd, dim, dim, 0.5).to('cuda')
        
        # embed all meds, including "start_token"
        # use parameter matrix to allow for "global embedding" step
        self.med_embed =nn.Parameter(torch.FloatTensor(self.nm+2,dim))
        nn.init.xavier_uniform_(self.med_embed.data)
        
        self.state_pat_attn = state_patient_attention(dim)
        
        self.gru = nn.GRU(input_size = dim, hidden_size = dim, batch_first=True)
        
        self.linear = nn.Linear(dim, dim)
        self.final = nn.Linear(dim, self.nm+1) # exclude start
        
    def hsr(self, p, d):
        #b_size = p.shape[0]
        b_size = len(p)
        diag, proc = [], []
        for b in range(b_size):
            diag.append(self.cnn_diagnosis(torch.cat(d[b]).unsqueeze(0)))
            proc.append(self.cnn_procedure(torch.cat(p[b]).unsqueeze(0)))
        hsr =torch.stack([torch.cat(diag),torch.cat(proc)],dim=1)
        return hsr # f.shape:(batch, 2, dim)

    def single_step(self, hsr, state = None, med_vec = None):
        # hsr: batch, 2, dim
        # med_vec: 1, batch, dim (or None)
        if med_vec is None:
            med_vec = self.med_embed[self.start_token].unsqueeze(0).repeat(
                hsr.shape[0],1).unsqueeze(1) # batch, 1, dim
        if state is None:
            state = torch.zeros((hsr.shape[0],1,hsr.shape[2])).to('cuda')
        c = self.state_pat_attn(state, hsr) # c: batch, dim
        
        # state: b, 1, dim
        # med_vec: 1, b, dim
        # c: 1, b, dim
        c = c.unsqueeze(0)
        state, hidden = self.gru(med_vec, c)
        h_lin = self.linear(hidden.squeeze(0))
        pred = self.final(h_lin) # batch x dim
        return pred, state

    def predict(self, p, d, T=36):
        hsr = self.hsr(p, d)
        rv = []
        med_vec = None
        state = None
        for step in range(T):
            pred, state = self.single_step(hsr, state, med_vec)
            # global pooling...
            med_vec = torch.mm(torch.softmax(pred, dim =1),
                       self.med_embed[:-1]).unsqueeze(1) # batch, 1, dim
            rv.append(pred)
        return torch.sigmoid(torch.stack(rv, dim=1))
    
    def discrete(self, p, d, T=36):
        preds = self.predict(p,d,T)
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

class discriminator(nn.Module):
    def __init__(self, n_medications):
        super(discriminator, self).__init__()
        self.gru = nn.GRU(input_size = n_medications+1, hidden_size = 64,
                              batch_first = True)
        self.linear = nn.Linear(64, 1)
    def reward(self, meds):
        x, h = self.gru(meds.unsqueeze(dim=1))
        h = torch.tanh(h.squeeze(dim=0))
        return torch.sigmoid(self.linear(h))
    def discriminate(self, pairs):
        x1, h1 = self.gru(pairs[:,0,:].unsqueeze(dim=1))
        h1 = torch.tanh(h1.squeeze(dim=0))
        h1 = self.linear(h1)
        x2, h2 = self.gru(pairs[:,1,:].unsqueeze(dim=1))
        h2 = torch.tanh(h2.squeeze(dim=0))
        h2 = self.linear(h2)
        return torch.sigmoid(torch.cat([h1, h2], dim = 1))

def train_SeqMed(train_DL, val_DL, model, discriminator, epochs = 1, lr = 1e-2, T = 24):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dopt = torch.optim.Adagrad(discriminator.parameters(), lr = lr)
    criterion = torch.nn.BCELoss(reduction='none')
    history = {k: [] for k in ['jaccard','recall','precision','f1','loss']}
    best_ja = 0
    for epoch in range(epochs):
        model.train()
        discriminator.train()
        start_time = time.time()
        cntr = 0
        n_batch = len(train_DL)
        loss_record = []
        dloss_record = []
        aloss_record = []
        for p_, d_, m_, mh, n_p, n_d, n_m, n_v in train_DL:
            cntr+=1
            b_size = len(p_)
            #p, pm = pad_seq(p_)
            #d, dm = pad_seq(d_)
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
                # after each token token is added to set, its target becomes 0
                #for s in range(n_meds[b]):
                #    targs[b][(s+1):,meds[b][s]]=0
                # finally, predict end token
                targs[b,n_meds[b]:,-1] = 1
                
            preds = model.predict(p_, d_, T = max_meds)
            #loss = nn.BCELoss(reduction='none')(preds,targs) * loss_masks.unsqueeze(-1)
            loss = criterion(preds,targs)* loss_masks.unsqueeze(-1)
            loss = loss.sum(1).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_record.append(loss.item())
            
            # train discriminator
            gen_meds = model.predict(p_,d_,T=max_meds)
            pairs = torch.stack([
                gen_meds.reshape((-1,gen_meds.shape[-1])),
                targs.reshape((-1,gen_meds.shape[-1]))],dim=1)
            dtargs = torch.stack([torch.zeros((pairs.shape[0])),
                      torch.ones((pairs.shape[0]))], dim = 1).to('cuda')
            dis = discriminator.discriminate(pairs)
            loss = criterion(dis,dtargs)*loss_masks.flatten().unsqueeze(1)
            loss = loss.sum(1).mean()
            loss.backward()
            dopt.step()
            dopt.zero_grad()
            dloss_record.append(loss.item())            
            
            # adversarially train generator
            gen_meds = model.predict(p_,d_,T=max_meds)
            reward = discriminator.reward(gen_meds.reshape((-1,gen_meds.shape[-1])))
            loss = criterion(reward, torch.ones((reward.shape[0],1)).to('cuda')) *loss_masks.flatten().unsqueeze(1)
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            aloss_record.append(loss.item())
            
            # train discriminator again
            gen_meds = model.predict(p_,d_,T=max_meds)
            pairs = torch.stack([
                gen_meds.reshape((-1,gen_meds.shape[-1])),
                targs.reshape((-1,gen_meds.shape[-1]))],dim=1)
            dtargs = torch.stack([torch.zeros((pairs.shape[0])),
                      torch.ones((pairs.shape[0]))], dim = 1).to('cuda')
            dis = discriminator.discriminate(pairs)
            loss = criterion(dis,dtargs)*loss_masks.flatten().unsqueeze(1)
            loss = loss.sum(1).mean()
            loss.backward()
            dopt.step()
            dopt.zero_grad()
            dloss_record[-1] = (dloss_record[-1]+loss.item())/2
            
            print(f"\rEpoch {epoch+1}/{epochs}: "+
                  f"[{'*'*int(cntr/n_batch*50)+' '*(50-int(cntr/n_batch*50))}]", 
                  end = '\r')
        print(f"{' '*150}\r", end = '\r')
        print(f" Gen loss: [{str(np.round(np.mean(loss_record),2))}]  "+
              f" Gen Add loss: [{str(np.round(np.mean(aloss_record),2))}]  "+
              f" Disc loss: [{str(np.round(np.mean(dloss_record),2))}]")
      
        js, rs, ps, f1 = eval_model(val_DL, model, return_xy=False, T = T)
        history['loss'].append(loss_record)
        history['jaccard'].append(js)
        history['recall'].append(rs)
        history['precision'].append(ps)
        history['f1'].append(f1)
        
        if js > best_ja:
            torch.save(model.state_dict(), open('SeqMed.model','wb'))
            best_ja = js
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f'Epoch: {epoch+1}, Loss: {str(np.mean(loss_record).round(2))}, '+
              f'Epoch Time: {str(np.round(elapsed_time,2))}, '+
              f'Appro Left Time: {str(np.round(elapsed_time * (epochs - epoch - 1), 2))}')
    pickle.dump(history, open('SeqMed_history.pkl', 'wb'))
    print('Completed!')

    
def eval_model(dataLoader, model, T = 36, return_xy = True):
    pmeds = []
    ymeds = []
    model = model.eval()
    n_batch=len(dataLoader)
    cntr=0
    for p, d, m, mh, n_p, n_d, n_m, n_v in dataLoader:
        cntr+=1
        discrete = model.discrete(p,d)
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





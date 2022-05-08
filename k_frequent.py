#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:45:13 2022

@author: steven
"""
import pickle
import torch
import numpy as np

from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)


print('Training k-frequent................................................\n')
print('Note- this takes a while, and saves off test-predictions')
print('for quick-comparisons later on.')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
test = torch.load('test.pkl')


pro = np.zeros((len(train.dataset), proc_dict.n))
dia = np.zeros((len(train.dataset), diag_dict.n))
y = np.zeros((len(train.dataset), med_dict.n))

cntr = 0
for p_, d_, m_, mh, n_p, n_d, n_m, n_v in train:
    b_size = len(p_)
    for b in range(b_size):
        for i in torch.cat(p_[b]).to('cpu').numpy():
            pro[cntr,i]=1
        for i in torch.cat(d_[b]).to('cpu').numpy():
            dia[cntr,i]=1
        y[cntr] = mh[b].to('cpu').numpy()
        cntr+=1
x_train = np.concatenate([pro,dia],axis=1)
y_train = y.copy()

most_frequent = []
for c in range(x_train.shape[1]):
    most_frequent.append(y[x_train[:,c]>0].sum(0).argsort()[-5:])
    
y_pred = []
y_truth = []

cntr = 0
n_test = len(test.dataset)
for p_, d_, m_, mh, n_p, n_d, n_m, n_v in test:
    b_size = len(p_)
    for b in range(b_size):
        cntr+=1
        pro = np.zeros((proc_dict.n))
        dia = np.zeros((diag_dict.n))
        for i in torch.cat(p_[b]).to('cpu').numpy():
            pro[i]=1
        for i in torch.cat(d_[b]).to('cpu').numpy():
            dia[i]=1
        y_truth.append(mh[b].to('cpu').numpy())
        hsr = np.unique(np.where(np.concatenate([pro,dia],axis=0)))
        pred = np.zeros((med_dict.n))
        for h in hsr:
            pred[most_frequent[h]]=1
        y_pred.append(pred)
        print(f'Completed {cntr} / {n_test}')
print('Complete... Saving and printing scores.')

y_pred = np.stack(y_pred, axis = 0)
y_truth = np.stack(y_truth, axis = 0)
pickle.dump(y_pred, open('k_frequent_test_preds.pkl','wb'))
pickle.dump(y_truth, open('k_frequent_test_truth.pkl','wb'))

print(f'Jaccard:   {str(jaccard_score(y_truth, y_pred, average = "samples").round(4))}')
print(f'Recall:    {str(recall_score(y_truth, y_pred, average = "samples").round(4))}')
print(f'Precision: {str(precision_score(y_truth, y_pred, average = "samples", zero_division = 0).round(4))}')
print(f'F1:        {str(f1_score(y_truth, y_pred, average = "samples").round(4))}')

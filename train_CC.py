#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:45:13 2022

@author: steven
"""
import pickle
import torch


print('Training CC................................................\n')
print('This model is explicitly defined, therefore, ')
print('I fit and save off the test predictions.... be patient...')
proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
test = torch.load('test.pkl')

import numpy as np
import torch
pro = np.zeros((len(train.dataset), proc_dict.n))
dia = np.zeros((len(train.dataset), diag_dict.n))
y = np.zeros((len(train.dataset), med_dict.n))

cntr = 0
for p_, d_, m_, mh, n_p, n_d, n_m, n_v in train:
    b_size = len(p_)
    for b in range(b_size):
        for i in torch.cat(p_[b]).to('cpu').numpy():
            pro[cntr,i]+=1
        for i in torch.cat(d_[b]).to('cpu').numpy():
            dia[cntr,i]+=1
        y[cntr] = mh[b].to('cpu').numpy()
        cntr+=1
x = np.concatenate([pro,dia],axis=1)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 4)
from sklearn.multioutput import ClassifierChain
cc1 = ClassifierChain(dtc, order='random', random_state=0).fit(x,y)
p1 = cc1.predict(x)
cc2 = ClassifierChain(dtc, order='random', random_state=0).fit(x, y-p1)
p2 = cc2.predict(x)
cc3 = ClassifierChain(dtc, order='random', random_state=0).fit(x, y-(p1+p2))


pro = np.zeros((len(test.dataset), proc_dict.n))
dia = np.zeros((len(test.dataset), diag_dict.n))
y_truth = np.zeros((len(test.dataset), med_dict.n))
cntr = 0
for p_, d_, m_, mh, n_p, n_d, n_m, n_v in test:
    b_size = len(p_)
    for b in range(b_size):
        for i in torch.cat(p_[b]).to('cpu').numpy():
            pro[cntr,i]+=1
        for i in torch.cat(d_[b]).to('cpu').numpy():
            dia[cntr,i]+=1
        y_truth[cntr] = mh[b].to('cpu').numpy()
        cntr+=1
x = np.concatenate([pro,dia],axis=1)

p1 = cc1.predict(x)
p2 = cc2.predict(x)
p3 = cc3.predict(x)
y_pred = (p1+p2+p3).clip(0,1)

pickle.dump(y_pred, open('cc_test_preds.pkl','wb'))
pickle.dump(y_truth, open('cc_test_truth.pkl','wb'))

from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)

print(f'Jaccard:   {str(jaccard_score(y_truth, y_pred, average = "samples").round(4))}')
print(f'Recall:    {str(recall_score(y_truth, y_pred, average = "samples").round(4))}')
print(f'Precision: {str(precision_score(y_truth, y_pred, average = "samples", zero_division = 0).round(4))}')
print(f'F1:        {str(f1_score(y_truth, y_pred, average = "samples").round(4))}')


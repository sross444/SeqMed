#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:42:40 2022

@author: steven
"""

from MLP import *
print('Training MLP................................................\n')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
val = torch.load('val.pkl')

patients = pickle.load(open('patients.pkl','rb'))

mlp = MLP(proc_dict.n, diag_dict.n, med_dict.n).to('cuda')

train_MLP(train, val, mlp, epochs = 8, lr = .00001)

test = torch.load('test.pkl')
x, y = eval_model(test, mlp)
pickle.dump(x, open('MLP_test_preds.pkl','wb'))
pickle.dump(y, open('MLP_test_truth.pkl','wb'))
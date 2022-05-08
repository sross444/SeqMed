#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:44:04 2022

@author: steven
"""
from SGM import *

print('Training SGM................................................\n')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
val = torch.load('val.pkl')

patients = pickle.load(open('patients.pkl','rb'))

sgm = SGM(proc_dict.n, diag_dict.n, med_dict.n).to('cuda')

train_SGM(train, val, sgm, epochs = 4, lr = 1e-4, T = 24)

test = torch.load('test.pkl')
x, y= eval_model(test, sgm, T = 12)
pickle.dump(x, open('SGM_test_preds.pkl','wb'))
pickle.dump(y, open('SGM_test_truth.pkl','wb'))
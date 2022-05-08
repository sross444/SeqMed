#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:44:04 2022

@author: steven
"""
from LEAP import *

print('Training LEAP................................................\n')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
val = torch.load('val.pkl')

patients = pickle.load(open('patients.pkl','rb'))
mc_dict = med_count_dict(patients, med_dict)
leap = LEAP(proc_dict.n, diag_dict.n, med_dict.n, mc_dict).to('cuda')

train_leap(train, val, leap, epochs = 4, lr = 1e-4, T = 24)

test = torch.load('test.pkl')
x, y= eval_model(test, leap, T = 12)
pickle.dump(x, open('LEAP_test_preds.pkl','wb'))
pickle.dump(y, open('LEAP_test_truth.pkl','wb'))
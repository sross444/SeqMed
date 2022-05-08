#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:42:40 2022

@author: steven
"""

from COMPNet import *
print('Training COMPNet................................................\n')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
val = torch.load('val.pkl')

patients = pickle.load(open('patients.pkl','rb'))

ehr_adj = get_weighted_ehr_adj(patients, n_meds = med_dict.n,  only_first_visit = True)

compnet = Agent(proc_dict.n, diag_dict.n, med_dict.n, ehr_adj = ehr_adj)

train_compnet(train, val, compnet, epochs = 4, lr = .00001, gamma = 1,
                  target_update_iter = 16, prop_train = .1, teacher_force = .6, epsilon = .1)


test = torch.load('test.pkl')
x, y = eval_model(test, compnet)
pickle.dump(x, open('COMPNet_test_preds.pkl','wb'))
pickle.dump(y, open('COMPNet_test_truth.pkl','wb'))
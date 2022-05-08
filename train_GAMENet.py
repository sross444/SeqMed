#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:44:04 2022

@author: steven
"""
from GAMENet import * 
print('Training GAMENet................................................\n')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
val = torch.load('val.pkl')

patients = pickle.load(open('patients.pkl','rb'))

ehr_adj = get_weighted_ehr_adj(patients, n_meds = med_dict.n,  only_first_visit = True)


gamenet= GAMENet((proc_dict.n, diag_dict.n, med_dict.n), emb_dim = 64, ehr_adj = ehr_adj)

train_gamenet(train, val, gamenet, epochs = 8, lr = .0002)

test = torch.load('test.pkl')
x, y = eval_model(test, gamenet)
pickle.dump(x, open('GAMENet_test_preds.pkl','wb'))
pickle.dump(y, open('GAMENet_test_truth.pkl','wb'))
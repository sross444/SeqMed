#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:49:55 2022

@author: steven
"""
from SeqMed import *
print('Training SeqMed................................................\n')

proc_dict = pickle.load(open('proc_dict.pkl','rb'))
diag_dict = pickle.load(open('diag_dict.pkl','rb'))
med_dict = pickle.load(open('med_dict.pkl','rb'))

train = torch.load('train.pkl')
val = torch.load('val.pkl')

sm = SeqMed(proc_dict.n, diag_dict.n, med_dict.n).to('cuda')
disc = discriminator(med_dict.n).to('cuda')


train_SeqMed(train, val, sm, disc, epochs = 8, lr = .004)

test = torch.load('test.pkl')
x, y= eval_model(test, sm, T = 12)
pickle.dump(x, open('SeqMed_test_preds.pkl','wb'))
pickle.dump(y, open('SeqMed_test_truth.pkl','wb'))

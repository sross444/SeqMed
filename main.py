#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:40:07 2022

@author: steven
"""
import os
os.chdir('/home/steven/Desktop/school_stuff/deepLearning/final_final')

import munge #(done)
import train_k_nearest #(done)
import train_k_frequent #(done)
import train_MLP #(done)
import train_CC #(done)
import train_SGM #(done)
import train_LEAP #(done)
import train_GAMENet #(done)
import train_COMPNet #(done)
import train_SeqMed  #(done)
import train_SeqMed_teacherforce  #(done)



from sklearn.metrics import jaccard_score # (y_true, y_pred)
from sklearn.metrics import precision_score # (y_true, y_pred)
from sklearn.metrics import recall_score # (y_true, y_pred)
from sklearn.metrics import f1_score # (y_true, y_pred)

import pickle

kn_preds = pickle.load(open('k_nearest_test_preds.pkl','rb'))
kn_truths = pickle.load(open('k_nearest_test_truth.pkl','rb'))

kf_preds = pickle.load(open('k_frequent_test_preds.pkl','rb'))
kf_truths = pickle.load(open('k_frequent_test_truth.pkl','rb'))

cc_preds = pickle.load(open('cc_test_preds.pkl','rb'))
cc_truths = pickle.load(open('cc_test_truth.pkl','rb'))

MLP_preds = pickle.load(open('MLP_test_preds.pkl','rb'))
MLP_truths = pickle.load(open('MLP_test_truth.pkl','rb'))

SGM_preds = pickle.load(open('SGM_test_preds.pkl','rb'))
SGM_truths = pickle.load(open('SGM_test_truth.pkl','rb'))

LEAP_preds = pickle.load(open('LEAP_test_preds.pkl','rb'))
LEAP_truths = pickle.load(open('LEAP_test_truth.pkl','rb'))

GAMENet_preds = pickle.load(open('GAMENet_test_preds.pkl','rb'))
GAMENet_truths = pickle.load(open('GAMENet_test_truth.pkl','rb'))

COMPNet_preds = pickle.load(open('COMPNet_test_preds.pkl','rb'))
COMPNet_truths = pickle.load(open('COMPNet_test_truth.pkl','rb'))

SeqMed_preds = pickle.load(open('SeqMed_test_preds.pkl','rb'))
SeqMed_truths = pickle.load(open('SeqMed_test_truth.pkl','rb'))

SeqMed_teacherforce_preds = pickle.load(open('SeqMed_teacherforce_test_preds.pkl','rb'))
SeqMed_teacherforce_truths = pickle.load(open('SeqMed_teacherforce_test_truth.pkl','rb'))

names = ['k-nearest','k-frequent','classifier-chain','MLP','SGM','LEAP','GAMENet','COMPNet','SeqMed','SeqMed_teacherforce']
p_list = [kn_preds, kf_preds, cc_preds, MLP_preds, SGM_preds, LEAP_preds, GAMENet_preds, COMPNet_preds, SeqMed_preds, SeqMed_teacherforce_preds]
t_list = [kn_truths, kf_truths, cc_truths, MLP_truths, SGM_truths, LEAP_truths, GAMENet_truths, COMPNet_truths, SeqMed_truths, SeqMed_teacherforce_truths]

for i in range(len(p_list)):
    print(f'==================={names[i]}===================')
    print(f'Jaccard: [{str(jaccard_score(t_list[i], p_list[i], average = "samples").round(4))}]  '+
          f'Recall: [{str(recall_score(t_list[i], p_list[i], average = "samples").round(4))}]  '+
          f'Precision: [{str(precision_score(t_list[i], p_list[i], average = "samples", zero_division = 0).round(4))}]  '+
          f'F1: [{str(f1_score(t_list[i], p_list[i], average = "samples").round(4))}]')
    




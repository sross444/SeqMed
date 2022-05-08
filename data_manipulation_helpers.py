#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:29:32 2022

@author: 
    FROM:  https://github.com/WOW5678/CompNet
    Citation to:  Order-free Medicine Combination Prediction With Graph
    Convolutional Reinforcement Learning, 2019
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle

def process_procedure(procedure_file):
    pro_pd= pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'}, low_memory=False)
    pro_pd.drop(columns=['ROW_ID'],inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'],inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'],inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True,inplace=True)
    return pro_pd

def process_med(med_file):
    med_pd=pd.read_csv(med_file,dtype={'NDC':'category'}, low_memory=False)
    #filter
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                     'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','FORM_UNIT_DISP',
                      'ROUTE','ENDDATE','DRUG'],axis=1,inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC']=='0'].index,axis=0,inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new

    med_pd=filter_first24hour_med(med_pd)
    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    def process_vist_lg2(med_pd):
        a=med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_vist_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
    return med_pd

def process_diag(diag_file):
    diag_pd=pd.read_csv(diag_file, low_memory=False)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)

def ndc2atc4(med_pd, path):
    with open(path + 'ndc2rxnorm_mapping.txt', 'r') as f:
        ndc2rxnorm=eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(path + 'rxnorm2atc4.csv', low_memory=False)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)

def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
    return diag_pd.reset_index(drop=True)

def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)

def process_all(path = '/home/steven/Desktop/school_stuff/deepLearning/'+\
    'project/mimic-iii-clinical-database-1.4/'):
    med_pd = process_med(path + 'PRESCRIPTIONS.csv')
    med_pd = ndc2atc4(med_pd, path)
    diag_pd = process_diag(path + 'DIAGNOSES_ICD.csv')
    diag_pd = filter_2000_most_diag(diag_pd)
    pro_pd = process_procedure(path + 'PROCEDURES_ICD.csv')

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data


def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape[0])
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag = 0
    avg_med = 0
    avg_pro = 0
    max_diag = 0
    max_med = 0
    max_pro = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        z = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x = set(x)
        y = set(y)
        z = set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
    print('#avg of diagnoses ', np.round(avg_diag / cnt,2))
    print('#avg of medicines ', np.round(avg_med / cnt, 2))
    print('#avg of procedures ', np.round(avg_pro / cnt, 2))
    print('#avg of vists ', np.round(avg_visit / len(data['SUBJECT_ID'].unique()),2))
    '''
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)
    '''

class dictionary():
    def __init__(self, vocab):
        self.token_dict = {vocab[i]: i for i in range(len(vocab))}
        self.word_dict = {i: vocab[i] for i in range(len(vocab))}
        self.n = len(vocab)
    def tok(self, word):
        return self.token_dict[word]
    def word(self, tok):
        return self.word_dict[tok]

def build_dictionaries(data):
    pro_dict = dictionary(
        np.unique(
            [item for sublist in list(data['PRO_CODE']) for item in sublist]))
    diag_dict = dictionary(
        np.unique(
            [item for sublist in list(data['ICD9_CODE']) for item in sublist]))
    med_dict =  dictionary(
        np.unique(
            [item for sublist in list(data['NDC']) for item in sublist]))
    return pro_dict, diag_dict, med_dict

# this creates lists of patient tokens.  result is a list of patients, each 
# a list of visits each with lists of procedures, diagnoses and medications
def process_patients(data, proc_dict, diag_dict, med_dict):
    patients = []
    data.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    for name, group in data.groupby(['SUBJECT_ID']):
        visits = []
        for r in range(group.shape[0]):
            pro = [proc_dict.tok(j) for j in group.iloc[r]['PRO_CODE']]
            dia = [diag_dict.tok(j) for j in group.iloc[r]['ICD9_CODE']]
            med = [med_dict.tok(j) for j in group.iloc[r]['NDC']]
            visits.append([pro, dia, med])
        patients.append(visits)
    return patients


def build_dataloaders(patients, n_meds, min_visits = 2,
                        train_test_val = [2/3,1/6,1/6], only_last_visit = False,
                        batch_size = 32, seed = 0):
    lst = []
    for p in patients:
        if len(p)<min_visits: continue
        if not only_last_visit:
            for i in range(1,len(p)):
                lst.append(p[:(i+1)])
        else:
            lst.append(p)
    np.random.seed(seed)
    splits = np.random.choice(
        range(3), size = len(lst), replace = True,
        p= train_test_val)
    train, test, val = [], [], []
    for i, s in enumerate(splits):
        if s == 0:
            train.append(lst[i])
        elif s == 1:
            test.append(lst[i])
        else:
            val.append(lst[i])
    return DL(train, n_meds, batch_size), \
        DL(test, n_meds, batch_size, shuffle = False), \
            DL(val, n_meds, batch_size, shuffle = False)
        
class DS(Dataset):
    def __init__(self, x, n_meds):
        # all tensors as longs, sent to cuda (counts aren't necessary on gpu)
        self.p = [[v[0] for v in p] for p in x]
        self.d = [[v[1] for v in p] for p in x]
        self.m = [[v[2] for v in p] for p in x]
        # create multi-hot target vectors
        self.multi_hot_target = torch.zeros((len(x), n_meds)).to('cuda')
        for i, t in enumerate(self.m):
            self.multi_hot_target[i,t[-1]]=1
            
        # count number of vists, and max # proc, diags, meds per patient
        # return counts for padding, looping, ect...
        self.n_v, self.n_d, self.n_p, self.n_m = [[] for _ in range(4)]
        for i in range(len(self.p)):
            self.n_v.append(len(self.p[i]))
            np, nd, nm = [[] for g in range(3)]
            for v in range(len(self.p[i])):
                self.p[i][v] = torch.Tensor(self.p[i][v]).type(torch.long).to('cuda')
                self.d[i][v] = torch.Tensor(self.d[i][v]).type(torch.long).to('cuda')
                self.m[i][v] = torch.Tensor(self.m[i][v]).type(torch.long).to('cuda')
                np.append(len(self.p[i][v]))
                nd.append(len(self.d[i][v]))
                nm.append(len(self.m[i][v]))
            self.n_p.append(max(np))
            self.n_d.append(max(nd))
            self.n_m.append(max(nm))
            
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.p[i], self.d[i], self.m[i], self.multi_hot_target[i], \
            self.n_p[i], self.n_d[i], self.n_m[i], self.n_v[i]
    
def my_collate_fn(data):
    p, d, m, mh, np, nd, nm, nv = zip(*data)
    return p, d, m, torch.stack(mh), np, nd, nm, nv

def DL(x, n_meds, batch_size=64, shuffle=True):
    dataSet = DS(x, n_meds)
    return DataLoader(dataSet, batch_size=batch_size,
                      collate_fn=my_collate_fn, shuffle=shuffle)

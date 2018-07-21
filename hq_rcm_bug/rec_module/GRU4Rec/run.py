#-*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
#sys.path.append('../')

import numpy as np
import pandas as pd
import gru4rec
import evaluation
import baselines
import matplotlib.pyplot as plt

PATH_TO_TRAIN = '../report_labels_train_openoffice'
PATH_TO_TEST = '../report_labels_test_openoffice'

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t')
    valid = pd.read_csv(PATH_TO_TEST, sep='\t')
    
    #Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    
    #print('Training GRU4Rec with 100 hidden units')    
    top=np.array([1,5,10,15,20])
    #top=11
    gru = gru4rec.GRU4Rec(n_epochs=10,loss='top1', final_act='tanh', hidden_act='relu', layers=[512], batch_size=32, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    gru.fit(data)
    #cf=baselines.ItemKNN()
    #cf.fit(data)
    #mf=baselines.BPR()
    #mf.fit(data)
    #acc_gru=[]
    #acc_cf=[]
    #acc_mf=[]
    mrr_gru=[]
    #mrr_cf=[]
    #mrr_mf=[]
    for i in top:
        res_gru = evaluation.evaluate_sessions_batch(gru, valid, None,cut_off=i+1)
        #res_cf = evaluation.evaluate_sessions(cf, valid, data, None,cut_off=i+1)
     #   res_mf = evaluation.evaluate_sessions(mf, valid, data, None,cut_off=i+1)
        mrr_gru.append(res_gru[1])
    print mrr_gru
    #    mrr_gru.append(res_gru[1])
    #    acc_cf.append(res_cf[0])
    #    mrr_cf.append(res_cf[1])
    #    acc_mf.append(res_mf[0])
    #    mrr_mf.append(res_mf[1])
    #bar_width=0.6
    #X=top
    #Y_gru=acc_gru
    #Y_cf=acc_cf
    #Y_mf=acc_mf
    #print Y_gru
    #print Y_cf
    #print Y_mf
    #plt.bar(X,Y_gru,width=0.6,color='red',label='GRU')
    #plt.bar(X+bar_width,Y_cf,width=0.6,color='blue',label='CNN')
    #plt.bar(X+bar_width*2,Y_mf,width=0.6,color='green',label='Matrix Factorization')
    #plt.xlabel('TOP k')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.show()
    #print('Accuracy@{}: {}'.format(top-1,res_gru[0]))
    #print('MRR@{}: {}'.format(top-1,res_gru[1]))
    #print('Accuracy@{}: {}'.format(top,res_mf[0]))
    #print('MRR@{}: {}'.format(top,res_mf[1]))
    
    
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    
    #print('Training GRU4Rec with 100 hidden units')

    #gru = gru4rec.GRU4Rec(loss='bpr-max-0.5', final_act='linear', hidden_act='tanh', layers=[100], batch_size=32, dropout_p_hidden=0.0, learning_rate=0.2, momentum=0.5, n_sample=2048, sample_alpha=0, time_sort=True)
    #gru.fit(data)
    #
    #res = evaluation.evaluate_sessions_batch(gru, valid, None)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))

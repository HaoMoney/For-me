#*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015
@author: Balázs Hidasi
"""
from sklearn.cluster import DBSCAN,KMeans,AffinityPropagation,MeanShift,AgglomerativeClustering
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
import string
from nltk.stem.porter import PorterStemmer
import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.pipeline import Pipeline
import  matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import calinski_harabaz_score,silhouette_score 
#print af.labels_
clusters=range(100,1100,100)
score_km=[449.7212261229711, 1653.418115918799, 54912.306600452685, 46263.64103389814, 40192.33333667678, 35740.117281353545, 32794.66722238412, 30419.22552323544, 28729.801150292435, 27135.904235644717]
X=clusters
#Y1=score_ac
Y2=score_km
print Y2
#plt.plot(X,Y1,color='blue',label='AgglomerativeClustering')
plt.plot(X,Y2,'x--',label='k-means')
plt.xlabel('The value of k')
plt.ylabel('calinski_harabaz_score')
plt.legend()
plt.show()
#print ac.labels_
#labels=ms.labels_.reshape(len(km.labels_),1)
#print X_np_train.shape
#print labels.shape
#X_labels_train=np.concatenate((X_np_train,labels),axis=1)
#a=X_labels_train[0]
#print len(X_labels_train[0])
#print type(X_labels_train)
#print X_labels_train.shape
#f=open('labels_train','w')
#f.write(a)
#report_dict={}
#for report in X_labels_train:
#    report_dict[report[2]]=[]
#for report in X_labels_train:
#    if report_dict.has_key(report[2]):
#        t=time.mktime(time.strptime(report[4],'%Y-%m-%d %H:%M:%S'))
#        report_dict[report[2]].append(tuple((str(t),str(report[5]))))
#for key in report_dict:
#    report_dict[key].sort()
#with open('labels_train','w') as f:
#    for key in report_dict:
#        for item in report_dict[key]:
#            f.write(key+'\t'+item[1]+'\t'+item[0])
#            f.write('\n')
#ass_dict=[]
#for report in X_labels_train:
#    ass_dict.append(report[2]+'\t'+report[4]+'\t'+str(report[5]))
#for each in ass_dict:
#    each_arr=each.split('\t')
    
    
#print X_labels_train
#X_train_data=pd.merge(X_train_data,pd.Series(km.labels_))
#print X_labels_train
#X_test=np.concatenate((X_dcr_test,X_tfidf_test),axis=1)
#    svc=LinearSVC(multi_class='crammer_singer')
#    svc_begin_time=time.time()
#    svc.fit(X_train,y_train)
#    svc_end_time=time.time()
#    print "SVM training time:%fs" % (svc_end_time-svc_begin_time)
#    svc_y_pred=svc.predict(X_test)
#    svc_acc=accuracy_score(y_test,svc_y_pred)
#    print "accuracy:%f" % svc_acc
#    plex.append(svc_acc)
#X=topics
#C=plex
#print plex
#plt.xlim(min(X) * 1.1, max(X) * 1.1)
#plt.ylim(min(C) * 1.1, max(C) * 1.1)
#plt.plot(X,C)
#plt.xlabel('topics')
#plt.ylabel('acc')
#plt.show()
#X_tmp=lda.fit_transform(X_tfidf_train)
#tfidf_feature_names=tfidf_vec.get_feature_names()
#print tfidf_feature_names
#print_top_words(lda,tfidf_feature_names,20)
#print X_tmp
#print X_tmp.shape
#print lda.perplexity(X_tfidf_train)
#X_tfidf_train=list(X_tfidf_train)
#X_tfidf_test=tfidf_vec.transform(X_tfidf_test)

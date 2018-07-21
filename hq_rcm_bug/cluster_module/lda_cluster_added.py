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
#def textPrecessing(text):
#    #小写化
#    text = text.lower()
#    #去除特殊标点
#    for c in string.punctuation:
#        text = text.replace(c, ' ')
#    text=re.sub('[^a-z]',' ',text)
#    #分词
#    wordLst = nltk.word_tokenize(text)
#    #去除停用词
#    filtered = [w for w in wordLst if w not in stopwords.words('english')]
#    #仅保留名词或特定POS   
#    #refiltered =nltk.pos_tag(filtered)
#    #filtered = [w for w, pos in refiltered if pos.startswith('NN')]
#    #词干化
#    ps = PorterStemmer()
#    filtered = [ps.stem(w) for w in filtered]
#    return " ".join(filtered)
def clear_title(title,remove_stopwords):
    raw_text=BeautifulSoup(title,'html').get_text()
    letters=re.sub('[^a-zA-Z]',' ',raw_text)
    words=letters.lower().split()
    if remove_stopwords:
	stop_words=set(stopwords.words('english'))
	words=[w for w in words if w not in stop_words]	
    return ' '.join(words)
dict_vec=DictVectorizer(sparse=False)
PATH_TO_ORIGINAL_DATA = '../datasets/'
#PATH_TO_PROCESSED_DATA = '/path/to/store/processed/data/'
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'fixed.csv',sep='\t')
f_data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'data.csv',sep='\t')
selected_columns=['Product','Component','Assignee','Summary','Changed']
data=data[selected_columns]
#print len(data['Product'].unique())
#print len(data['Component'].unique())
classes=data['Assignee'].unique()
n_classes=len(classes)
print n_classes
#print n_classes
classmap=pd.Series(data=np.arange(n_classes),index=classes)
#print classmap
data=pd.merge(data,pd.DataFrame({'Assignee':classes,'ClassId':classmap[classes].values}),on='Assignee',how='inner')
X_data=data[['Product','Component','Assignee','Summary','Changed']]
y_data=data['ClassId']
ass_dict={}
for ass in f_data['Assignee']:
    if not ass_dict.has_key(ass):
        ass_dict[ass]=[0,0,0]
for i in range(len(f_data)):
    ass=f_data['Assignee'][i]
    create=time.mktime(time.strptime(f_data['Create'][i],'%Y-%m-%d %H:%M:%S'))
    changed=time.mktime(time.strptime(f_data['Changed'][i],'%Y-%m-%d %H:%M:%S'))
    t=changed-create
    ass_dict[ass][0]+=1
    ass_dict[ass][1]+=t
    if f_data['Resolution'][i]=='FIXED':
        ass_dict[ass][2]+=1
total=float(len(f_data))
#print y_data
#print data.info()
X_np_train,X_np_test,y_np_train,y_np_test=cross_validation.train_test_split(X_data,y_data,test_size=0.25)
#print len(y_train)
#print len(y_test)
#print y_train
X_train=pd.DataFrame(X_np_train)
X_test=pd.DataFrame(X_np_test)
y_train=pd.Series(y_np_train)
y_test=pd.Series(y_np_test)
#X_train_data=pd.merge(X_train,y_train)
#X_test_data=pd.merge(X_test,y_test)
X_train.columns=['Product','Component','Assignee','Summary','Changed']
X_dcr_train=X_train[['Product','Component']]
X_test.columns=['Product','Component','Assignee','Summary','Changed']
X_dcr_test=X_test[['Product','Component']]
X_dcr_train=dict_vec.fit_transform(X_dcr_train.to_dict(orient='records'))
X_dcr_test=dict_vec.transform(X_dcr_test.to_dict(orient='records'))
#X_train.groupby('Assignee')
#print len(X_test)
#X_new_test=X_test[np.in1d(X_train.Assignee,X_test.Assignee)]
#print len(X_new_test)
X_tfidf_train=[]
for title in X_train['Summary']:
    X_tfidf_train.append(clear_title(title,True))
#print X_tfidf_train
X_tfidf_test=[]
for title in X_test['Summary']:
    X_tfidf_test.append(clear_title(title,True))
tfidf_vec=TfidfVectorizer(analyzer='word')
X_tfidf_train=tfidf_vec.fit_transform(X_tfidf_train)
X_tfidf_test=tfidf_vec.transform(X_tfidf_test)
#pip=Pipeline([('tfidf_vec',TfidfVectorizer(analyzer='word')),('lda',LatentDirichletAllocation(learning_method='batch'))])
#params={'tfidf_vec_binary':[True,False],'tfidf_vec_ngram_range':[(1,1),(1,2)],'lda_n_components':[30,50,100,200]}
#gs=GridSearchCV(pip,params,cv=4,n_jobs=-1,verbose=1)
#gs.fit(X_tfidf_train)
#print gs.best_score_
#print gs.best_params_
#print X_tfidf_train
#parameters = {'learning_method':('batch', 'online'), 
#              'n_topics':(30,50,100,200),
#              'perp_tol': (0.001, 0.01, 0.1),
#              'doc_topic_prior':(0.001, 0.01, 0.05, 0.1, 0.2,0.5),
#              'topic_word_prior':(0.001, 0.01, 0.05, 0.1, 0.2,0.5),
#              }
#lda = LatentDirichletAllocation(max_iter=1000)
#model = GridSearchCV(lda, parameters,verbose=1)
#model.fit(X_tfidf_train)
#print model.best_score_
#print model.best_params_
def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
    #打印主题-词语分布矩阵
    #print model.components_.shape
    #print model.components_
#doc_topic_prior=[0.001, 0.01, 0.05, 0.1, 0.2,0.5]
#topic_word_prior=[0.001, 0.01, 0.05, 0.1, 0.2,0.5]
#topics=[50,100,500,1000]
#iters=[50,100,500,1000]
#plex=[]
lda=LatentDirichletAllocation(n_components=180,max_iter=100,learning_method='batch',doc_topic_prior=0.5,topic_word_prior=0.2)
lda_begin_time=time.time()
lda.fit(X_tfidf_train)
lda_end_time=time.time()
print "LDA training time:%fs" % (lda_end_time-lda_begin_time)
X_tfidf_train=lda.transform(X_tfidf_train)
X_tfidf_test=lda.transform(X_tfidf_test)
X_train=np.concatenate((X_dcr_train,X_tfidf_train),axis=1)
X_test=np.concatenate((X_dcr_test,X_tfidf_test),axis=1)
#ac=AgglomerativeClustering(n_clusters=500)
#ac_begin_time=time.time()
#ac.fit(X_train)
#ac_end_time=time.time()
#print "AgglomerativeClustering training time:%fs" % (ac_end_time-ac_begin_time)
#test_labels=ac.fit_predict(X_test)
km=KMeans(n_clusters=500)
km_begin_time=time.time()
km.fit(X_train)
km_end_time=time.time()
print "KMeans training time:%fs" % (km_end_time-km_begin_time)
train_labels=km.labels_.reshape(len(km.labels_),1)
tmp_labels=km.predict(X_test)
test_labels=tmp_labels.reshape(len(tmp_labels),1)
X_labels_train=np.concatenate((X_np_train,train_labels),axis=1)
X_labels_test=np.concatenate((X_np_test,test_labels),axis=1)
#print X_np_train.shape
#print labels.shape
#a=X_labels_train[0]
#print len(X_labels_train[0])
#print type(X_labels_train)
#print X_labels_train.shape
#f=open('labels_train','w')
#f.write(a)
train_assignee_dict={}
train_report_dict={}
for report in X_labels_train:
    train_report_dict[report[5]]=[]
    train_assignee_dict[report[2]]=1
for report in X_labels_train:
    if train_report_dict.has_key(report[5]):
        t=time.mktime(time.strptime(report[4],'%Y-%m-%d %H:%M:%S'))
        train_report_dict[report[5]].append(tuple((str(t),report[2])))
for key in train_report_dict:
    train_report_dict[key].sort()
with open('../rec_module/report_labels_train','w') as f:
    f.write('SessionId\tItemId\tActive\tSkill\tTime\n')
    for key in train_report_dict:
        for item in train_report_dict[key]:
            active=ass_dict[item[1]][0]
            tmp=1/(ass_dict[item[1]][1]/total)
            repair_time=(tmp-1.8033736213e-07)/(5.364150943399999-1.8033736213e-07)
            vict=(active-1)/629.0
            skill=(repair_time*0.8+vict*0.2)/2  
            f.write(str(key)+'\t'+item[1]+'\t'+str(active/total)+'\t'+str(skill)+'\t'+item[0])
            f.write('\n')
test_report_dict={}
for report in X_labels_test:
    test_report_dict[report[5]]=[]
for report in X_labels_test:
    if test_report_dict.has_key(report[5]):
        t=time.mktime(time.strptime(report[4],'%Y-%m-%d %H:%M:%S'))
        test_report_dict[report[5]].append(tuple((str(t),report[2])))
for key in test_report_dict:
    test_report_dict[key].sort()
with open('../rec_module/report_labels_test','w') as f:
    f.write('SessionId\tItemId\tActive\tSkill\tTime\n')
    for key in test_report_dict:
        for item in test_report_dict[key]:
            if train_assignee_dict.has_key(item[1]):
                active=ass_dict[item[1]][0]
                tmp=1/(ass_dict[item[1]][1]/total)
                repair_time=(tmp-1.8033736213e-07)/(5.364150943399999-1.8033736213e-07)
                vict=(active-1)/629.0
                skill=(repair_time*0.8+vict*0.2)/2  
                f.write(str(key)+'\t'+item[1]+'\t'+str(active/total)+'\t'+str(skill)+'\t'+item[0])
                f.write('\n')
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

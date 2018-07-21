#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np 
top = np.array([5,10,15,20])
bar_width=1.5
mrr_gru=np.array([0.49,0.51,0.62,0.644])
mrr_cnn=np.array([0.498,0.468,0.624,0.68])
#print np.append(top,2)
#print top
#print av_mrr
#mrr_gru=np.concatenate([mrr_gru,[av_mrr[2]]])
#mrr_devrec=np.concatenate([mrr_devrec,[av_mrr[1]]])
#mrr_cnn=np.concatenate([mrr_cnn,[av_mrr[0]]])
X=top
Y_gru=mrr_gru
Y_cnn=mrr_cnn
#Y_av=av_mrr
#Y_mf=mrr_mf
plt.bar(X-bar_width/2,Y_cnn,width=1.5,color='red',label='method 2-5')
plt.bar(X+bar_width/2,Y_gru,width=1.5,color='blue',label='method 2-6')
#plt.bar(X+bar_width*3,Y_av,width=1.5,color='black',label='Average')
#plt.title('Eclipse')
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.xlabel('Model')
plt.ylabel('Accuracy')
new_ticks=['KNN','RF','SVM','GBDT']
plt.xticks(top,new_ticks)
plt.ylim(0.0,0.7)
plt.legend()
plt.show()

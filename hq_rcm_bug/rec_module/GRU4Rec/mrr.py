import matplotlib.pyplot as plt
import numpy as np 
top = np.array([5,10,15,20])
bar_width=1.5
mrr_gru=np.array([0.5061760563188318,0.59937227196594033,0.6657428256634608])
mrr_devrec=np.array([0.51709519790916004,0.59800140104554094,0.661810126341555])
mrr_cnn=np.array([0.43980582524271846,0.5099949264332826,0.5354497354497354])
#print np.append(top,2)
#print top
mrr_gru=np.append(mrr_gru,np.mean(mrr_gru))
mrr_devrec=np.append(mrr_devrec,np.mean(mrr_devrec))
mrr_cnn=np.append(mrr_cnn,np.mean(mrr_cnn))
#print av_mrr
#mrr_gru=np.concatenate([mrr_gru,[av_mrr[2]]])
#mrr_devrec=np.concatenate([mrr_devrec,[av_mrr[1]]])
#mrr_cnn=np.concatenate([mrr_cnn,[av_mrr[0]]])
print mrr_gru
print mrr_devrec
print mrr_cnn
X=top
Y_gru=mrr_gru
Y_devrec=mrr_devrec
Y_cnn=mrr_cnn
#Y_av=av_mrr
#Y_mf=mrr_mf
plt.bar(X-bar_width,Y_cnn,width=1.5,color='red',label='CNN')
plt.bar(X,Y_devrec,width=1.5,color='blue',label='DevRec')
plt.bar(X+bar_width,Y_gru,width=1.5,color='green',label='Cluster+GRU')
#plt.bar(X+bar_width*3,Y_av,width=1.5,color='black',label='Average')
#plt.title('Eclipse')
plt.xlabel('Dataset')
plt.ylabel('MRR')
new_ticks=['Mozilla','Openoffice','Eclipse','Average']
plt.xticks(top,new_ticks)
plt.ylim(0.0,0.7)
plt.legend()
plt.show()

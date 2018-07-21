import matplotlib.pyplot as plt
import numpy as np 
top = np.array([0,5,10,15,20])
bar_width=1.5
acc_gru=[0.38124454148471616, 0.5238427947598253, 0.59735807860262, 0.6462882096069869, 0.6793886462882096]
acc_devrec=[0.3931454148471616, 0.52349927947598253, 0.5705807860262, 0.601882096069869, 0.6313786462882096]
acc_cnn=[0.29124454148471616, 0.4338427947598253, 0.4807860262, 0.5292096069869, 0.5473886462882096]
X=top
Y_gru=acc_gru
Y_devrec=acc_devrec
Y_cnn=acc_cnn
#Y_mf=mrr_mf
plt.bar(X,Y_cnn,width=1.5,color='red',label='CNN')
plt.bar(X+bar_width,Y_devrec,width=1.5,color='blue',label='DevRec')
plt.bar(X+2*bar_width,Y_gru,width=1.5,color='green',label='Cluster+GRU')
plt.title('Mozilla')
plt.xlabel('Top-k')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.0,0.7)
plt.legend()
plt.show()

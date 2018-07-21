import matplotlib.pyplot as plt
import numpy as np 
top = np.array([1,5,10,15,20])
bar_width=1.5
acc_gru=[0.38124454148471616, 0.5238427947598253, 0.59735807860262, 0.6462882096069869, 0.6793886462882096]
acc_devrec=[0.3931454148471616, 0.52349927947598253, 0.5705807860262, 0.601882096069869, 0.6313786462882096]
acc_cnn=[0.29124454148471616, 0.4338427947598253, 0.4807860262, 0.5292096069869, 0.5473886462882096]
X=top
Y_gru=acc_gru
Y_devrec=acc_devrec
Y_cnn=acc_cnn
#Y_mf=mrr_mf
plt.plot(X,Y_cnn,'x-',color='red',label='CNN')
plt.plot(X,Y_devrec,'^-',color='blue',label='DevRec')
plt.plot(X,Y_gru,'o-',color='green',label='Cluster+GRU')
plt.title('Mozilla')
plt.xlabel('k')
plt.ylabel('Top-k accuracy')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.25,0.7)
plt.legend()
plt.show()

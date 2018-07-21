import matplotlib.pyplot as plt
import numpy as np 
top = np.array([1,5,10,15,20])
bar_width=1.5
acc_gru=[0.41124454148471616, 0.6738427947598253, 0.75735807860262, 0.7862882096069869, 0.8093886462882096]
acc_devrec=[0.4131454148471616, 0.6799927947598253, 0.7305807860262, 0.7561882096069869, 0.7743786462882096]
acc_cnn=[0.34124454148471616, 0.4738427947598253, 0.57807860262, 0.6392096069869, 0.6673886462882096]
X=top
Y_gru=acc_gru
Y_devrec=acc_devrec
Y_cnn=acc_cnn
#Y_mf=mrr_mf
plt.plot(X,Y_cnn,'x-',color='red',label='CNN')
plt.plot(X,Y_devrec,'^-',color='blue',label='DevRec')
plt.plot(X,Y_gru,'o-',color='green',label='Cluster+GRU')
plt.title('Open Office')
plt.xlabel('k')
plt.ylabel('Top-k accuracy')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.legend()
plt.show()

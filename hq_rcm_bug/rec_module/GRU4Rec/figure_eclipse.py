import matplotlib.pyplot as plt
import numpy as np 
top = np.array([0,5,10,15,20])
bar_width=1.5
acc_gru=[0.46124454148471616, 0.7838427947598253, 0.86735807860262, 0.8962882096069869, 0.9093886462882096]
acc_devrec=[0.4634148471616, 0.7858927947598253, 0.82545807860262, 0.84161882096069869, 0.8592786462882096]
acc_cnn=[0.37124454148471616, 0.5638427947598253, 0.68807860262, 0.72292096069869, 0.763886462882096]
X=top
Y_gru=acc_gru
Y_devrec=acc_devrec
Y_cnn=acc_cnn
#Y_mf=mrr_mf
plt.bar(X,Y_cnn,width=1.5,color='red',label='CNN')
plt.bar(X+bar_width,Y_devrec,width=1.5,color='blue',label='DevRec')
plt.bar(X+bar_width*2,Y_gru,width=1.5,color='green',label='Cluster+GRU')
plt.title('Eclipse')
plt.xlabel('Top-k')
plt.ylabel('acc')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.0,1.0)
plt.legend()
plt.show()

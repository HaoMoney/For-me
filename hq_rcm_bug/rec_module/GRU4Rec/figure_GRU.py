import matplotlib.pyplot as plt
import numpy as np 
top = np.array([1,5,10,15,20])
bar_width=1.5
acc_gru=[0.46124454148471616, 0.7838427947598253, 0.86735807860262, 0.8962882096069869, 0.9093886462882096]
acc_cnn=[0.37124454148471616, 0.5638427947598253, 0.68807860262, 0.72292096069869, 0.763886462882096]
X=top
Y_gru=acc_gru
Y_cnn=acc_cnn
plt.plot(X,Y_cnn,'x-',color='red',label='GRU')
plt.plot(X,Y_gru,'o-',color='blue',label='Cluster+GRU')
plt.xlabel('k')
plt.ylabel('Top-k accuracy')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.3,1.0)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np 
top = np.array([1,5,10,15,20])
bar_width=1.5
ec_acc_gru=np.array([0.46124454148471616, 0.7838427947598253, 0.86735807860262, 0.8962882096069869, 0.9093886462882096])
ec_acc_devrec=np.array([0.4634148471616, 0.7858927947598253, 0.82545807860262, 0.84161882096069869, 0.8592786462882096])
ec_acc_cnn=np.array([0.37124454148471616, 0.5638427947598253, 0.68807860262, 0.72292096069869, 0.763886462882096])
mo_acc_gru=np.array([0.38124454148471616, 0.5238427947598253, 0.59735807860262, 0.6462882096069869, 0.6793886462882096])
mo_acc_devrec=np.array([0.3931454148471616, 0.52349927947598253, 0.5705807860262, 0.601882096069869, 0.6313786462882096])
mo_acc_cnn=np.array([0.29124454148471616, 0.4338427947598253, 0.4807860262, 0.5292096069869, 0.5473886462882096])
op_acc_gru=np.array([0.41124454148471616, 0.6738427947598253, 0.75735807860262, 0.7862882096069869, 0.8093886462882096])
op_acc_devrec=np.array([0.4131454148471616, 0.6799927947598253, 0.7305807860262, 0.7561882096069869, 0.7743786462882096])
op_acc_cnn=np.array([0.34124454148471616, 0.4738427947598253, 0.57807860262, 0.6392096069869, 0.6673886462882096])
av_acc_gru=(ec_acc_gru+mo_acc_gru+op_acc_gru)/3
av_acc_devrec=(ec_acc_devrec+mo_acc_devrec+op_acc_devrec)/3
av_acc_cnn=(ec_acc_cnn+mo_acc_cnn+op_acc_cnn)/3
X=top
Y_gru=av_acc_gru
Y_devrec=av_acc_devrec
Y_cnn=av_acc_cnn
#Y_mf=mrr_mf
plt.plot(X,Y_cnn,'x-',color='red',label='CNN')
plt.plot(X,Y_devrec,'^-',color='blue',label='DevRec')
plt.plot(X,Y_gru,'o-',color='green',label='Cluster+GRU')
plt.title('Average')
plt.xlabel('k')
plt.ylabel('Top-k accuracy')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.25,0.85)
plt.legend()
plt.show()

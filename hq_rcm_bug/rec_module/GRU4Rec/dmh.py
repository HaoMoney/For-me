import matplotlib.pyplot as plt
import numpy as np 
top = np.array([5,10,15])
bar_width=2
acc_nonskill=[0.42, 0.61, 0.69]
#acc_devrec=[0.4154148471616, 0.7258927947598253, 0.86545807860262, 0.896161882096069869, 0.9092786462882096]
acc_skill=[0.45, 0.64, 0.74]
X=top
Y_non=acc_nonskill
#Y_devrec=acc_devrec
Y_a=acc_skill
#Y_mf=mrr_mf
plt.bar(X-1,Y_non,width=2,color='red',label='Pre_Approach')
plt.bar(X+1,Y_a,width=2,color='blue',label='My_Approach')
#plt.bar(X+2*bar_width,Y_cnn,width=1,color='green',label='CNN')
#plt.title('Eclipse')
plt.xlabel('Top-k')
plt.ylabel('acc')
plt.legend()
new_ticks=['1','5','10']
plt.xticks(top,new_ticks)
plt.ylim(0.0,1.0)
plt.show()

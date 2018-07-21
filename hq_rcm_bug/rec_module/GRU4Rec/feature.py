import matplotlib.pyplot as plt
import numpy as np 
top = np.array([5,10,15,20,25])
bar_width=2
acc_nonskill=[0.4568777292576419, 0.7718340611353712, 0.8482532751091703, 0.87117903930131, 0.8820960698689956]
#acc_devrec=[0.4154148471616, 0.7258927947598253, 0.86545807860262, 0.896161882096069869, 0.9092786462882096]
acc_skill=[0.5354497354497354, 0.8507936507936508, 0.8984126984126984, 0.9121693121693122, 0.9195767195767196]
X=top
Y_non=acc_nonskill
#Y_devrec=acc_devrec
Y_a=acc_skill
#Y_mf=mrr_mf
plt.plot(X,Y_non,'x-',color='red',label='normal')
plt.plot(X,Y_a,'o-',color='blue',label='feature')
#plt.bar(X+2*bar_width,Y_cnn,width=1,color='green',label='CNN')
#plt.title('Eclipse')
plt.xlabel('k')
plt.ylabel('Top-k accuracy')
plt.legend()
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.4,1.0)
plt.show()

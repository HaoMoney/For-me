import matplotlib.pyplot as plt
import numpy as np 
top = np.array([5,10,15,20,25])
bar_width=2
acc_nonskill=[0.4568777292576419, 0.7718340611353712, 0.8482532751091703, 0.87117903930131, 0.8820960698689956]
#acc_devrec=[0.4154148471616, 0.7258927947598253, 0.86545807860262, 0.896161882096069869, 0.9092786462882096]
acc_skill=[0.4471232876712329, 0.7808219178082192, 0.8701369863013698, 0.8904109589041096, 0.9068493150684932]
X=top
Y_non=acc_nonskill
#Y_devrec=acc_devrec
Y_a=acc_skill
#Y_mf=mrr_mf
plt.plot(X,Y_non,'x-',color='red',label='normal')
plt.plot(X,Y_a,'o-',color='blue',label='skill_added')
#plt.bar(X+2*bar_width,Y_cnn,width=1,color='green',label='CNN')
#plt.title('Eclipse')
plt.xlabel('k')
plt.ylabel('Top-k accuracy')
new_ticks=['1','5','10','15','20']
plt.xticks(top,new_ticks)
plt.ylim(0.4,1.0)
plt.legend()
plt.show()

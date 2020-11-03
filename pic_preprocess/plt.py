import  matplotlib.pyplot as plt
X=[1,2,3,4,5]
#knn=[0.652222222222,0.712222222222,0.872222222222,0.891666666667,0.97]
svm=[0.866666666667,0.791666666667,0.788888888889,0.766666666667,0.566666666667]
#cnn=[0.97,0.97,0.99,0.99,0.99]
#plt.plot(X,knn,'o-',color='blue',label='kNN')
plt.plot(X,svm,'o-',label='SVM')
#plt.plot(X,cnn,'o-',color='red',label='CNN')
plt.legend()
plt.xlabel('The value of C')
plt.ylabel('Accuracy')
plt.show()


## Import Libraries


```python
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
```

## Learn Module


```python
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    Xp=np.concatenate((y, X), axis=1)
    means = np.zeros(shape=(Xp.shape[1], np.unique(y).size))
    covmat = np.zeros(shape=(X.shape[1], X.shape[1]))
    for i in range(np.unique(y).size):
        means[:,i]=Xp[Xp[:,0] == np.unique(y)[i],:].mean(0)
    covmat=np.cov(X.T)

    return means,covmat
```

## Test Module


```python
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    ypred = np.zeros(shape=(ytest.shape[0],1))
    for i in range(Xtest[:,0].size):
        newmat = np.dot(np.dot((np.tile(Xtest[i],(means[1:,].shape[1],1)).T-means[1:,]).T,inv(covmat)),
                        (np.tile(Xtest[i],(means[1:,].shape[1],1)).T-means[1:,]))
        ypred[i] = means[0,np.where(newmat.diagonal() == newmat.diagonal().min())]
    acc = np.sum(ypred == ytest)/ytest.shape[0]

    return acc,ypred
```

## Execute and Test



```python
#Check for Python 2 or Python 3
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
```


```python
# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
```

    LDA Accuracy = 0.97
    


```python
# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[8,8])
# plt.subplot(1, 1, 1)  #If you want to split up the plot space and create more plots or add plots

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.show()
```


![png](output_9_0.png)



```python

```

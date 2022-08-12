'''   sigma = sigmoid(np.dot(x, weight) + bias)
        loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
        dW = 1/size * np.dot(x.T, (sigma - y))
        db = 1/size * np.sum(sigma - y)
        weight -= learning_rate * dW
        bias -= learning_rate * db '''

from mnist import MNIST
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# load dataset
# download files from http://yann.lecun.com/exdb/mnist/
mndata = MNIST('./')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0
labels_train = labels_train.astype(float)
labels_test = labels_test.astype(float)
# for breaking apart the data, see the following
# boolean indexing:
labels_train == 7
df = pd.DataFrame(X_train,labels_train)
df_test = pd.DataFrame(X_test,labels_test)
mask = df.index == 2.0
df2 = df[mask]
mask = df.index == 7.0
df7 = df[mask]
mask = df_test.index == 2.0
df22 = df_test[mask]
mask = df_test.index == 7.0
df72 = df_test[mask]

frames = [df2, df7]
frames2 = [df22,df72]

result2 = pd.concat(frames2)
result = pd.concat(frames)

result = result.sample(frac=1)
result2 = result2.sample(frac=1)
df_x = result.reset_index(drop = True)
df_y = result.index.to_series()

df_x2 = result2.reset_index(drop = True)
df_y2 = result2.index.to_series()

df_y2 = df_y2.replace(2.0,-1)
df_y2 = df_y2.replace(7.0,1)
df_y2 = df_y2.reset_index(drop=True)
df_y2 = df_y2
df_x2 = df_x2.to_numpy()
df_y2 = df_y2.to_numpy()
df_y2 = df_y2.reshape(len(df_y2),1)
df_y = df_y.replace(2.0,-1)
df_y = df_y.replace(7.0,1)
df_y = df_y.reset_index(drop=True)
df_x = df_x.to_numpy()
df_y = df_y.to_numpy()
df_y = df_y.reshape(len(df_y),1)


def predict2(X,w):
    X = X@w
    return X

def predict(X,w,df_y):
    predictions = predict2(X,w)
    accuracy = []
    for i in range(len(predictions)):
        if(sigmoid_function(predictions[i]) >= .5):
            predictions[i] = 1.0
        else:
            predictions[i] = -1.0
    falsecount = 0
    for i in range(len(predictions)):
        if(df_y[i] != predictions[i]):
            accuracy.append(False)
            falsecount = falsecount + 1
        else:
            accuracy.append(True)
    return falsecount/len(X)


def thetafunc(X):
    return np.random.randn(len(X[0])+1,1)
def generateX(X):
    return np.c_[np.ones((len(X),1)),X]
def sigmoid_function(X):
    return 1/(1+np.exp(-X))

def gradient_descent(X,y,alpha,itterations):
    falsecountList = []
    cost = []
    y_new = np.reshape(y,(len(y),1))
    vectX = generateX(X)
    w = np.zeros((len(X[0])+1,1))
    b = 0
    n = len(X)
    for i in range(itterations):
        
        
        
        z = -y*(b+vectX.dot(w))
        phiz = sigmoid_function(z)
        y_pred = sigmoid_function(y*(vectX.dot(w)) + b)
        Wgradient = (-1/n)*np.dot(vectX.T,y*phiz) + (2*alpha*w)
        Bgradient = (-1/n)*np.sum((y*phiz)) 
        w = w - (alpha * Wgradient)
        b = b - (alpha * Bgradient)
        cost_value = (np.sum(np.log(1 + phiz)) + alpha*np.linalg.norm(w,ord=2)**2) /len(vectX)
        cost.append(cost_value)
        falsecountList.append(predict(vectX,w,y))
        
    
    plt.plot(np.arange(1,itterations),cost[1:])
    plt.xlabel('Itterations')
    plt.ylabel('Cost')
    return b,w,falsecountList,vectX

b, theta,falseCountList,xvect =gradient_descent(df_x,df_y,.005,100)
df = pd.DataFrame(xvect@theta)
df.sort_values(by = 0, ascending = True)

plt.plot(falseCountList)
plt.xlabel('Itterations') 
plt.ylabel('% ERROR') 
  
# displaying the title
plt.title("Train set error")


zeros = np.ones(len(df_x))
df_x = np.column_stack((zeros,df_x))
df_y2.shape

b,w,falseco,x = gradient_descent(df_x2,df_y2,.005,200)

plt.plot(falseco)
plt.xlabel('Itterations') 
plt.ylabel('% ERROR') 
  
# displaying the title
plt.title("Test set error")


def predict2(X,w):
    X = X@w
    return X

predictions = predict2(df_x,theta)

for i in range(len(predictions)):
    if(predictions[i] >= .5):
        predictions[i] = 1.0
    else:
        predictions[i] = -1.0
falsecount = 0
for i in range(len(predictions)):
    if(df_y[i] != predictions[i]):
        falsecount = falsecount + 1


def gradient_descent2(X,y,alpha,itterations,w,b):
    # f = y - (m*X + b)
    
        # Updating m and b
        #m -= lr * (-2 * X.dot(f).sum() / N)
        #b -= lr * (-2 * f.sum() / N)
    
    cost = []
    y_new = np.reshape(y,(len(y),1))
    vectX = generateX(X)
   
    theta = thetafunc(X)
    n = len(X)
    for i in range(itterations):
        z = -y*(b+vectX.dot(w))
        phiz = sigmoid_function(z)
        y_pred = sigmoid_function(y*(vectX.dot(w)) + b)
        Wgradient = (-1/n)*np.dot(vectX.T,y*phiz) + (2*alpha*w)
        Bgradient = (-1/n)*np.sum((y*phiz)) 
        w = w - (alpha * Wgradient)
        b = b - (alpha * Bgradient)
        cost_value = (np.sum(np.log(1 + phiz)) + alpha*np.linalg.norm(w,ord=2)**2) /len(vectX)
        cost.append(cost_value)
        
    
    plt.plot(np.arange(1,itterations),cost[1:])
    plt.xlabel('Itterations')
    plt.ylabel('Cost')
    return b,w


def SGD(X,y,learning_rate,batchsize):
    wold = np.zeros((len(X[0])+1,1))
    bold = 0
    meanB = 0
    meanW = 0
    index = 0
    for i in range(int(len(X)/batchsize)):
        index = index + 1
        x1 = X[i*batchsize:(i+1)*batchsize]
        y1 = y[i*batchsize:(i+1)*batchsize]
        b, w = gradient_descent2(x1,y1,learning_rate,100,wold,bold)
        bold = b
        wold = w
        meanB = meanB + bold
        meanW = meanW + wold
        
    meanB = meanB/index
    meanW = meanW / index
    return meanB, meanW



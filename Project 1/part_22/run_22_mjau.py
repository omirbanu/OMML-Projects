# in order to execute please change folder to part_22
# python run_22_mjau.py
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import itertools
from sklearn.cluster import KMeans
import part_22_mjau as our_f

np.random.seed(1848425)
df = pd.read_excel("dataPoints.xlsx")
data = df.to_numpy()#change to matrix
data_train, data_rest = train_test_split(data, test_size=0.30)
data_test, data_validate = train_test_split(data_rest, test_size=0.50)
X_train = np.transpose(data_train[:,0:2])
Y_train = np.transpose(data_train[:,2:])
X_validate = np.transpose(data_validate[:,0:2])
Y_validate = np.transpose(data_validate[:,2:])
X_test = np.transpose(data_test[:,0:2])
Y_test = np.transpose(data_test[:,2:])
# inits
rho=0.00001
sigma=1.11
N=25
rep=25 # number of times repeating minimizing procedure
print("value of N: "+str(N))
print("value of rho: " + str(rho))
print("value of sigma: "+str(sigma))

# initializing an object of class with given values of hyperparameters
rbf=our_f.RBF_Q2(rho,sigma,N)
cv_data=np.concatenate((data_train,data_validate))
X_train = np.transpose(np.matrix(cv_data)[:,0:2])
Y_train = np.transpose(np.matrix(cv_data)[:,2:])
# starting time
st=time.time()
## calling function in order to repeat extreme learing 'rep' times
best,test_err,train_err, nfev,njev=our_f.q_2_rbf(rho,sigma,N,rep, X_train,Y_train,X_test,Y_test)
#V,C,omega=our_f.initializeParams_q22(N,X_train)

#res=minimize(rbf.reg_tr_error,V.flatten(), args=[X_train,Y_train, C,N,rho,sigma],method='CG',jac=our_f.grad_reg_error)
V_star=best[1]#res['x']
#nfev = res['nfev']
#njev = res['njev']
C = best[0]
stop=time.time()

print("method chosen: "+"CG")
print("Number of function evaluations: "+ str(nfev))
print("Number of gradient evaluations: "+str(njev))
print("time of optimizing: "+str(stop-st))
Y_hat = rbf.predict22(V_star,C,X_train).reshape((X_train.shape[1],1))
print("Train_error: "+str(our_f.tr_error(Y_hat.T,Y_train)))
Y_hat = rbf.predict22(V_star,C,X_test).reshape((X_test.shape[1],1))
print("Test_error: "+str(our_f.tr_error(Y_hat.T,Y_test)))

#Cross validation procedure for 2.2
#our_f.CV_exec(X_train, X_validate, X_test, Y_train, Y_validate, Y_test,data_train,data_validate)


# preparation for plotting
'''
xy = np.mgrid[-2:2.002:0.05,-1:1.002:0.05].reshape(2,-1).T
X = xy[:, 0].squeeze()
Y = xy[:, 1].squeeze()


XY=np.concatenate((X,Y)).reshape(2,X.shape[0]).T

Z = rbf.predict22(V_star,C,XY.T)
Z = np.transpose(Z)
Z = np.array(Z).flatten()

Z = Z.T.reshape(X.shape[0],)
our_f.plotting(X,Y,Z)
'''
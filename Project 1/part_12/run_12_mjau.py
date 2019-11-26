# in order to execute please change folder to part_12
# python run_12_mjau.py
import random
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
import part_12_mjau as our_f # file with all classes and methods

# dataset
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
# initial values of hyperparameters
rho=0.00001
sigma=1.11
N=25
print("value of N: "+str(N))
print("value of rho: " + str(rho))
print("value of sigma: "+str(sigma))
C,V,omega=our_f.initializeParams_q12(N,X_train)
rbf=our_f.RBF(rho,sigma,N)
cv_data=np.concatenate((data_train,data_validate)) # merging data of train and validation
X_train = np.transpose(np.matrix(cv_data)[:,0:2])
Y_train = np.transpose(np.matrix(cv_data)[:,2:])
st=time.time()
res=minimize(rbf.reg_tr_error,omega.flatten(), args=[X_train,Y_train,N,rho,sigma],method='CG')
omega_star=res['x']
stop=time.time()

print("method chosen: "+"CG")
print("Number of function evaluations: "+ str(res['nfev']))
print("Number of gradient evaluations: "+str(res['njev']))
print("time of optimizing: "+str(stop-st))
Y_hat = rbf.predict2(omega_star,X_train)
print("Train_error: "+str(our_f.tr_error(Y_hat,Y_train)))
Y_hat = rbf.predict2(omega_star,X_test)
print("Test_error: "+str(our_f.tr_error(Y_hat,Y_test)))

'''
# preparing to plot with parameters that we got from minimizer
xy = np.mgrid[-2:2.002:0.05,-1:1.002:0.05].reshape(2,-1).T
X = xy[:, 0].squeeze()
Y = xy[:, 1].squeeze()
XY=np.concatenate((X,Y)).reshape(2,X.shape[0]).T
Z = rbf.predict2(omega_star,XY.T) # getting predictions
Z = Z.T.reshape(X.shape[0],)
our_f.plotting(X,Y,Z)
'''

#### method CV_exec is written in order to perform cross validation
####our_f.CV_exec(X_train, X_validate, X_test, Y_train, Y_validate, Y_test,data_train,data_validate)
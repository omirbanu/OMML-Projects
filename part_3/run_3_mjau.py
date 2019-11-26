# in order to execute please change folder to part_3
# python run_3_mjau.py
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time
import itertools
from sklearn.cluster import KMeans
import part_3_mjau as our_f
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
cv_data=np.concatenate((data_train,data_validate))
X_train = np.transpose(np.matrix(cv_data)[:,0:2])
Y_train = np.transpose(np.matrix(cv_data)[:,2:])

N = 25

rho = 0.00001
sigma = 1.11

print("value of N: "+str(N))
print("value of rho: " + str(rho))
print("value of sigma: "+str(sigma))

def alg1(N,rho,sigma): # two block decomposition method
	E= 1
	theta = 0.5
	sub_pr = 0
	V,C,omega = our_f.initializeParams_q22(N,X_train)
	args=[X_train,Y_train,C,N,rho,sigma]
	args2=[X_train,Y_train,V,N,rho,sigma]
	rbf= our_f.RBF_Q3(rho,sigma,N)
	beg_err = rbf.reg_tr_error_V(V, args)
	beg_err2 = rbf.reg_tr_error_C(C, args2)
	#print(beg_err)
	#print(beg_err2)
	C_old=C*0.1
	njev,nfev=0,0
	e_k1, e_k2 = 1000000000*our_f.second_norm_jac(our_f.grad_reg_error_V(V,args)),1000000000*our_f.second_norm_jac(our_f.grad_reg_error_C(C,args2))
	while E>0.00001:
		V_old= V
		k=0
		g1=0
		g2=0
		
		args=[X_train,Y_train,C,N,rho,sigma]
		if (np.array_equal(C_old,C)==False): # first block
			res1 = minimize(rbf.reg_tr_error_V,V.flatten(), args=args,method='CG',jac=our_f.grad_reg_error_V)
			tr_E_V = res1['fun']
	    #updating
			g1 = our_f.second_norm_jac(our_f.grad_reg_error_V(res1['x'],args))
			k=0
			if tr_E_V< beg_err and (g1 < e_k1):
				sub_pr=sub_pr+1
				V = res1['x']
				njev=njev+res1['njev']
				nfev=nfev+res1['nfev']
				beg_err = tr_E_V
				e_k1 = theta*e_k1
				k=k+1
	    
	    
	    
		args2=[X_train,Y_train,V,N,rho,sigma]
		if (np.array_equal(V,V_old)==False): # second block
			res2 = minimize(rbf.reg_tr_error_C,C.flatten(), args=args2,method="CG",jac=our_f.grad_reg_error_C)
			tr_E_C = res2['fun']

			g2 = our_f.second_norm_jac(our_f.grad_reg_error_C(res2['x'],args2))
		#updating
			if  (abs(beg_err2-tr_E_C)<0.000000001):
				#print('4')
				return (V,C,sub_pr,nfev,njev)
			if tr_E_C < beg_err2 and (g2< e_k2):
				C = res2['x']
				njev=njev + res2['njev']
				nfev = nfev + res2['nfev']
				sub_pr=sub_pr+1
				beg_err2 = tr_E_C
				e_k2 = theta*e_k2
				k=k+1
			else:
				C_old=C
		if k==0:
			return(V,C,sub_pr,nfev,njev)
        
		E = g1+g2
	return (V,C,sub_pr,nfev,njev)

st =time.time()
# calling decomposition method
finalV,finalC,count_sub,nfev,njev=alg1(N,rho,sigma)
stop = time.time()
print("method chosen: "+"CG")

rbf= our_f.RBF_Q3(rho,sigma,N)
print("Number of function evaluations: "+ str(nfev))
print("Number of gradient evaluations: "+str(njev))
print("time of optimizing: "+str(stop-st))
Y_pred = rbf.predict22(finalV,finalC,X_train)
print("train_error:  "  +str(our_f.tr_error(Y_pred,Y_train)))

Y_pred = rbf.predict22(finalV,finalC,X_test)
print("test_error:  " + str(our_f.tr_error(Y_pred,Y_test)))

#print("count of sub_pr solved: "+str(count_sub))

'''
#plotting preparation
xy = np.mgrid[-2:2.002:0.05,-1:1.002:0.05].reshape(2,-1).T
X = xy[:, 0].squeeze()
Y = xy[:, 1].squeeze()

XY=np.concatenate((X,Y)).reshape(2,X.shape[0]).T

Z = rbf.predict22(finalV,finalC,XY.T)
Z = Z.T.reshape(X.shape[0],)
# plotting
our_f.plotting(X,Y,Z)
'''
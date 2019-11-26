
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
from PIL import Image
class RBF_Q3(object):
    
    def __init__(self, ro, sigma, N):
        self.ro = ro
        self.sigma = sigma 
        self.N = N

    def set_N(self, x): 
        self.N = x 
    def get_N(self): 
        return self.N     
    
    
    def second_norm(self,omega):
        return np.linalg.norm(omega)**2 

    def predict22(self,V,C,X):  

        N=self.N
        sig = self.sigma
        V=V.flatten().reshape((self.N,1))
        C=C.flatten().reshape((self.N,2))
        C1 = C[:,0].T
        C2 = C[:,1].T
        X1 = (X[0].T).reshape((X.shape[1],1))
        X2 = (X[1].T).reshape((X.shape[1],1))
        X1 = X1 - C1
        X2 = X2 - C2
        XX = -np.square(np.sqrt(np.square(X1)+np.square(X2))/sig)
        XX= np.exp(XX)
        CX= XX.dot(V)
        CX= CX.reshape((X.shape[1],))
        
        return CX


    def reg_tr_error_C(self,C,functionArgs):
        X=functionArgs[0]
        true=functionArgs[1]
        V=functionArgs[2]
        V = V.flatten().reshape((self.N,1))
        #C=C.flatten().reshape((self.N,2))
        #print(C)
        
        predicted=self.predict22(V,C,X)


        err=np.array(predicted)-true #err_all=np.sum(np.array(predicted)-true)**2
        err_all=err.dot(err.T)


        P=X.shape[1]
        return ((err_all)/(2*P)+self.ro*self.second_norm(C)).item(0)
    def reg_tr_error_V(self,V,functionArgs):
        X=functionArgs[0]
        true=functionArgs[1]
        C=functionArgs[2]
        V = V.flatten().reshape((self.N,1))
        
        predicted=self.predict22(V,C,X)


        err=np.array(predicted)-true
        err_all=err.dot(err.T)


        P=X.shape[1]
        return ((err_all)/(2*P)+self.ro*self.second_norm(V)).item(0)

def second_norm_jac(omega): 
    return np.linalg.norm(omega)

def initializeParams_q22(N, X):
    n=2
    n_y=1
    kmeans= KMeans(n_clusters=N, random_state=1848425).fit(X.T)
    
    C = kmeans.cluster_centers_ 
    V = np.random.randn(N,n_y) 
    a=pd.DataFrame(V)
    a[2]=C[:,0]
    a[3]=C[:,1]
    
    omega=np.matrix(a)
    
 
   
    return V,C,omega

def grad_reg_error_V(V, args): # gradient according to weights
    try:
        X = args[0]
        Y = args[1]
        C0 = args[2]
        N = args[3]
        rho = args[4]
        sigma = args[5]
    
    
        V=V.flatten().reshape((N,1))

        C1 = C0[:,0].T
        C2 = C0[:,1].T
   
        X1 = (X[0].T).reshape((X.shape[1],1))
        X2 = (X[1].T).reshape((X.shape[1],1))
  
        X1 = X1 - C1
        X2 = X2 - C2

        XX = -np.square(np.sqrt(np.square(X1)+np.square(X2))/sigma)

        

        Phi= np.exp(XX)    

        
        grad1=((1/X.shape[1])*(((Phi.T).dot((Phi.dot(V))-Y.T))))
        gg=(2*rho*V)
        grad=np.sum([grad1,gg],axis=0)#
    except:
        grad=V
    return grad.flatten()

def grad_reg_error_C(C0, args): # gradient according to centers
    
    X = args[0]
    Y = args[1]
    V = args[2]
    N = args[3]
    rho = args[4]
    sigma = args[5]
    
    C=C0.flatten().reshape((N,2))
    V=V.flatten().reshape((N,1))
    
    C1 = C[:,0].T
    C2 = C[:,1].T
    X1 = (X[0].T).reshape((X.shape[1],1))
    X2 = (X[1].T).reshape((X.shape[1],1))
    X1 = X1 - C1
    #print(X1.shape)
    X2 = X2 - C2
    XX = -np.square(np.sqrt(np.square(X1)+np.square(X2))/sigma)
    
    # derivative part
    Phi= np.exp(XX) * 2/np.square(sigma)
    
    
    Phi1 = np.multiply(Phi,X1)
    Phi2 = np.multiply(Phi,X2)
    
    #part where sum by neurons
    
    Phi_N = (np.exp(XX)).dot(V) - Y.T
    
    Phi_final_1 = np.multiply( V.T , (np.multiply(Phi_N , Phi1)).sum(axis=0) )+ rho*C1
    Phi_final_2 =  np.multiply(V.T ,(np.multiply(Phi_N,Phi2)).sum(axis=0) )+ rho*C2

    Final = np.zeros((N,2))
    Final[:,0]=Phi_final_1
    Final[:,1]=Phi_final_2
    Final = Final
    return Final.flatten()/X.shape[1]

def tr_error(Y_hat,Y_true):
	return(np.sum(np.array(Y_hat-Y_true)**2))/(Y_true.shape[1]*2)

def plotting(X,Y,Z): 
 
    fig = plt.figure(figsize=(70,40))
    ax = plt.axes(projection='3d')

    ax = fig.gca(projection='3d')
    ax.view_init(azim=-120)
    surf = ax.plot_trisurf(X,Y,Z, linewidth=0.010, antialiased=True,cmap='viridis')
    fig.savefig('3_Rbf')
    plt.grid()


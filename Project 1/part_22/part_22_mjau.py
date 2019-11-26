import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D
import time
import os
import itertools
from sklearn.cluster import KMeans
from PIL import Image
folder = os.getcwd()

class RBF_Q2(object): # RBF class
    
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
    
    def predict2(self,omega,X):  # predicting method with omega

        N=self.N
        sig = self.sigma
        omega=omega.flatten().reshape((N,3))
        V = omega[:,0] 
        C1 = omega[:,1].T
        C2 = omega[:,2].T
        X1 = (X[0].T).reshape((X.shape[1],1))
        X2 = (X[1].T).reshape((X.shape[1],1))
        X1 = X1 - C1
        X2 = X2 - C2
        XX = -np.square(np.sqrt(np.square(X1)+np.square(X2))/sig)
        XX= np.exp(XX)
        CX= XX.dot(V)
        CX= CX.reshape((X.shape[1],))        
        return CX

    def predict22(self,V,C,X):  # predicting function when centers and weights are give separately

        N=self.N
        sig = self.sigma
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


    def reg_tr_error(self,V,functionArgs):
        X=functionArgs[0]
        true=functionArgs[1]
        C=functionArgs[2]
        V = V.flatten().reshape((self.N,1))
        
        predicted=self.predict22(V,C,X)


        err=np.array(predicted)-true 
        err_all=err.dot(err.T)
        P=X.shape[1]  
        
        return (err_all.item(0))/(2*P)+self.ro*(self.second_norm(V))



def initializeParams_q22(N, X): # initializing centers and weights
    n=2
    n_y=1
    kmeans= KMeans(n_clusters=N).fit(X.T)
    
    C = kmeans.cluster_centers_ 
    
    V = np.random.randn(N,n_y) 
    a=pd.DataFrame(V)
    a[2]=C[:,0]
    a[3]=C[:,1]
    omega=np.matrix(a)
    
    return V,C,omega

def grad_reg_error(V, args):
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
        grad=np.sum([grad1,gg],axis=0)
    except:
        grad=V
    return grad.flatten()

def second_norm_jac(omega): 
    return np.linalg.norm(omega)
def tr_error(Y_hat,Y_true):
    return(np.sum(np.array(Y_hat-Y_true)**2))/(Y_true.shape[1]*2)

def mse(true,predicted):
    return (np.sum(np.array(true-predicted)**2))/(true.shape[1]*2)

def fivefoldCV_q22(params,X_train, X_validate, X_test, Y_train, Y_validate, Y_test ,data_train,data_validate):
    '''
    params a list N,rho,sigma
    '''
    
    K=5   # SO 4 folds for training 1-validation for testing; they switch every time
    cv_data=np.concatenate((data_train,data_validate))
    np.random.shuffle(cv_data)
    indices=np.arange(0,255,51)# [  0,  51, 102, 153, 204]
    folds=[]


    N=params[0]
    rho=params[1]#10**-5 #10-5 unti, 10-3
    sigma=params[2]#1

    V,C,omega = initializeParams_q22(N,X_train)

    rbf=RBF_Q2(rho,sigma,N)


    P=cv_data.shape[0]

    rbf.set_N(N)
    

    val_err_mse=[]
    train_err_mse=[]
    fun=[]
    jac_norm=[]
    init=[]

    data=cv_data.copy()


    res_df=pd.DataFrame(columns=['neurons','rho','sigma','fun','init','err_tr','jac_norm','err_val'])
    for i in range(len(indices)):
        cv_data=data
        if i<4:
            l=[i for i in range(indices[i],indices[i+1])]
            #(VALIDATION fold) for testing
            validate_cv=cv_data[indices[i]:indices[i+1],:]

            #train folds together for training
            df=pd.DataFrame(cv_data)
            train_cv=df.drop(df.index[l])



            X_train = np.transpose(np.matrix(train_cv)[:,0:2])
            Y_train = np.transpose(np.matrix(train_cv)[:,2:])
            X_validate = np.transpose(validate_cv[:,0:2])
            Y_validate = np.transpose(validate_cv[:,2:])
            init.append(mse(Y_train,rbf.predict22(V,C,X_train)))

            res=minimize(rbf.reg_tr_error,V, args=[X_train,Y_train,C,N,rho,sigma],method='CG',jac=grad_reg_error)
     #       
            V=res['x']
            fun.append(res['fun'])
            jac_norm.append(second_norm_jac(res['jac'].T))

        #    err_tr=mse(reg_tr_error(omega.flatten(),[X_train,Y_train]))
            err_tr=mse(Y_train,rbf.predict22(V,C,X_train))
            err_val=mse(Y_validate,rbf.predict22(V,C,X_validate))

            train_err_mse.append(err_tr)
            val_err_mse.append(err_val)
            
            


        else:
            #for the last element
            l=list([i for i in range(indices[i],255)])
            #(VALIDATION fold) for testing
            validate_cv=cv_data[indices[i]:,:]

            #train folds together for training
            df=pd.DataFrame(cv_data)
            train_cv=df.drop(df.index[l])
            init.append(mse(Y_train,rbf.predict22(V,C,X_train)))

            #CHOSEN OMEGA? ->Fitting of the model
            res=minimize(rbf.reg_tr_error,V, args=[X_train,Y_train,C,N,rho,sigma],method='CG',jac=grad_reg_error)
          #  for i in range(10):
          #      omega=res['x']
          #      res=minimize(mlp.reg_tr_error,omega.flatten(), args=[X_train,Y_train],method='L-BFGS-B')
            
            V=res['x']
            fun.append(res['fun'])
            jac_norm.append(second_norm_jac(res['jac'].T))


    
            err_tr=mse(Y_train,rbf.predict22(V,C,X_train))
       #     err_tr=mse(reg_tr_error(omega.flatten(),[X_train,Y_train]))
            err_val=mse(Y_validate,rbf.predict22(V,C,X_validate))

            train_err_mse.append(err_tr)
            val_err_mse.append(err_val)

    res_df=res_df.append({'neurons':N,'rho':rho,'sigma':sigma,'fun':np.mean(fun),'init':np.mean(init),\
                              'err_tr':np.mean(train_err_mse),'jac_norm':np.mean(jac_norm),\
                              'err_val':np.mean(val_err_mse)},ignore_index=True )
    return res_df

def CV_exec(X_train, X_validate, X_test, Y_train, Y_validate, Y_test,data_train,data_validate):
    neurons_count=[2,9,25,33,55]
    rho_values=[0.0001,0.00001]
    sigma_vals=[1.11,1.455,1.99]

    all_poss_conf=[]
    for i in neurons_count:
        for r in rho_values:
            for s in sigma_vals:
                all_poss_conf.append((i,r,s))

    from tqdm import tqdm
    res_df=pd.DataFrame()
    for conf in tqdm(all_poss_conf):
        N=conf[0]
        rho=conf[1]
        sigma=conf[2]
        res_df=res_df.append(fivefoldCV_q22(conf,X_train, X_validate, X_test, Y_train, Y_validate, Y_test,data_train,data_validate))
    # -CV
    res_df.to_csv('q22_cv_results.csv')

def plotting(X,Y,Z): 
 
    fig = plt.figure(figsize=(70,40))
    ax = plt.axes(projection='3d')
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-120)
    surf = ax.plot_trisurf( X,Y, Z, linewidth=0.010, antialiased=True,cmap='viridis')
    fig.savefig('22_Rbf')
    plt.grid()


    
def q_2_rbf(rho,sigma,N,rep, X,Y,X_test,Y_test):
    test_err=[]
    train_err=[]
    #from tqdm import tqdm
    for r in range(rep):
        V,C,omega=initializeParams_q22(N,X)
        
        rbf=RBF_Q2(rho,sigma,N)
        res=minimize(rbf.reg_tr_error,V.flatten(), args=[X,Y, C,N,rho,sigma],method='CG',jac=grad_reg_error)
        V_star=res['x']
        
        curr_err=mse(Y_test,rbf.predict22(V_star,C,X_test))#rbf.reg_tr_error(V_star,[X_test,Y_test,C])
        test_err.append(curr_err)
        train_err.append(mse(Y,rbf.predict22(V_star,C,X)))
        if np.min(test_err)==curr_err:
            best = (C,V_star)
            nfev=res['nfev']
            njev=res['njev']
    return best,test_err,train_err,nfev,njev

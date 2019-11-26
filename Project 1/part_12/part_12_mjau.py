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
from PIL import Image
'''
method below returns initial values for centers C and weights V
'''
def initializeParams_q12(N, X_train):
    n=2
    n_y=1
    idxx = list(np.random.choice(X_train.shape[1], N))
    C = X_train[:,idxx].T
    V = np.random.randn(N,n_y) 
    a=pd.DataFrame(V)
    a[2]=C[:,0]
    a[3]=C[:,1]
    omega=np.matrix(a)
    return C,V,omega

class RBF(object): #class containing prediction function and error function     
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

    def predict2(self,omega,X): #predicting function 

        N=self.N # neurons
        sig = self.sigma # values if sigma
        omega=omega.flatten().reshape((N,3))
        V = omega[:,0] #taking weights from omega
        C1 = omega[:,1].T # values from centers with first index 
        C2 = omega[:,2].T #  values from centers with second index
        X1 = (X[0].T).reshape((X.shape[1],1)) #  values from inputs with first index
        X2 = (X[1].T).reshape((X.shape[1],1)) #  values from inputs with second index
        X1 = X1 - C1 # preparing first term of input for activation function
        X2 = X2 - C2 # preparing second term of input for activation function
        XX = -np.square(np.sqrt(np.square(X1)+np.square(X2))/sig) # square of norm of (X - C)/sigma
        XX= np.exp(XX) # putting the term above to exponenta
        CX= XX.dot(V) # final prediction
        CX= CX.reshape((X.shape[1],)) # reshaping        
        return CX

    def reg_tr_error(self,omega,functionArgs):
        X=functionArgs[0] # inputs
        true=functionArgs[1] # true values of outputs
        predicted=self.predict2(omega,X) # predictions
        err=np.array(predicted)-true 
        err_all=err.dot(err.T) 
        P=X.shape[1] # number of samples
        # this method return mse/2 summed with regularization term
        return ((err_all)/(2*P)+self.ro*self.second_norm(omega)).item(0)
def fivefoldCV_q12(params, X_train, X_validate, X_test, Y_train, Y_validate, Y_test,data_train,data_validate):
    '''
    params a list N,rho,sigma
    '''
    

    K=5   # SO 4 folds for training 1-validation for testing; they switch every time
    cv_data=np.concatenate((data_train,data_validate))
    np.random.shuffle(cv_data)
    indices=np.arange(0,255,51)# [  0,  51, 102, 153, 204]
    folds=[]

    N=params[0]
    rho=params[1]
    sigma=params[2]

    C,V,omega=initializeParams_q12(N,X_train)

    rbf=RBF(rho,sigma,N)


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
            init.append(mse(Y_train,rbf.predict2(omega.flatten(),X_train))) 
            #CHOSEN OMEGA? ->Fitting of the model
            res=minimize(rbf.reg_tr_error,omega.flatten(), args=[X_train,Y_train],method='CG')
     
            omega=res['x']
            fun.append(res['fun'])
            jac_norm.append(second_norm_jac(res['jac'].T))

            err_tr=mse(Y_train,rbf.predict2(omega.flatten(),X_train))
            err_val=mse(Y_validate,rbf.predict2(omega.flatten(),X_validate))

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
            init.append(mse(Y_train,rbf.predict2(omega.flatten(),X_train)))

            #CHOSEN OMEGA? ->Fitting of the model
            res=minimize(rbf.reg_tr_error,omega.flatten(), args=[X_train,Y_train],method='CG')#,jac=total_grad)
         
            omega=res['x']
            fun.append(res['fun'])
            jac_norm.append(second_norm_jac(res['jac'].T))     
            err_tr=mse(Y_train,rbf.predict2(omega.flatten(),X_train))
            err_val=mse(Y_validate,rbf.predict2(omega.flatten(),X_validate))
            train_err_mse.append(err_tr)
            val_err_mse.append(err_val)

    res_df=res_df.append({'neurons':N,'rho':rho,'sigma':sigma,'fun':np.mean(fun),'init':np.mean(init),\
                              'err_tr':np.mean(train_err_mse),'jac_norm':np.mean(jac_norm),\
                              'err_val':np.mean(val_err_mse)},ignore_index=True )
    return res_df

def second_norm_jac(omega): 
    return np.linalg.norm(omega)

def mse(true,predicted):
    return (np.sum(np.array(true-predicted)**2))/(true.shape[1]*2)
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
        res_df=res_df.append(fivefoldCV_q12(conf,X_train, X_validate, X_test, Y_train, Y_validate, Y_test,data_train,data_validate))
    # saving results
    res_df.to_csv('q12_cv_results.csv')

def tr_error(Y_hat,Y_true):
	return(np.sum(np.array(Y_hat-Y_true)**2))/(Y_true.shape[1]*2)

def plotting(X,Y,Z):  
    fig = plt.figure(figsize=(70,40))
    ax = plt.axes(projection='3d')	
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-120)
    surf = ax.plot_trisurf( X,Y, Z, linewidth=0.010, antialiased=True,cmap='viridis')

    fig.savefig('12_Rbf')
    
    plt.grid()

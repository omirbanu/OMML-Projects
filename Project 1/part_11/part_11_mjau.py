#!/usr/bin/env python
# coding: utf-8
'''
Part 1.1 - Group mjau

    Question 1.1. (Full minimization)

         MLP network
'''
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
from datetime import timedelta
from PIL import Image
np.random.seed (1861402)


# Method for initializing Params randomly (W, bias and V)
def initializeParams(N):
    n = 2
    n_y = 1
    W = np.random.randn ( N, n )
    bias = np.random.randn ( N, 1 )
    V = np.random.randn ( n_y, N )
    a = pd.DataFrame ( V.T )
    a[2] = W[:, 0]
    a[3] = W[:, 1]
    a[4] = bias
    omega = np.matrix ( a )

    return W, bias, V, omega



# MLP class defined with its method for error calculation
#objective function -> regularized mse
class MLP ( object ):

    def __init__(self, ro, sigma, N):
        self.ro = ro
        self.sigma = sigma
        self.N = N

    def set_N(self, x):
        self.N = x

    def get_N(self):
        return self.N

    def second_norm(self, omega):
        return np.linalg.norm ( omega ) ** 2

    def activation_f(self, t, sigma):
        return (np.exp ( 2 * sigma * t ) - 1) / (np.exp ( 2 * sigma * t ) + 1)

    def predict(self, omega, X):
        N = self.N
        V = omega.T[:N].reshape ( 1, N )
        W = omega.T[N:N + 2 * N].reshape ( N, 2 )

        bias = omega.T[N + 2 * N:].reshape ( N, 1 )

        t = W.dot ( X ) - bias

        predicted_values = V.dot ( self.activation_f ( t, self.sigma ) )
        return predicted_values  # , W,bias

    def reg_tr_error(self, omega, functionArgs):
        X = functionArgs[0]
        true = functionArgs[1]
        predicted = self.predict ( omega, X )

        err = np.array ( predicted ) - true  # err_all=np.sum(np.array(predicted)-true)**2
        err_all = err.dot ( err.T )

        P = X.shape[1]
        return ((err_all) / (2 * P) + self.ro * self.second_norm ( omega )).item ( 0 )

# just sample try to test if method works
def tryout1():
    ro = 10 ** -5  # 10-5 unti, 10-3
    sigma = 1
    N = 7
    # Initialize params randomly (W, bias and V)
    W, bias, V, omega = initializeParams ( N )
    # Initialized params:
    print ( V, '\n' )
    print ( W, '\n' )
    print ( bias )
    mlp = MLP ( ro, sigma, N )
    mlp.reg_tr_error ( omega.flatten (), [X_train, Y_train] )

    # after minimizing
    res = minimize ( mlp.reg_tr_error, omega.flatten (), args=[X_train, Y_train], method='BFGS' )
    print ( res['fun'], '\n', res['x'] )


    print ( mlp.reg_tr_error ( res['x'].flatten (), [X_validate, Y_validate, N] ) )
    print ( mlp.reg_tr_error ( res['x'].flatten (), [X_test, Y_test, N] ) )
    return


''''
Run method to try out
tryout1()
'''


# Hyperparameters:
#
#     the number of neurons N of the hidden layer
#     the spread delta in the activation function g(t)
#     the regularization parameter rho


def second_norm_jac(omega):
    return np.linalg.norm ( omega )


def mse(true, predicted):
    return (np.sum ( np.array ( true - predicted ) ** 2 )) / (2 * true.shape[1])

# ## DRAFT VERSION OF THE ""HOMEMADE"" CROSS-VALIDATION METHOD

# repeated CROSS Validation was not implemented! But a regular 5 fold CV!

def fivefoldCV(params,cv_data):
    '''
    params a list N,rho,sigma
    '''
    global X_train, X_validate, X_test, Y_train, Y_validate, Y_test

    K = 5  # SO 4 folds for training 1-validation for testing; they switch every time

    np.random.shuffle ( cv_data )
    indices = np.arange ( 0, 255, 51 )  # [  0,  51, 102, 153, 204]
    folds = []

    N = params[0]
    rho = params[1]  # 10**-5 #10-5 until 10-3
    sigma = params[2]  # from 1 to 2 try out

    W, bias, V, omega = initializeParams ( N )
    init_omega=omega
    mlp = MLP ( rho, sigma, N )

    P = cv_data.shape[0]

    mlp.set_N ( N )

    val_err_mse = []
    train_err_mse = []
    fun = []
    jac_norm = []
    init_tr_err = []
    initial_fun_vals=[]
    data = cv_data.copy ()
    res_df = pd.DataFrame (columns=['neurons', 'rho', 'sigma', 'init_trr_err', 'fun',\
                                    'err_tr', 'jac_norm', 'err_val'])

    for i in range ( len ( indices ) ):
        cv_data = data
        if i < 4:  ###FOR FIRST 4 FOLDS
            l = [i for i in range ( indices[i], indices[i + 1] )]
            # (VALIDATION fold) for testing
            validate_cv = cv_data[indices[i]:indices[i + 1], :]

            # train folds together for training
            df = pd.DataFrame ( cv_data )
            train_cv = df.drop ( df.index[l] )

            X_train = np.transpose ( np.matrix ( train_cv )[:, 0:2] )
            Y_train = np.transpose ( np.matrix ( train_cv )[:, 2:] )
            X_validate = np.transpose ( validate_cv[:, 0:2] )
            Y_validate = np.transpose ( validate_cv[:, 2:] )

            # CHOSEN OMEGA? ->Fitting of the model

            res = minimize ( mlp.reg_tr_error, omega.flatten (), args=[X_train, Y_train],method='BFGS')
            print(res['message'], res['fun'],'success', res['success'])
   #         print(res['nfev'],res['njev'] )
            omega = res['x']
            fun.append ( res['fun'] )
            jac_norm.append ( second_norm_jac ( res['jac'].T ) )


        ##    initial_fun_vals.append(mlp.reg_tr_error ( init_omega.flatten (), [X_validate, Y_validate, N] ))

        #    err_tr=mse(reg_tr_error(omega.flatten(),[X_train,Y_train]))
            err_tr = mse ( Y_train, mlp.predict ( omega.flatten (), X_train ) )
            err_val = mse ( Y_validate, mlp.predict ( omega.flatten (), X_validate ) )

            train_err_mse.append ( err_tr )
            val_err_mse.append ( err_val )

        else:

            # for the last element (###FOR the last FOLD)
            l = list ( [i for i in range ( indices[i], 255 )] )
            # (VALIDATION fold) for testing
            validate_cv = cv_data[indices[i]:, :]

            # train folds together for training
            df = pd.DataFrame ( cv_data )
            train_cv = df.drop ( df.index[l] )
            init_tr_err.append ( mse ( Y_train, mlp.predict ( init_omega.flatten (), X_train ) ) )
            # CHOSEN OMEGA? ->Fitting of the model
            res = minimize ( mlp.reg_tr_error, omega.flatten (), args=[X_train, Y_train], method='BFGS',options={'disp':True})

            print ( res['message'], res['fun'], 'success', res['success'] )
            omega = res['x']
            fun.append ( res['fun'] )
            jac_norm.append ( second_norm_jac ( res['jac'].T ) )
            err_tr = mse ( Y_train, mlp.predict ( omega.flatten (), X_train ) )
            err_val = mse ( Y_validate, mlp.predict ( omega.flatten (), X_validate ) )

            train_err_mse.append ( err_tr )
            val_err_mse.append ( err_val )
  ##          initial_fun_vals.append(mlp.reg_tr_error ( init_omega.flatten (), [X_validate, Y_validate, N] ))


    res_df = res_df.append ({'neurons': N, 'rho': rho, 'sigma': sigma, 'fun': np.mean ( fun ), 'init_trr_err': np.mean ( init_tr_err ),\
         'err_tr': np.mean ( train_err_mse ), 'jac_norm': np.mean ( jac_norm ), 'err_val': np.mean ( val_err_mse )}, ignore_index=True)
    print(res_df)
    return res_df

def plotTVerror():
    res_df_all = pd.read_csv ( 'cv_results_MLP.csv' )
    min_val_err = res_df_all.loc[res_df_all.err_val.idxmin ()]
    s=min_val_err['sigma']
    r=min_val_err['rho']
    reduced_res_df = res_df_all[(res_df_all.rho == r) & (res_df_all.sigma == s) & (res_df_all.neurons%3==0)& (res_df_all.neurons>9) & (res_df_all.neurons<70)]
    fig = plt.figure ( figsize=(20, 10) )

    plt.plot ( reduced_res_df.neurons, reduced_res_df.err_tr, linewidth=3, label='sigma='+str(s)+ ', rho='+str(r)+'   (Training error)',color='orange')
    plt.plot ( reduced_res_df.neurons, reduced_res_df.err_val, linewidth=3, label='sigma='+str(s)+ ', rho='+str(r)+'   (Validation error)',color='orchid')
    t='Training and Validation error MLP-full minimization'

    plt.title(t)
    plt.legend ()
    plt.grid ()
    plt.show ()
    fig.savefig (t)
#    fig = plt.figure ( figsize=(20, 10) )
#    plt.plot ( res_df_all.neurons, res_df_all.init_trr_err, linewidth=3, label='Initial Training error' )
#    fig.savefig ('Initial training error')


def plotting(omega_star,mlp):
    fig = plt.figure ( figsize=(70, 40) )
    ax = plt.axes ( projection='3d' )
    omega2 = omega_star

    xy = np.mgrid[-2:2.002:0.02, 1:-1.002:-0.02].reshape ( 2, -1 ).T

    X = xy[:, 0].squeeze ()
    Y = xy[:, 1].squeeze ()

    XY = np.concatenate ( (X, Y) ).reshape ( 2, X.shape[0] ).T

    Z = mlp.predict ( omega_star, XY.T )
    Z = Z.T.reshape ( X.shape[0], )

    ax = fig.gca ( projection='3d' )
    ax.view_init(azim=-120)
#    ax.set_xlim ( -2, 2 )
#    ax.set_ylim ( 1, -1 )
 #   ax.set_zlim ( -2, 6 )
    surf = ax.plot_trisurf ( X, Y, Z, linewidth=0.010, antialiased=True, cmap='viridis' )
    fig.savefig ( '11_MLP' )
    plt.grid()
    plt.show()




# ### FINDING AN OPTIMAL configuration (the best approximation we found)

# ### Test error on the best configuration we found


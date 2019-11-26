'''
Part 2.1 - Group mjau


    Question 2.1. (V based minimization)

         Extreme learning

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

class MLP_Q2 ( object ):
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

    def predict(self, X, W, bias, V):
        N = self.N
        t = W.dot ( X ) - bias
        predicted_values = V.dot ( self.activation_f ( t, self.sigma ) )
        return predicted_values  # , W,bias

    def reg_tr_error(self, V, functionArgs):
        X = functionArgs[0]
        true = functionArgs[1]
        W = functionArgs[2]
        bias = functionArgs[3]

        predicted = self.predict ( X, W, bias, V )

        err = np.array ( predicted ) - true  # err_all=np.sum(np.array(predicted)-true)**2
        err_all = err.dot ( err.T )

        P = X.shape[1]
        a = pd.DataFrame ( V.T )
        a[2] = W[:, 0]
        a[3] = W[:, 1]
        a[4] = bias
        omega_upd = np.matrix ( a )
        return ((err_all) / (2 * P) + self.ro * self.second_norm ( omega_upd )).item ( 0 )


def initializeParams_q2(N):
    n = 2
    n_y = 1
    W = np.random.randn ( N, n ) * np.random.randn ( N, n ) * np.random.randn ( N, n )
    bias = np.random.randn ( N, 1 ) * np.random.randn ( N, 1 ) * np.random.randn ( N, 1 )
    V = np.random.randn ( n_y, N )
    a = pd.DataFrame ( V.T )
    a[2] = W[:, 0]
    a[3] = W[:, 1]
    a[4] = bias
    omega = np.matrix ( a )

    return W, bias, V, omega

def mse(true,predicted):
    return (np.sum(np.array(true-predicted)**2))/true.shape[1]

def q_2_mlp(rho, sigma, N, rep, X, Y, X_test, Y_test):
    test_err = []
    train_err = []

    for r in tqdm ( range ( rep ) ):
        W, bias, V, omega = initializeParams_q2 ( N )
        mlp = MLP_Q2 ( rho, sigma, N )
        # Optimizing just vector V with minimize
        res = minimize ( mlp.reg_tr_error, V, args=[X, Y, W, bias], method='CG' )
        V_star = res['x']
        curr_err = mlp.reg_tr_error ( V_star, [X_test, Y_test, W, bias] )
        test_err.append ( curr_err )
        train_err.append ( mlp.reg_tr_error ( V_star, [X, Y, W, bias] ) )
        if np.min ( test_err ) == curr_err:
            best = (W, bias, V_star)
    return best, test_err, train_err


def q_2_mlp(rho, sigma, N, rep, X, Y, X_test, Y_test):
    test_err = []
    train_err = []

    for r in tqdm ( range ( rep ) ):
        W, bias, V, omega = initializeParams_q2 ( N )
        mlp = MLP_Q2 ( rho, sigma, N )
        # Optimizing just vector V with minimize
        res = minimize ( mlp.reg_tr_error, V, args=[X, Y, W, bias], method='CG' )
        V_star = res['x']
        curr_err = mlp.reg_tr_error ( V_star, [X_test, Y_test, W, bias] )
        test_err.append ( curr_err )
        train_err.append ( mlp.reg_tr_error ( V_star, [X, Y, W, bias] ) )
        if np.min ( test_err ) == curr_err:
            best = (W, bias, V_star)
    return best, test_err, train_err




def plotting_q2(W_inp, bias_inp, V_inp,mlp):
    fig = plt.figure ( figsize=(70, 40) )
    ax = plt.axes ( projection='3d' )
    a = pd.DataFrame ( V_inp.T )
    a[2] = W_inp[:, 0]
    a[3] = W_inp[:, 1]
    a[4] = bias_inp
    omega2 = np.matrix ( a )

    #  xy = np.mgrid[-1:1.002:0.05, -2:2.002:0.05].reshape(2,-1).T
    xy = np.mgrid[-2:2.002:0.02, 1:-1.002:-0.02].reshape ( 2, -1 ).T

    X = xy[:, 0].squeeze ()
    Y = xy[:, 1].squeeze ()

    XY = np.concatenate ( (X, Y) ).reshape ( 2, X.shape[0] ).T

    Z = mlp.predict ( XY.T, W_inp, bias_inp, V_inp )
    Z = Z.T.reshape ( X.shape[0], )

    ax = fig.gca ( projection='3d' )
    ax.view_init(azim=-120)

    #ax.set_xlim ( -2, 2 )
    #ax.set_ylim ( 1, -1 )
    #ax.set_zlim ( -2, 6 )

    surf = ax.plot_trisurf ( X, Y, Z, linewidth=0.010, antialiased=True, cmap='viridis' )
    fig.savefig ( '21_MLP_EL' )
    plt.grid ()

    plt.show ()


def val_error(X_train,Y_train,x_val,y_val,rho, sigma, N):
    # <---------- Making a prediction --------->#
    # read file where V star was saved with W and b
    omega_upd = pd.read_csv ( 'results_El.csv', usecols=[1, 2, 3, 4] )

    # read just the V from the files to avoid reexecution now
    vv = omega_upd.values[:, 3].reshape ( N, 1 ).T
    # now training just on TRAINING data



    W, bias, V, omega = initializeParams_q2 ( N )
    ww = W
    bb = bias

    mlp = MLP_Q2 ( rho, sigma, N )
    # Optimizing just vector V with minimize and taking optmized V vv is already been optimized
    res = minimize ( mlp.reg_tr_error, vv, args=[X_train, Y_train, ww, bb], method='CG' )
    V_star = res['x']
    Y_predicted = mlp.predict ( x_val, ww, bb, V_star )
    #final_err_tr = mse ( Y_train, mlp.predict ( X_train, ww, bb, V_star ) )
    err_test = mse ( y_val, Y_predicted )
    return err_test

#finalTest()
#plotting_q2(ww,bb,vv)


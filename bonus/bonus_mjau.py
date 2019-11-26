import mlp_11 as pm
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import time
from tqdm import tqdm
np.random.seed ( 1861402 )

# Define train, test and validation set
# Reading the file
df = pd.read_excel ( "dataPoints.xlsx" )
data = df.to_numpy ()  # change to matrix

# Create input vectors
X_train = np.transpose ( data[:, 0:2] )
Y_train = np.transpose ( data[:, 2:] )

params_11=pd.read_csv('../part_11/chosen_params.txt').loc[0]
# PARAMS chosen from hyperpatemeter optimization in MLP full minimization in Q1.1
N = final_N = int(params_11['N'])
sigma = final_sigma = params_11['sigma']
rho = final_rho = params_11['rho']

print(N,sigma,rho)

W, bias, V, omega = pm.initializeParams ( N )
mlp = pm.MLP ( rho, sigma, N )
res = minimize ( mlp.reg_tr_error, omega.flatten (), args=[X_train, Y_train], method='BFGS',options={'disp':True})
omega_star=res['x']


# Values are stored --> optimized params--> omega


##pd.DataFrame(omega_star).to_csv("omega.csv")

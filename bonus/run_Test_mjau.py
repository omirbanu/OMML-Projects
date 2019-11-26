import mlp_11 as pm
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import time
from tqdm import tqdm
np.random.seed ( 1861402 )
# Reading the file
df = pd.read_excel ( "dataPointsTest.xlsx" )
data = df.to_numpy ()  # change to matrix

# Create input vectors
X_test = np.transpose ( data[:, 0:2] )
Y_test = np.transpose ( data[:, 2:] )

omega_star=pd.read_csv('omega.csv')
omega_star=omega_star.values
#print(omega_star)


params_11=pd.read_csv('../part_11/chosen_params.txt').loc[0]
# PARAMS chosen from hyperpatemeter optimization in MLP full minimization in Q1.1
N = final_N = int(params_11['N'])
sigma = final_sigma = params_11['sigma']
rho = final_rho = params_11['rho']

mlp = pm.MLP ( rho, sigma, N )

predicted=mlp.predict ( omega_star.flatten (), X_test )

final_test_err = pm.mse ( Y_test,  predicted)
print('MSE: ',final_test_err)

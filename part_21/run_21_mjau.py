'''
Part 2.1 - Group mjau


a little memo:
The file that we used for reading has stored V, bias and W.
BUT we are reading V in order to create the approximation f-on here faster and
prevent slow execution time now; since we are repating that process N, whereby N=10
times it is slowly executing it and instead of you running it now we stored it and read it.

WHILE bias and W are randomly initialized of course.
'''

import part_21_mjau as pm2
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


# Reading the file
df = pd.read_excel ( "dataPoints.xlsx" )
data = df.to_numpy ()  # change to matrix


# Define train, test and validation set
data_train, data_rest = train_test_split ( data, test_size=0.30 )
data_test, data_validate = train_test_split ( data_rest, test_size=0.50 )

# Create input vectors
X_train = np.transpose ( data_train[:, 0:2] )
Y_train = np.transpose ( data_train[:, 2:] )
X_validate = np.transpose ( data_validate[:, 0:2] )
Y_validate = np.transpose ( data_validate[:, 2:] )
X_test = np.transpose ( data_test[:, 0:2] )
Y_test = np.transpose ( data_test[:, 2:] )


params_11=pd.read_csv('../part_11/chosen_params.txt').loc[0]
# PARAMS chosen from hyperpatemeter optimization in MLP full minimization in Q1.1
N = final_N = int(params_11['N'])
sigma = final_sigma = params_11['sigma']
rho = final_rho = params_11['rho']
final_opt_solver = 'CG'




'''
# not calling cause V star is saved to avoid slow execution of py file now in training
iter_num=10
best_V_W_b, test_errs, train_errs = pm2.q_2_mlp( rho, sigma, N, iter_num, X_train, Y_train, X_test, Y_test )

WW = best_V_W_b[0]  # W -> w1, w2
res_q2 = pd.DataFrame ( WW[:, 0] )  # w1
res_q2[2] = WW[:, 1]  # w2
res_q2[3] = best_V_W_b[1]  # bias
res_q2[4] = best_V_W_b[2]  # Vstar
res_q2.to_csv ( 'results_El.csv' )  # columns = w1, w2, bias, vstar
'''



#<---------- Making a prediction --------->#

# read file where V star was saved with W and b
omega_upd = pd.read_csv ( 'results_El.csv', usecols=[1, 2, 3, 4] )

# read just the V from the files to avoid reexecution now
vv = omega_upd.values[:, 3].reshape ( N, 1 ).T
cv_data = np.concatenate ( (data_train, data_validate) )  # now training and validation used together as TRAINING data
np.random.shuffle(cv_data)
X_train = np.transpose ( np.matrix ( cv_data )[:, 0:2] )
Y_train = np.transpose ( np.matrix ( cv_data )[:, 2:] )

W, bias, V, omega = pm2.initializeParams_q2 ( N )

ww = W
bb = bias

mlp = pm2.MLP_Q2 ( rho, sigma, N )
# Optimizing just vector V with minimize

###measuring time start
start = time.time ()
res = minimize ( mlp.reg_tr_error, vv, args=[X_train, Y_train, ww, bb], method='CG' )
end = time.time ()
###measuring time end

V_star = res['x']
Y_predicted = mlp.predict ( X_test, ww, bb, V_star )
final_err_tr = pm2.mse ( Y_train, mlp.predict ( X_train, ww, bb, V_star ) )
err_test = pm2.mse ( Y_test, Y_predicted )

opt_time = end - start

output = f"""\
   {'-' * 75}
   # Number of neurons N chosen : {final_N}
   # Value of sigma chosen : {final_sigma}
   # Value of r chosen : {final_rho}
   # Optimization solver chosen : {final_opt_solver}  
   # Number of function evaluations : {res['nfev']}
   # Number of gradient evaluations : {res['njev']}
   # Time for optimizing the network : {opt_time}   
   # Training Error : {final_err_tr} 
   # Test Error : {err_test}

   {'-' * 75}
   """
print ( output )

#<---------- Plotting approximation f-on --------->#

ww = omega_upd.values[:, 0:2]
bb = omega_upd.values[:, 2].reshape ( N, 1 )
vv = omega_upd.values[:, 3].reshape ( N, 1 ).T
#pm2.plotting_q2(ww, bb, vv,mlp)


#<---------- validation error --------->#
'''
val_errs=[]
for i in range(3):
   val_err=pm2.val_error(X_train,Y_train,X_validate,Y_validate,rho, sigma, N)
   val_errs.append(val_err)
   print('validation error ',val_err)
print('final validation error ',np.mean(val_errs))
'''
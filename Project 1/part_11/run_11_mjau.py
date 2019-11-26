import part_11_mjau as pm
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
data_train, data_rest = train_test_split ( data, test_size=0.30 )
data_test, data_validate = train_test_split ( data_rest, test_size=0.50 )

# Create input vectors
X_train = np.transpose ( data_train[:, 0:2] )
Y_train = np.transpose ( data_train[:, 2:] )
X_validate = np.transpose ( data_validate[:, 0:2] )
Y_validate = np.transpose ( data_validate[:, 2:] )
X_test = np.transpose ( data_test[:, 0:2] )
Y_test = np.transpose ( data_test[:, 2:] )


# reading the results from the CV
# read file where CV was saved
res_df_all = pd.read_csv ( 'cv_results_MLP.csv' )
res_df_all = res_df_all[res_df_all.neurons < 50]  # to prevent overfitting
vals = res_df_all.loc[res_df_all.err_val.idxmin ()]  # find the one where validation error was minimized

# best hyperparams found via 5 CV
N = final_N = int ( vals['neurons'] )
sigma = final_sigma = vals['sigma']
rho = final_rho = vals['rho']
final_opt_solver = 'BFGS'

W, bias, V, omega = pm.initializeParams ( N )
mlp = pm.MLP ( rho, sigma, N )
cv_data = np.concatenate (
    (data_train, data_validate) )  # now training and validation used together as TRAINING data
# np.random.shuffle(cv_data)

X_train = np.transpose ( np.matrix ( cv_data )[:, 0:2] )
Y_train = np.transpose ( np.matrix ( cv_data )[:, 2:] )

###measuring time start
start = time.time ()
res = minimize ( mlp.reg_tr_error, omega.flatten (), args=[X_train, Y_train], method=final_opt_solver)#,options={'disp':True})
#print(res['message'])

end = time.time ()
###measuring time end
omega_star = res['x']

final_err_tr = pm.mse ( Y_train, mlp.predict ( omega_star.flatten (), X_train ) )
err_test = pm.mse ( Y_test, mlp.predict ( omega_star.flatten (), X_test ) )
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
with open ( "chosen_params.txt", "w" ) as text_file:
    print ( f"N,sigma,rho\n{N},{sigma},{rho}", file=text_file )

# Printing the final output of the chosen values and calculated values
print(output)

#pm.plotting ( omega_star, mlp ) # Plotting the approximation function
#pm.plotTVerror()



'''
Grid Search CV

#example of calling five fold cv method
###fivefoldCV ( [25, 0.0001, 1.2] ,cv_data)
'''

'''
#This is just an example of one configuration but for Grid Search CV hyper parameter optimization we defined more configs
#Moreover in this run.py file we just take choice of N, ro and sigma based on those results; we don't execute them again 

cv_data = np.concatenate ( (data_train, data_validate) )
neurons_count = [28]#[37,41,50]#[5,10, 15,20]  # [2,7,11,17,23,27,33]
rho_values = [0.00001]
sigma_vals = [1]#1.2, 1,1.

all_poss_conf = []
for i in neurons_count:
    for r in rho_values:
        for s in sigma_vals:
            all_poss_conf.append ( (i, r, s) )

res_df = pd.DataFrame ()
for conf in tqdm ( all_poss_conf ):
    res_df = res_df.append ( pm.fivefoldCV ( conf ,cv_data) )
res_df.to_csv('cv_results_MLP.csv',index=False)
'''



#initial_function_value on whole training data (data used for CV)
init_fun=mlp.reg_tr_error ( omega.flatten (), [X_train, Y_train, N] )
#initial_training_error on whole training data (data used for CV)
init_tr_err = pm.mse ( Y_train, mlp.predict ( omega.flatten (), X_train ) )
#final norm of the gradient
final_jac_norm = pm.second_norm_jac ( res['jac'].T )

#print('initial function value',init_fun)
#print('initial training error',init_tr_err)
#print('final norm of the gradient',final_jac_norm)






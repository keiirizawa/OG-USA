#%%
import os
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle

#%%
# Data: http://www.computer-services.e.u-tokyo.ac.jp/p/cemano/research/DP/documents/coe-f-213.pdf?fbclid=IwAR2Q5JFemo8hNWF-rD7dZshJbz7a7CWFGiPJeUIHwa8iwlYqBmvfgeaZn8Q
ages = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62, 65])
ability = np.array([0.646, 0.843, 0.999, 1.107, 1.165, 1.218, 1.233, 1.127, 0.820, 0.727])
plt.plot(ages, ability)
plt.xlabel('Ages')
plt.ylabel('Ability')
abil_fun = si.splrep(ages, ability)
new_bins = np.linspace(21, 65, 80)
output = si.splev(new_bins, abil_fun)
plt.plot(new_bins, output)
plt.xlabel('Ages')
plt.ylabel('Ability')
print(si.splev(65, abil_fun))

#%%
def data_func(vals): 
    ages = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62, 65])
    ability = np.array([0.646, 0.843, 0.999, 1.107, 1.165, 1.218, 1.233, 1.127, 0.820, 0.727])
    plt.plot(ages, ability)
    plt.xlabel('Ages')
    plt.ylabel('Ability')
    abil_fun = si.splrep(ages, ability)
    new_bins = np.linspace(21, 65, 80)
    output = si.splev(vals, abil_fun)
    return output

data_func([64,65])
#%%
def data_moments(vals):
    return data_func(vals)

def model_moments(x, a, b, c, d):
    y = - a * np.arctan(b * x + c) + d 
    return y

def err_vec(params, *args):
    a, b, c, d = params
    vals, = args
    data_mms = data_moments(vals)
    model_mms = model_moments(vals, a, b, c, d)

    sumsq = ((model_mms - data_mms) ** 2).sum()
    return sumsq


#%%
# optimization Problem
a = 0.5
b = 0.5
c = 0.5
d = 0.5
params_init = np.array([a,b,c,d])
gmm_args = np.array([62, 63, 64, 65])

results_GMM = opt.minimize(err_vec, params_init, args = gmm_args, method = 'L-BFGS-B')
print(results_GMM)
a,b,c,d = results_GMM.x

ages = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62, 65])
ability = np.array([0.646, 0.843, 0.999, 1.107, 1.165, 1.218, 1.233, 1.127, 0.820, 0.727])
plt.plot(ages, ability)
plt.xlabel('Ages')
plt.ylabel('Ability')
abil_fun = si.splrep(ages, ability)
new_bins = np.linspace(21, 65, 80)
output = si.splev(new_bins, abil_fun)
plt.plot(new_bins, output)
plt.xlabel('Ages')
plt.ylabel('Ability')
#%%
ages = np.linspace(20, 100, 81)
ages_full = np.linspace(20, 100, 81)
ages_beg = np.linspace(20, 65, 46)
print(ages_beg)
ages_end = np.linspace(66, 100, 35)
print(ages_end)
ages_full[:46] = si.splev(ages_beg, abil_fun)
ages_full[46:] = model_moments(ages_end, a,b,c,d)
plt.plot(ages, ages_full)
matrix = []
for i in ages_full:
    line = [4 * i] * 7
    matrix.append(line)
matrix = pd.DataFrame(matrix)
print(matrix)
pickle.dump(matrix, open('run_examples/ability.pkl', 'wb'))

#%%
ages = np.linspace(20, 100, 81)
data = model_moments(ages, a,b,c,d)
plt.plot(ages, data)
#%%
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def chi_n_func(s, a0, a1, a4, a6):#, a3, a4, a5):
    chi_n = a0 + a1 * s + a2 * s ** 2\
            + a3 * s ** 3 + a4 * s ** 4\
    return chi_n


def crit(params, *args):
    a0, a1, a2, a3 = params
    data_moments, ages = args
    model_moments = chi_n_func(ages, a0, a1, a2, a3)#, a3, a4, a5)
  
    err_vec = (model_mms - data_mms) / data_mms

    return err_vec


def criterion(params, *args):
    
    phi0, phi1, phi2 = params
    I_array, I_array_2, W = args
    
    err = err_vec(I_array, I_array_2, phi0, phi1, phi2, simple = False)
    crit_val = err.T @ W @ err
    return crit_val

ages = np.linspace(20, 100, S)
params_init = np.array([a0, a1, a4, a6])#a2, a3, a4])#, a3, a4, a5])
data_moments = np.array([38.12000874, 33.22762421, 25.3484224, 26.67954008, 24.41097278, 23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263, 21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942, 20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951, 20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057, 19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606, 19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741, 21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398, 22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164, 29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997, 38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562, 37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405, 37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256, 43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704, 53.86896121, 53.90029708, 61.83586775, 64.87563699, 66.91207845, 68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.6230815])
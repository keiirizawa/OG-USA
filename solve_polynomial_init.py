#%%
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def chi_n_func(s, a0, a3, a8, b8):
    chi_n = (s/20 - a0/2) ** (-1) + a3 * s ** 3 + a8 * (s + b8) ** 8
    return chi_n

# def chi_n_func(s, a0, a1, a4, a6):#, a3, a4, a5):
#     #chi_n = a0 + a1 * s + a2 * s ** 2\
#     #        + a3 * s ** 3 + a4 * s ** 4\
#     #        + a5 * s ** 5 + a6 * s ** 6
#     chi_n = a0 + a1 * s + a4 * s ** 3\
#             + a6 * s ** 8#\
#             #+ a3 * s ** 3 + a4 * s ** 4
#     return chi_n


def crit(params, *args):
    a0, a3, a8, b8 = params#a0, a1, a4, a6 = params#, a3, a4, a5 = params
    data_moments, ages = args
    model_moments = chi_n_func(ages, a0, a3, a8, b8)#, a3, a4, a5)

    distance = ((data_moments - model_moments) ** 2).sum()
    print('DATA and MODEL DISTANCE: ', distance)
    return distance
#%%
a0 = 1.95000000e+00
a3 = 8.07672128e-05
a8 = -2.60943136e-13
b8 = -5.50000000e+01
#a0 = 5.38312524e+01
#a1 = -1.55746248e+00
#a2 = 1.77689237e-02
#a3 = -8.04751667e-06
#a4 = 5.65432019e-08
#a4 = 0
#a5 = 0
#a6 = 2.48169999e-10

params_init = np.array([a0, a3, a8, b8])#a2, a3, a4])#, a3, a4, a5])
data_moments = np.array([38.12000874, 33.22762421, 25.3484224, 26.67954008, 24.41097278, 23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263, 21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942, 20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951, 20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057, 19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606, 19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741, 21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398, 22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164, 29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997, 38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562, 37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405, 37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256, 43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704, 53.86896121, 53.90029708, 61.83586775, 64.87563699, 66.91207845, 68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.6230815])
S = 80
ages = np.linspace(20, 100, S)

est_output = opt.minimize(crit, params_init,\
                            args=(data_moments, ages),\
                            method="L-BFGS-B", tol=1e-50,\
                            options={'eps': 1e-15, 'maxiter': 1e20})

print(est_output)

results = est_output.x

print("Coefficients: ", results)
a0 = results[0]
a1 = results[1]
#a2 = results[2]
#a3 = results[3]
a4 = results[2]
#a5 = results[5]
a6 = results[3]

model_moments = chi_n_func(ages, a0, a1, a4, a6)#, a3, a4, a5)
#%%
plt.plot(ages, data_moments)
plt.xlabel("Age")
plt.ylabel("Chi_n")
plt.title("Data moments")
plt.show()

plt.plot(ages, model_moments)
plt.xlabel("Age")
plt.ylabel("Chi_n")
plt.title("Model moments")
plt.show()
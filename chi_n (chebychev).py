#%%
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

S = 80
ages = np.linspace(20, 100, S)
ages = np.linspace(20, 60, 40)

#### BASICALLY, MANUALLY CHANGE THESE VALUES TO MAKE DISUTILITY OF LABOR FOR HIGHER AGES HIGHER!!!!
chi_n_vals = np.array([38.12000874, 33.22762421, 25.3484224, 26.67954008, 24.41097278, \
                         23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263, \
                         21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942, \
                         20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951, \
                         20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057, \
                         19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606, \
                         19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741, \
                         21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398, \
                         22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164, \
                         29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997, \
                         38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562, \
                         37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405, \
                         37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256, \
                         43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704, \
                         53.86896121, 53.90029708, 61.83586775, 64.87563699, 70.91207845, \
                         75.07449767, 80.27919965, 85.57195873, 90.95045988, 95.6230815])

#%%
#### Chebyshev function
def func(x, a0, a1, a2, a3, a4):
    func = np.polynomial.chebyshev.chebval(x, [a0, a1, a2, a3, a4])
    return func

#%%
### Finds the best coefficient to fit degree for Chebyshev function to the chi_n_vals:
a0, a1, a2, a3, a4 = np.polynomial.chebyshev.chebfit(ages, chi_n_vals, 4)
data = func(ages, a0, a1, a2, a3, a4)
#data = func(ages, 170, -2.35122641e+01, 4.27581467e-01, -3.40808933e-03, 1.00404321e-05)

plt.xlabel('Ages')
plt.ylabel('Chi_n')
plt.title('Chi_n values')
plt.plot(ages, data, color = 'r', label = r'Estimated')
plt.legend(loc='upper right')
plt.plot(ages, chi_n_vals, color = 'b', label = r'Data')
plt.legend(loc='upper right')

#%%
labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164, 166, 164])

labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709, 0.212, 0.212])

employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968, 0.978, 0.978])

labor_hours_adj = labor_hours * labor_part_rate * employ_rate
    # get fraction of time endowment worked (assume time
    # endowment is 24 hours minus required time to sleep 6.5 hours)
labor_moments = labor_hours_adj * 12 / (365 * 17.5)
labor_moments[9] = 0.1
labor_moments[10] = 0.1

#%%
### MODIFY THE model_moments to see the Labor Supply Graphs
model_moments = np.array([0.2259028129867931, 0.21295422296198854, 0.22059442365687051, 0.22740392749112828, 0.23383671063046393, 0.2362033936361526, 0.23317386766416834, 0.2253931205453907, 0.21104539204176087, 0.19079652009071224, 0.1467245679348507])
labels = np.linspace(20, 70, 11)
labels[-1] = 85
plt.xlabel('Age')
plt.ylabel('Labor Supply as Percent of Total Time Endowment')
plt.title('Labor Suppy vs. Age')
plt.scatter(labels, labor_moments, color = 'r', label = r'Data Moments')
plt.legend(loc='upper right')
plt.plot(labels, model_moments, color = 'b', label = r'Model Moments')
plt.legend(loc='upper right')

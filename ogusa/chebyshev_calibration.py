#%%
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def func(x, a0, a1, a2, a3, a4):
    func = np.polynomial.chebyshev.chebval(x, [a0, a1, a2, a3, a4])
    return func

### Gives the parameters of chebyshev that gives best fit to the given data
#data_moments = np.array([38.12000874, 33.22762421, 25.3484224, 26.67954008, 24.41097278, 23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263, 21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942, 20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951, 20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057, 19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606, 19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741, 21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398, 22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164, 29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997, 38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562, 37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405, 37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256, 43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704, 53.86896121, 53.90029708, 61.83586775, 64.87563699, 66.91207845, 68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.6230815])
#np.polynomial.chebyshev.chebfit(ages, data_moments, 4)

 
#%%
# Chi_n Graph
# Parameter Guesses 
a0 = 1.15807470e+03 
a1 = -1.05805189e+02  
a2 = 1.92411660e+00 
a3 = -1.53364020e-02
a4 = 4.51819445e-05

ages_beg = np.linspace(20, 65, 46)
data_beg = func(ages_beg, a0, a1,a2,a3,a4)
ages_end = np.linspace(65, 100, 36)
data_end = (data_beg[-1] - data_beg[-2]) * (ages_end - 65) + data_beg[-1]
data = np.linspace(20, 100, 81)
ages = np.linspace(20, 100, 81)
data[:46] = data_beg
data[45:] = data_end
plt.xlabel('Age')
plt.ylabel(r'$\chi_n$')
plt.title(r'$\chi_n$ Calibration')
plt.plot(ages, data, color = 'r', label = r'Estimated')
plt.savefig("chi_n.png")


#%%
# Labor Data Moments
labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164, 166, 164])
labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709, 0.212, 0.212])
employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968, 0.978, 0.978])
labor_hours_adj = labor_hours * labor_part_rate * employ_rate
    # get fraction of time endowment worked (assume time
    # endowment is 24 hours minus required time to sleep 6.5 hours)
data_moments = labor_hours_adj * 12 / (365 * 17.5)

#%% 
# Labor Moments
model_moments = np.array([0.21092244724143888, 0.23663473960483364, 0.2512248460552426, 0.2554279329457508, 0.25988610741069157, 0.2674734351216116, 0.2756735811864669, 0.26881862807130374, 0.23334421444663225, 0.18809948673028212, 0.12602132474609165])

labels = np.linspace(20, 70, 11)
labels[-1] = 85

plt.xlabel('Age')
plt.ylabel('Labor Supply as Percent of Total Time Endowment')
plt.title('Labor Suppy vs. Age')
plt.scatter(labels, data_moments, color = 'r', label = r'Data Moments')
plt.legend(loc='upper right')
plt.scatter(labels, model_moments, color = 'b', label = r'Model Moments')
plt.legend(loc='upper right')
plt.savefig("labor_moments.png")


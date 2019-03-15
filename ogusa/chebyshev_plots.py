#%%
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
def func(x, a0, a1, a2, a3, a4):
    func = np.polynomial.chebyshev.chebval(x, [a0, a1, a2, a3, a4])
    return func

#%%
a0 = 1.15807470e+03 
a1 = -1.05805189e+02  
a2 = 1.92411660e+00 
a3 = -1.53364020e-02
a4 = 4.51819445e-05
ages_beg = np.linspace(20, 65, 46)
print('beg', ages_beg)
data_beg = func(ages_beg, a0, a1,a2,a3,a4)
ages_end = np.linspace(65, 100, 36)
print('end', ages_end)
data_end = (data_beg[-1] - data_beg[-2]) * (ages_end - 65) + data_beg[-1]
data = np.linspace(20, 100, 81)
ages = np.linspace(20, 100, 81)
print('data',data)
data[:46] = data_beg
data[45:] = data_end
plt.xlabel('Age')
plt.ylabel('Chi_n')
plt.title(r'$\Chi_n$')
plt.plot(ages, data, color = 'r', label = r'Estimated')
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

#%%
### Effective Tax Function
#Source: https://www.jetro.go.jp/en/invest/setting_up/section3/page7.html
def calc_income_tax(income_x, income_y, year):
    total_income = income_x + income_y
    deducted_x = income_x - find_tax_deduction(income_x, year)
    taxable_income = deducted_x + income_y
    tax_cost = find_tax_cost(taxable_income)
    effective_tax_rate = tax_cost / total_income
    if 2013 <= year <= 2037:
        #Withholding Tax
        effective_tax_rate *= 1.021
    return max(effective_tax_rate, -0.15)
    

def find_tax_cost(income):
    if income <= 1950000:
        return income * 0.05
    elif 1950000 < income <= 3300000:
        return (income - 1950000) * 0.1 + find_tax_cost(1950000)
    elif 3300000 < income <= 6950000:
        return (income - 3300000) * 0.2 + find_tax_cost(3300000)
    elif 6950000 < income <= 9000000:
        return (income - 6950000) * 0.23 + find_tax_cost(6950000)
    elif 9000000 < income <= 18000000:
        return (income - 9000000) * 0.33 + find_tax_cost(9000000)
    elif 18000000 < income <= 40000000:
        return (income - 18000000) * 0.33 + find_tax_cost(18000000)
    elif 40000000 < income:
        return (income - 40000000) * 0.33 + find_tax_cost(40000000)
    
def find_tax_deduction(income, year):
    if year < 2020:
        if income <= 1625000:
            return 650000
        elif 1625000 < income <= 1800000:
            return income * 0.4
        elif 1800000 < income <= 3600000:
            return income * 0.3 + 180000
        elif 3600000 < income <= 6600000:
            return income * 0.2 + 540000
        elif 6600000 < income <= 10000000:
            return income * 0.1 + 1200000
        elif 10000000 < income:
            return 2200000
    else:
        if income <= 1625000:
            return 550000
        elif 1625000 < income <= 1800000:
            return income * 0.4 - 100000
        elif 1800000 < income <= 3600000:
            return income * 0.3 + 80000
        elif 3600000 < income <= 6600000:
            return income * 0.2 + 440000
        elif 6600000 < income <= 10000000:
            return income * 0.1 + 1100000
        elif 10000000 < income:
            return 1950000

calc_income_tax(5000000, 0, 2015)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
incomes = np.linspace(1, 40000000, 10)
vals = pd.Series(incomes).apply(calc_income_tax, args=[0, 2015])
plt.plot(incomes, vals)
plt.xlabel("Income")
plt.ylabel("Effect tax rate")
plt.title("ETR Over Income")
plt.show()

#%%
### GS Tax Function
# URL: https://www.jstor.org/stable/pdf/41789070.pdf
def tax_func(I, phi0, phi1, phi2):
    #txrates = ((phi0 * (I - ((I ** -phi1) + phi2) ** (-1 / phi1))) / I)
    txrates = phi0 - phi0 * (phi1 * I ** phi2 + 1)**(-1 / phi2)
    return txrates

#%%
def model_moments(I_array, phi0, phi1, phi2):
    return tax_func(I_array, phi0, phi1, phi2)

def data_moments(I_array):
    dms = []
    for i in I_array:
        dms.append(calc_income_tax(i, 0, 2018))
    return np.array(dms)

def err_vec(I_array, I_array_2, phi0, phi1, phi2, simple):
    
    data_mms = data_moments(I_array)
    model_mms = model_moments(I_array_2, phi0, phi1, phi2)
    
    if simple:
        err_vec = model_mms - data_mms
    else:
        err_vec = (model_mms - data_mms) / data_mms
    
    return err_vec

def criterion(params, *args):
    
    phi0, phi1, phi2 = params
    I_array, I_array_2, W = args
    
    err = err_vec(I_array, I_array_2, phi0, phi1, phi2, simple = False)
    crit_val = err.T @ W @ err
    return crit_val

#%%
### Optimization Problem: 
# Initial guess of parameters
phi0 = 0.479
phi1 = 0.022
phi2 = 0.817
params_init = np.array([phi0, phi1, phi2])

# Weighting matrix 
W_hat = np.eye(10)

# Arguments
I_array = np.linspace(1, 40000000, 10)
I_array_2 = I_array * 10 ** (-6)
gmm_args = (I_array, I_array_2, W_hat)

# Optimization
results_GMM = opt.minimize(criterion, params_init, args = (gmm_args), method = 'L-BFGS-B')

print(results_GMM)
phi0_GMM, phi1_GMM, phi2_GMM = results_GMM.x
print(phi0_GMM, phi1_GMM, phi2_GMM)


#%%
### Plots
I = np.linspace(1,20,40)

tax_rate = tax_func(I, phi0_GMM, phi1_GMM, phi2_GMM)
plt.xlabel('Income (Million yen)')
plt.ylim(0, 0.4)
plt.ylabel('Tax Rate')
plt.title('Incomve Vs. Tax Rate (GS)')
plt.plot(I, tax_rate, color = 'r', label = r'Estimated Tax Rates')
plt.legend(loc='upper right')

I_new = I * 10 ** 6
tax_rate_data = []
for i in I_new:
    tax_rate_data.append(calc_income_tax(i, 0, 2018))
tax_rate_data = np.array(tax_rate_data)
plt.plot(I, tax_rate_data, label = r'Calculated Tax Rates')
plt.legend(loc='upper right')
plt.show()
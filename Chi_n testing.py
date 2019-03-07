import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from ogusa import wealth
from ogusa import labor
from ogusa import SS
from ogusa import utils

from taxcalc import Calculator
from dask.distributed import Client
from ogusa.parameters import Specifications
from ogusa.utils import REFORM_DIR, BASELINE_DIR

output_base = BASELINE_DIR
client = Client(processes=False)
num_workers = 1  # multiprocessing.cpu_count()

alpha_T = np.zeros(50)
alpha_T[0:2] = 0.09
alpha_T[2:10] = 0.09 + 0.01
alpha_T[10:40] = 0.09 - 0.01
alpha_T[40:] = 0.09
alpha_G = np.zeros(7)
alpha_G[0:3] = 0.05 - 0.01
alpha_G[3:6] = 0.05 - 0.005
alpha_G[6:] = 0.05
small_open = False

user_params = {'frisch': 0.41, 'start_year': 2018,
                   'tau_b': [(0.21 * 0.55) * (0.017 / 0.055), (0.21 * 0.55) * (0.017 / 0.055)],
                   'debt_ratio_ss': 1.0, 'alpha_T': alpha_T.tolist(),
                   'alpha_G': alpha_G.tolist(), 'small_open': small_open}

kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': False, 'baseline': True,
              'user_params': user_params, 'guid': '_example',
              'run_micro': False, 'data': 'cps', 'client': client,
              'num_workers': num_workers}

p = Specifications(run_micro=False, output_base=output_base,
                          baseline_dir=BASELINE_DIR, test=False,
                          time_path=False, baseline=True,
                          #reform=reform,
                           guid='_example', data='cps',
                          client=client, num_workers=num_workers)

def chi_n_func(s, a0, a1, a2, a3, a4):
    chi_n = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4
    return chi_n


a0 = 1
a1 = 0
a2 = 0
a3 = 0
a4 = 0

params_init = np.array([a0, a1, a2, a3, a4])

labor_data = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164, 166, 164])
labor_moments = labor_data * 12 / (365 * 17.5)
data_moments = np.array(list(labor_moments.flatten()))
ages = np.linspace(20, 100, p.S)
p.chi_n = chi_n_func(ages, a0, a1, a2, a3, a4)
### had to add this to make it work:
ss_output = SS.run_SS(p, client)
model_moments = calc_moments(ss_output, p.omega_SS, p.lambdas, p.S, p.J)

print(labor_moments)
print(model_moments)

from __future__ import print_function
'''
------------------------------------------------------------------------
Created 12/28/2016

Fiscal policy functions for unbalanced budgeting. In particular, some
functions require time-path calculation.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def D_G_path(r_gov, dg_fixed_values, Gbaseline, p):
    '''
    Calculate the time paths of debt and government spending
    '''
    Y, REVENUE, T_H, D0, G0 = dg_fixed_values

    D = np.zeros(p.T + 1)
    D[0] = D0
    if not p.baseline_spending:
        G = p.ALPHA_G * Y[:p.T]
        G[0] = G0
    else:
        G = Gbaseline
    growth = (1 + p.g_n_vector) * np.exp(p.g_y)

    t = 1
    while t < p.T-1:
        D[t] = ((1 / growth[t]) * ((1 + r_gov[t - 1]) * D[t - 1] +
                                   G[t - 1] + T_H[t - 1] -
                                   REVENUE[t - 1]))
        if (t >= p.tG1) and (t < p.tG2):
            G[t] = (growth[t + 1] * (p.rho_G * p.debt_ratio_ss * Y[t] +
                                     (1 - p.rho_G) * D[t]) -
                    (1 + r_gov[t]) * D[t] + REVENUE[t] - T_H[t])
        elif t >= p.tG2:
            G[t] = (growth[t + 1] * (p.debt_ratio_ss * Y[t]) -
                    (1 + r_gov[t]) * D[t] + REVENUE[t] - T_H[t])
        t += 1

    # in final period, growth rate has stabilized, so we can replace
    # growth[t+1] with growth[t]
    t = p.T - 1
    D[t] = ((1 / growth[t]) * ((1 + r_gov[t - 1]) * D[t - 1] + G[t - 1]
                               + T_H[t - 1] - REVENUE[t - 1]))
    G[t] = (growth[t] * (p.debt_ratio_ss * Y[t]) - (1 + r_gov[t]) * D[t]
            + REVENUE[t] - T_H[t])
    D[t + 1] = ((1 / growth[t + 1]) * ((1 + r_gov[t]) * D[t] + G[t] +
                                       T_H[t] - REVENUE[t]))
    D_ratio_max = np.amax(D[:p.T] / Y[:p.T])
    print('Maximum debt ratio: ', D_ratio_max)

    return D, G

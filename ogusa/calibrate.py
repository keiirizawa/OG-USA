from __future__ import print_function

'''
This module should be organized as follows:

Main function:
chi_estimate() = returns chi_n, chi_b
    - calls:
        wealth.get_wealth_data() - returns data moments on wealth distribution
        labor.labor_data_moments() - returns data moments on labor supply
        minstat() - returns min of statistical objective function
            model_moments() - returns model moments
                SS.run_SS() - return SS distributions

'''

'''
------------------------------------------------------------------------
Last updated: 7/27/2016

Uses a simulated method of moments to calibrate the chi_n adn chi_b
parameters of OG-USA.

This py-file calls the following other file(s):
    wealth.get_wealth_data()
    labor.labor_data_moments()
    SS.run_SS

This py-file creates the following other file(s): None
------------------------------------------------------------------------
'''

import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from . import wealth
from . import labor
from . import SS
from . import utils


def chi_n_func(s, a0, a1, a2, a3, a4):
    chi_n = a0 + a1 * s + a2 * s ** 2 + a3 * s ** 3 + a4 * s ** 4
    return chi_n


def chi_estimate(p, client=None):
    '''
    --------------------------------------------------------------------
    This function calls others to obtain the data momements and then
    runs the simulated method of moments estimation by calling the
    minimization routine.

    INPUTS:
    income_tax_parameters = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    ss_parameters         = length 21 tuple, (J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                            g_n_ss, tau_payroll, retire, mean_income_data,\
                            h_wealth, p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params      = [2,] vector, vector with max iterations and tolerance
                             for SS solution
    chi_guesses           = [J+S,] vector, initial guesses of chi_b and chi_n stacked together
    baseline_dir          = string, path where baseline results located


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    wealth.compute_wealth_moments()
    labor.labor_data_moments()
    minstat()

    OBJECTS CREATED WITHIN FUNCTION:
    wealth_moments     = [J+2,] array, wealth moments from data
    labor_moments      = [S,] array, labor moments from data
    data_moments       = [J+2+S,] array, wealth and labor moments stacked
    bnds               = [S+J,] array, bounds for parameter estimates
    chi_guesses_flat   =  [J+S,] vector, initial guesses of chi_b and chi_n stacked
    min_arg            = length 6 tuple, variables needed for minimizer
    est_output         = dictionary, output from minimizer
    chi_params         = [J+S,] vector, parameters estimates for chi_b and chi_n stacked
    objective_func_min = scalar, minimum of statistical objective function


    OUTPUT:
    ./baseline_dir/Calibration/chi_estimation.pkl


    RETURNS: chi_params
    --------------------------------------------------------------------
    '''

    baseline_dir="./OUTPUT"
    #chi_b_guess = np.ones(80)

    a0 = 5.38312524e+01
    a1 = -1.55746248e+00
    a2 = 1.77689237e-02
    a3 = -8.04751667e-06
    a4 = 5.65432019e-08

    params_init = np.array([a0, a1, a2, a3, a4])

    # Generate labor data moments
    labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164, 166, 164])

    labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709, 0.212, 0.212])

    employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968, 0.978, 0.978])

    labor_hours_adj = labor_hours * labor_part_rate * employ_rate

    # get fraction of time endowment worked (assume time
    # endowment is 24 hours minus required time to sleep 6.5 hours)
    labor_moments = labor_hours_adj * 12 / (365 * 17.5)

    # combine moments
    data_moments = np.array(list(labor_moments.flatten()))

    # weighting matrix
    W = np.identity(p.J+2+p.S)
    W = np.identity(11)

    est_output = opt.minimize(minstat, params_init, args=(p, client, data_moments, W), method="L-BFGS-B", tol=1e-15)


    ss_output = SS.run_SS(p)
    return ss_output



def minstat(params, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared differences
    between the model and data moments.

    INPUTS:
    chi_guesses = [J+S,] vector, initial guesses of chi_b and chi_n stacked together
    arg         = length 6 tuple, variables needed for minimizer


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    SS.run_SS()
    calc_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    ss_output     = dictionary, variables from SS of model
    model_moments = [J+2+S,] array, moments from the model solution
    distance      = scalar, weighted, squared deviation between data and model moments

    RETURNS: distance
    --------------------------------------------------------------------
    '''

    a0, a1, a2, a3, a4 = params
    p, client, data_moments, W = args
    ages = np.linspace(20, 100, p.S)
    chi_n = chi_n_func(ages, a0, a1, a2, a3, a4)

    p.chi_n = chi_n
    print(chi_n)
    print("-----------------------------------------------------")
    ss_output = SS.run_SS(p, client)

    model_moments = calc_moments(ss_output, p.omega_SS, p.lambdas, p.S, p.J)
    print(model_moments)
    print("-----------------------------------------------------")

    # distance with levels
    distance = np.dot(np.dot((np.array(model_moments) - np.array(data_moments)).T,W),
                   np.array(model_moments) - np.array(data_moments))
    #distance = ((np.array(model_moments) - np.array(data_moments))**2).sum()
    print('DATA and MODEL DISTANCE: ', distance)

    # # distance with percentage diffs
    # distance = (((model_moments - data_moments)/data_moments)**2).sum()

    return distance


def calc_moments(ss_output, omega_SS, lambdas, S, J):
    '''
    --------------------------------------------------------------------
    This function calculates moments from the SS output that correspond
    to the data moments used for estimation.

    INPUTS:
    ss_output = dictionary, variables from SS of model
    omega_SS  = [S,] array, SS population distribution over age
    lambdas   = [J,] array, proportion of population of each ability type
    S         = integer, number of ages
    J         = integer, number of ability types

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    the_inequalizer()

    OBJECTS CREATED WITHIN FUNCTION:
    model_wealth_moments = [J+2,] array, wealth moments from the model
    model_labor_moments  = [S,] array, labor moments from the model
    model_moments        = [J+2+S,] array, wealth and data moments from the model solution
    distance             = scalar, weighted, squared deviation between data and model moments

    RETURNS: distance

    RETURNS: model_moments
    --------------------------------------------------------------------
    '''
    # unpack relevant SS variables
    n = ss_output['nssmat']

    # labor moments
    model_labor_moments = (n.reshape(S, J) * lambdas.reshape(1, J)).sum(axis=1)

    ### we have ages 20-100 so lets find binds based on population weights
    # converting to match our data moments
    model_labor_moments = pd.DataFrame(model_labor_moments * omega_SS)
    #print("--------------------------------------------------------------------")
    #print("Original:")
    #model_labor_moments = model_labor_moments.mean(axis=0)
    model_labor_moments.rename({0: 'labor_weighted'}, axis=1, inplace=True)
    #print(model_labor_moments)
    #print("--------------------------------------------------------------------")

    ages = np.linspace(20, 100, S)
    age_bins = np.linspace(20, 75, 12)
    age_bins[11] = 101
    labels = np.linspace(20, 70, 11)
    model_labor_moments['pop_dist'] = omega_SS
    model_labor_moments['age_bins'] = pd.cut(ages, age_bins, right=False, include_lowest=True, labels=labels)
    #print("--------------------------------------------------------------------")
    #print("Updated")
    #print(model_labor_moments)
    #print("--------------------------------------------------------------------")
    #print("Shape:")
    #print(model_labor_moments.shape)
    #print("--------------------------------------------------------------------")
    weighted_labor_moments = model_labor_moments.groupby('age_bins')['labor_weighted'].sum() /\
                                model_labor_moments.groupby('age_bins')['pop_dist'].sum()


    # combine moments
    model_moments = list(weighted_labor_moments)

    return model_moments


def the_inequalizer(dist, pop_weights, ability_weights, factor, S, J):
    '''
    --------------------------------------------------------------------
    Generates three measures of inequality.

    Inputs:
        dist            = [S,J] array, distribution of endogenous variables over age and lifetime income group
        pop_weights     = [S,] vector, fraction of population by each age
        ability_weights = [J,] vector, fraction of population for each lifetime income group
        factor          = scalar, factor relating model units to dollars
        S               = integer, number of economically active periods in lifetime
        J               = integer, number of ability types

    Functions called: None

    Objects in function:
        weights           = [S,J] array, fraction of population for each age and lifetime income group
        flattened_dist    = [S*J,] vector, vectorized dist
        flattened_weights = [S*J,] vector, vectorized weights
        sort_dist         = [S*J,] vector, ascending order vector of dist
        loc_90th          = integer, index of 90th percentile
        loc_10th          = integer, index of 10th percentile
        loc_99th          = integer, index of 99th percentile

    Returns: measure of inequality
    --------------------------------------------------------------------
    '''

    weights = np.tile(pop_weights.reshape(S, 1), (1, J)) * \
    ability_weights.reshape(1, J)
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.cumsum(sort_weights)

    # gini
    p = cum_weights/cum_weights.sum()
    nu = np.cumsum(sort_dist*sort_weights)
    nu = nu/nu[-1]
    gini_coeff = (nu[1:]*p[:-1]).sum() - (nu[:-1] * p[1:]).sum()


    # variance
    ln_dist = np.log(sort_dist*factor) # not scale invariant
    weight_mean = (ln_dist*sort_weights).sum()/sort_weights.sum()
    var_ln_dist = ((sort_weights*((ln_dist-weight_mean)**2)).sum())*(1./(sort_weights.sum()))


    # 90/10 ratio
    loc_90th = np.argmin(np.abs(cum_weights - .9))
    loc_10th = np.argmin(np.abs(cum_weights - .1))
    ratio_90_10 = sort_dist[loc_90th] / sort_dist[loc_10th]

    # top 10% share
    top_10_share= (sort_dist[loc_90th:] * sort_weights[loc_90th:]
           ).sum() / (sort_dist * sort_weights).sum()

    # top 1% share
    loc_99th = np.argmin(np.abs(cum_weights - .99))
    top_1_share = (sort_dist[loc_99th:] * sort_weights[loc_99th:]
           ).sum() / (sort_dist * sort_weights).sum()

    # calculate percentile shares (percentiles based on lambdas input)
    dist_weight = (sort_weights*sort_dist)
    total_dist_weight = dist_weight.sum()
    cumsum = np.cumsum(sort_weights)
    dist_sum = np.zeros((J,))
    cum_weights = ability_weights.cumsum()
    for i in range(J):
        cutoff = sort_weights.sum() / (1./cum_weights[i])
        dist_sum[i] = ((dist_weight[cumsum < cutoff].sum())/total_dist_weight)


    dist_share = np.zeros((J,))
    dist_share[0] = dist_sum[0]
    dist_share[1:] = dist_sum[1:]-dist_sum[0:-1]

    return np.append([dist_share], [gini_coeff,var_ln_dist])

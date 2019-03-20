'''
------------------------------------------------------------------------
Calculates steady state of OG-USA model with S age cohorts and J
ability types.

This py-file calls the following other file(s):
            tax.py
            household.py
            firm.py
            utils.py
            OUTPUT/SS/ss_vars.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/SS/ss_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import pandas as pd
import scipy.optimize as opt
import pickle
from dask import compute, delayed
import dask.multiprocessing
from ogusa import tax, household, firm, utils
from ogusa import aggregates as aggr
import os
import warnings


'''
Set minimizer tolerance
'''
MINIMIZER_TOL = 1e-13

'''
Set flag for enforcement of solution check
'''
ENFORCE_SOLUTION_CHECKS = True

'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''


def euler_equation_solver(guesses, *args):
    '''
    --------------------------------------------------------------------
    Finds the euler errors for certain b and n, one ability type at a time.
    --------------------------------------------------------------------

    INPUTS:
    guesses = [2S,] vector, initial guesses for b and n
    r = scalar, real interest rate
    w = scalar, real wage rate
    T_H = scalar, lump sum transfer
    factor = scalar, scaling factor converting model units to dollars
    j = integer, ability group
    params = length 21 tuple, list of parameters
    chi_b = [J,] vector, chi^b_j, the utility weight on bequests
    chi_n = [S,] vector, chi^n_s utility weight on labor supply
    tau_bq = scalar, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega_SS = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type
    tax_params = length 5 tuple, (tax_func_type, analytical_mtrs,
                 etr_params, mtrx_params, mtry_params)
    tax_func_type   = string, type of tax function used
    analytical_mtrs = boolean, =True if use analytical_mtrs, =False if
                       use estimated MTRs
    etr_params      = [S,BW,#tax params] array, parameters for effective
                      tax rate function
    mtrx_params     = [S,BW,#tax params] array, parameters for marginal
                      tax rate on labor income function
    mtry_params     = [S,BW,#tax params] array, parameters for marginal
                      tax rate on capital income function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    aggr.get_BQ()
    tax.replacement_rate_vals()
    household.FOC_savings()
    household.FOC_labor()
    tax.total_taxes()
    household.get_cons()

    OBJECTS CREATED WITHIN FUNCTION:
    b_guess = [S,] vector, initial guess at household savings
    n_guess = [S,] vector, initial guess at household labor supply
    b_s = [S,] vector, wealth enter period with
    b_splus1 = [S,] vector, household savings
    BQ = scalar, aggregate bequests to lifetime income group
    theta = scalar, replacement rate for social security benenfits
    error1 = [S,] vector, errors from FOC for savings
    error2 = [S,] vector, errors from FOC for labor supply
    tax1 = [S,] vector, total income taxes paid
    cons = [S,] vector, household consumption

    RETURNS: 2Sx1 list of euler errors

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    (r, w, bq, T_H, factor, j, p) = args

    b_guess = np.array(guesses[:p.S])
    n_guess = np.array(guesses[p.S:])
    b_s = np.array([0] + list(b_guess[:-1]))
    b_splus1 = b_guess
<<<<<<< HEAD
    # if np.isnan(b_s).any() or (b_s < 0).any()\
    #     or (n_guess < 0).any() or np.isnan(r).any()\
    #     or np.isnan(w).any() or (w < 0).any():
    #     b_s = pd.DataFrame(b_s)
    #     n_guess = pd.DataFrame(n_guess)
    #     w = np.array(w)
    #     error_sum = b_s[b_s < 0].sum().sum() +\
    #         n_guess[n_guess < 0].sum().sum() +\
    #         w[w < 0].sum().sum()
    #     error1 = [1e9 * error_sum] * 80
    #     error2 = [1e9 * error_sum] * 80
    #     return np.hstack((error1, error2))
=======
    if np.isnan(b_s).any() or (b_s < 0).any()\
        or (n_guess < 0).any() or np.isnan(r).any()\
        or np.isnan(w).any():
        error1 = [1e10]*80
        error2 = [1e10]*80
        return np.hstack((error1, error2))
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3

    theta = tax.replacement_rate_vals(n_guess, w, factor, j, p)

    error1 = household.FOC_savings(r, w, b_s, b_splus1, n_guess, bq,
                                   factor, T_H, theta, p.e[:, j], p.rho,
                                   p.tau_c[-1, :, j],
                                   p.etr_params[-1, :, :],
                                   p.mtry_params[-1, :, :], None, j, p,
                                   'SS')
    error2 = household.FOC_labor(r, w, b_s, b_splus1, n_guess, bq,
                                 factor, T_H, theta, p.chi_n, p.e[:, j],
                                 p.tau_c[-1, :, j],
                                 p.etr_params[-1, :, :],
                                 p.mtrx_params[-1, :, :], None, j, p,
                                 'SS')

    # Put in constraints for consumption and savings.
    # According to the euler equations, they can be negative.  When
    # Chi_b is large, they will be.  This prevents that from happening.
    # I'm not sure if the constraints are needed for labor.
    # But we might as well put them in for now.
    mask1 = n_guess < 0
    mask2 = n_guess > p.ltilde
    mask3 = b_guess <= 0
    mask4 = np.isnan(n_guess)
    mask5 = np.isnan(b_guess)
    error2[mask1] = 1e14
    error2[mask2] = 1e14
    error1[mask3] = 1e14
    error1[mask5] = 1e14
    error2[mask4] = 1e14
    taxes = tax.total_taxes(r, w, b_s, n_guess, bq, factor, T_H, theta,
                            None, j, False, 'SS', p.e[:, j],
                            p.etr_params[-1, :, :], p)
    cons = household.get_cons(r, w, b_s, b_splus1, n_guess, bq, taxes,
                              p.e[:, j], p.tau_c[-1, :, j], p)
    mask6 = cons < 0
    error1[mask6] = 1e14

    return np.hstack((error1, error2))


def inner_loop(outer_loop_vars, p, client):
    '''
    This function solves for the inner loop of
    the SS.  That is, given the guesses of the
    outer loop variables (r, w, Y, factor)
    this function solves the households'
    problems in the SS.

    Inputs:
        r          = [T,] vector, interest rate
        w          = [T,] vector, wage rate
        b          = [T,S,J] array, wealth holdings
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        Y        = [T,] vector, lump sum transfer amount(s)


    Functions called:
        euler_equation_solver()
        aggr.get_K()
        aggr.get_L()
        firm.get_Y()
        firm.get_r()
        firm.get_w()
        aggr.get_BQ()
        tax.replacement_rate_vals()
        aggr.revenue()

    Objects in function:


    Returns: euler_errors, bssmat, nssmat, new_r, new_w
             new_T_H, new_factor, new_BQ

    '''
    # unpack variables to pass to function
    if p.budget_balance:
        bssmat, nssmat, r, BQ, T_H, factor = outer_loop_vars
    else:
        bssmat, nssmat, r, BQ, Y, T_H, factor = outer_loop_vars

    euler_errors = np.zeros((2 * p.S, p.J))

    w = firm.get_w_from_r(r, p, 'SS')
    bq = household.get_bq(BQ, None, p, 'SS')

    lazy_values = []
    for j in range(p.J):
        guesses = np.append(bssmat[:, j], nssmat[:, j])
        euler_params = (r, w, bq[:, j], T_H, factor, j, p)
        lazy_values.append(delayed(opt.fsolve)(euler_equation_solver,
                                               guesses * .9,
                                               args=euler_params,
                                               xtol=MINIMIZER_TOL,
                                               full_output=True))
    results = compute(*lazy_values, scheduler=dask.multiprocessing.get,
                      num_workers=p.num_workers)

    # for j, result in results.items():
    for j, result in enumerate(results):
        [solutions, infodict, ier, message] = result
        euler_errors[:, j] = infodict['fvec']
        bssmat[:, j] = solutions[:p.S]
        nssmat[:, j] = solutions[p.S:]
<<<<<<< HEAD
    if (nssmat < 0).any():
        print('NSSMAT < 0 AT SS 233')
        #stop
=======
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3

    L = aggr.get_L(nssmat, p, 'SS')
    if not p.small_open:
        B = aggr.get_K(bssmat, p, 'SS', False)
        if p.budget_balance:
            K = B
            print('1st statement being used')
        else:
            K = B - p.debt_ratio_ss * Y
            print('2nd statement being used')
    else:
        K = firm.get_K(L, r, p, 'SS')
        print('3rd statement being used')
    new_Y = firm.get_Y(K, L, p, 'SS')
    print('NEW Y IS BEING USED')
    if p.budget_balance:
        Y = new_Y
    if not p.small_open:
        new_r = firm.get_r(Y, K, p, 'SS')
    else:
        new_r = p.hh_r[-1]
    new_w = firm.get_w_from_r(new_r, p, 'SS')
    if np.isnan(new_w).any():
        return np.nan
    print('inner factor prices: ', new_r, new_w)

    b_s = np.array(list(np.zeros(p.J).reshape(1, p.J)) +
                   list(bssmat[:-1, :]))
    average_income_model = ((new_r * b_s + new_w * p.e * nssmat) *
                            p.omega_SS.reshape(p.S, 1) *
                            p.lambdas.reshape(1, p.J)).sum()
    if p.baseline:
        new_factor = p.mean_income_data / average_income_model   ###### mean income data, !!!!!!!!! parameters.py
    else:
        new_factor = factor
    new_BQ = aggr.get_BQ(new_r, bssmat, None, p, 'SS', False)
    new_bq = household.get_bq(new_BQ, None, p, 'SS')
    theta = tax.replacement_rate_vals(nssmat, new_w, new_factor, None, p)

    if p.budget_balance:
        etr_params_3D = np.tile(np.reshape(
            p.etr_params[-1, :, :], (p.S, 1, p.etr_params.shape[2])),
                                (1, p.J, 1))
        taxss = tax.total_taxes(new_r, new_w, b_s, nssmat, new_bq,
                                factor, T_H, theta, None, None, False,
                                'SS', p.e, etr_params_3D, p)
        cssmat = household.get_cons(new_r, new_w, b_s, bssmat,
                                    nssmat, new_bq, taxss,
                                    p.e, p.tau_c[-1, :, :], p)
        new_T_H, _, _, _, _, _, _ = aggr.revenue(
            new_r, new_w, b_s, nssmat, new_bq, cssmat, new_Y, L, K,
            factor, theta, etr_params_3D, p, 'SS')
    elif p.baseline_spending:
        new_T_H = T_H
    else:
        new_T_H = p.alpha_T[-1] * new_Y

    return euler_errors, bssmat, nssmat, new_r, new_w, \
        new_T_H, new_Y, new_factor, new_BQ, average_income_model


def SS_solver(bmat, nmat, r, BQ, T_H, factor, Y, p, client,
              fsolve_flag=False):
    '''
    --------------------------------------------------------------------
    Solves for the steady state distribution of capital, labor, as well
    as w, r, T_H and the scaling factor, using a bisection method
    similar to TPI.
    --------------------------------------------------------------------

    INPUTS:
    b_guess_init = [S,J] array, initial guesses for savings
    n_guess_init = [S,J] array, initial guesses for labor supply
    wguess = scalar, initial guess for SS real wage rate
    rguess = scalar, initial guess for SS real interest rate
    T_Hguess = scalar, initial guess for lump sum transfer
    factorguess = scalar, initial guess for scaling factor to dollars
    chi_b = [J,] vector, chi^b_j, the utility weight on bequests
    chi_n = [S,] vector, chi^n_s utility weight on labor supply
    params = length X tuple, list of parameters
    iterative_params = length X tuple, list of parameters that determine
                       the convergence of the while loop
    tau_bq = [J,] vector, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    euler_equation_solver()
    aggr.get_K()
    aggr.get_L()
    firm.get_Y()
    firm.get_r()
    firm.get_w()
    aggr.get_BQ()
    tax.replacement_rate_vals()
    aggr.revenue()
    utils.convex_combo()
    utils.pct_diff_func()


    OBJECTS CREATED WITHIN FUNCTION:
    b_guess = [S,] vector, initial guess at household savings
    n_guess = [S,] vector, initial guess at household labor supply
    b_s = [S,] vector, wealth enter period with
    b_splus1 = [S,] vector, household savings
    b_splus2 = [S,] vector, household savings one period ahead
    BQ = scalar, aggregate bequests to lifetime income group
    theta = scalar, replacement rate for social security benenfits
    error1 = [S,] vector, errors from FOC for savings
    error2 = [S,] vector, errors from FOC for labor supply
    tax1 = [S,] vector, total income taxes paid
    cons = [S,] vector, household consumption

    OBJECTS CREATED WITHIN FUNCTION - SMALL OPEN ONLY
    Bss = scalar, aggregate household wealth in the steady state
    BIss = scalar, aggregate household net investment in the steady state

    RETURNS: solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # Rename the inputs

    if not p.budget_balance:
        if not p.baseline_spending:
            Y = T_H / p.alpha_T[-1]
    if p.small_open:
        r = p.ss_hh_r[-1]

    dist = 10
    iteration = 0
    dist_vec = np.zeros(p.maxiter)
    maxiter_ss = p.maxiter
    nu_ss = p.nu

    if fsolve_flag:
        maxiter_ss = 1

    while (dist > p.mindist_SS) and (iteration < maxiter_ss):
        # Solve for the steady state levels of b and n, given w, r,
        # Y and factor
        if p.budget_balance:
            outer_loop_vars = (bmat, nmat, r, BQ, T_H, factor)
        else:
            outer_loop_vars = (bmat, nmat, r, BQ, Y, T_H, factor)
        with open("output.txt", "a") as text_file:
            text_file.write('\nOuter_loop vars (SS_solver):\n' + str(outer_loop_vars))
        if np.isnan(bmat).any():
            print('bmat has nan')
            print(bmat)
<<<<<<< HEAD
            #stop
        try:
            (euler_errors, new_bmat, new_nmat, new_r, new_w, new_T_H, new_Y,
                new_factor, new_BQ, average_income_model) =\
                inner_loop(outer_loop_vars, p, client)
        except:
            print('Iteration has failed. No steady state found.')
=======
            stop
        (euler_errors, new_bmat, new_nmat, new_r, new_w, new_T_H, new_Y,
         new_factor, new_BQ, average_income_model) =\
            inner_loop(outer_loop_vars, p, client)
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3

        r = utils.convex_combo(new_r, r, nu_ss)
        factor = utils.convex_combo(new_factor, factor, nu_ss)
        BQ = utils.convex_combo(new_BQ, BQ, nu_ss)
        # bmat = utils.convex_combo(new_bmat, bmat, nu_ss)
        # nmat = utils.convex_combo(new_nmat, nmat, nu_ss)
        if p.budget_balance:
            T_H = utils.convex_combo(new_T_H, T_H, nu_ss)
            dist = np.array([utils.pct_diff_func(new_r, r)] +
                            list(utils.pct_diff_func(new_BQ, BQ)) +
                            [utils.pct_diff_func(new_T_H, T_H)] +
                            [utils.pct_diff_func(new_factor, factor)]).max()
        else:
            Y = utils.convex_combo(new_Y, Y, nu_ss)
            if Y != 0:
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                list(utils.pct_diff_func(new_BQ, BQ)) +
                                [utils.pct_diff_func(new_Y, Y)] +
                                [utils.pct_diff_func(new_factor,
                                                     factor)]).max()
            else:
                # If Y is zero (if there is no output), a percent difference
                # will throw NaN's, so we use an absoluate difference
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                list(utils.pct_diff_func(new_BQ, BQ)) +
                                [abs(new_Y - Y)] +
                                [utils.pct_diff_func(new_factor,
                                                     factor)]).max()
        dist_vec[iteration] = dist
        # Similar to TPI: if the distance between iterations increases, then
        # decrease the value of nu to prevent cycling
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration - 1] > 0:
                nu_ss /= 2.0
                print('New value of nu:', nu_ss)
        iteration += 1
        print('Iteration: %02d' % iteration, ' Distance: ', dist)

    '''
    ------------------------------------------------------------------------
        Generate the SS values of variables, including euler errors
    ------------------------------------------------------------------------
    '''
    bssmat_s = np.append(np.zeros((1, p.J)), bmat[:-1, :], axis=0)
    bssmat_splus1 = bmat
    nssmat = nmat
<<<<<<< HEAD
    if (nssmat < 0).any():
        print('NEGATIVE N AT SS 440')
=======

>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3

    rss = r
    wss = new_w
    BQss = new_BQ
    factor_ss = factor
    T_Hss = T_H
    bqssmat = household.get_bq(BQss, None, p, 'SS')

    Lss = aggr.get_L(nssmat, p, 'SS')
    if not p.small_open:
        Bss = aggr.get_K(bssmat_splus1, p, 'SS', False)
        if p.budget_balance:
            debt_ss = 0.0
        else:
            debt_ss = p.debt_ratio_ss * Y
        Kss = Bss - debt_ss
        Iss = aggr.get_I(bssmat_splus1, Kss, Kss, p, 'SS')
    else:
        # Compute capital (K) and wealth (B) separately
        Kss = firm.get_K(Lss, p.ss_firm_r[-1], p, 'SS')
        InvestmentPlaceholder = np.zeros(bssmat_splus1.shape)
        Iss = aggr.get_I(InvestmentPlaceholder, Kss, Kss, p, 'SS')
        Bss = aggr.get_K(bssmat_splus1, p, 'SS', False)
        BIss = aggr.get_I(bssmat_splus1, Bss, Bss, p, 'BI_SS')

        if p.budget_balance:
            debt_ss = 0.0
        else:
            debt_ss = p.debt_ratio_ss * Y

    Yss = firm.get_Y(Kss, Lss, p, 'SS')
    print('YSS IS BEING USED')
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, None, p)

    # Compute effective and marginal tax rates for all agents
    etr_params_3D = np.tile(np.reshape(
        p.etr_params[-1, :, :], (p.S, 1, p.etr_params.shape[2])), (1, p.J, 1))
    mtrx_params_3D = np.tile(np.reshape(
        p.mtrx_params[-1, :, :], (p.S, 1, p.mtrx_params.shape[2])),
                             (1, p.J, 1))
    mtry_params_3D = np.tile(np.reshape(
        p.mtry_params[-1, :, :], (p.S, 1, p.mtry_params.shape[2])),
                             (1, p.J, 1))
    mtry_ss = tax.MTR_income(rss, wss, bssmat_s, nssmat, factor, True,
                             p.e, etr_params_3D, mtry_params_3D, p)
    mtrx_ss = tax.MTR_income(rss, wss, bssmat_s, nssmat, factor, False,
                             p.e, etr_params_3D, mtrx_params_3D, p)
    etr_ss = tax.ETR_income(rss, wss, bssmat_s, nssmat, factor, p.e,
                            etr_params_3D, p)

    taxss = tax.total_taxes(rss, wss, bssmat_s, nssmat, bqssmat, factor_ss,
                            T_Hss, theta, None, None, False, 'SS',
                            p.e, etr_params_3D, p)
    cssmat = household.get_cons(rss, wss, bssmat_s, bssmat_splus1,
                                nssmat, bqssmat, taxss,
                                p.e, p.tau_c[-1, :, :], p)
    Css = aggr.get_C(cssmat, p, 'SS')

    (total_revenue_ss, T_Iss, T_Pss, T_BQss, T_Wss, T_Css,
     business_revenue) =\
        aggr.revenue(rss, wss, bssmat_s, nssmat, bqssmat, cssmat, Yss,
                     Lss, Kss, factor, theta, etr_params_3D, p, 'SS')
    r_gov_ss = rss
    debt_service_ss = r_gov_ss * p.debt_ratio_ss * Yss
    new_borrowing = p.debt_ratio_ss * Yss * ((1 + p.g_n_ss) *
                                             np.exp(p.g_y) - 1)
    # government spends such that it expands its debt at the same rate as GDP
    if p.budget_balance:
        Gss = 0.0
    else:
        Gss = total_revenue_ss + new_borrowing - (T_Hss + debt_service_ss)
        print('G components = ', new_borrowing, T_Hss, debt_service_ss)

    # Compute total investment (not just domestic)
    Iss_total = p.delta * Kss

    # solve resource constraint
    if p.small_open:
        # include term for current account
        resource_constraint = (Yss + new_borrowing - (Css + BIss + Gss)
                               + (p.ss_hh_r[-1] * Bss -
                                  (p.delta + p.ss_firm_r[-1]) *
                                  Kss - debt_service_ss))
        print('Yss= ', Yss, '\n', 'Css= ', Css, '\n', 'Bss = ', Bss,
              '\n', 'BIss = ', BIss, '\n', 'Kss = ', Kss, '\n', 'Iss = ',
              Iss, '\n', 'Lss = ', Lss, '\n', 'T_H = ', T_H, '\n',
              'Gss= ', Gss)
        print('D/Y:', debt_ss / Yss, 'T/Y:', T_Hss / Yss, 'G/Y:',
              Gss / Yss, 'Rev/Y:', total_revenue_ss / Yss,
              'Int payments to GDP:', (rss * debt_ss) / Yss)
        print('resource constraint: ', resource_constraint)
    else:
        resource_constraint = Yss - (Css + Iss + Gss)
        print('Yss= ', Yss, '\n', 'Gss= ', Gss, '\n', 'Css= ', Css, '\n',
              'Kss = ', Kss, '\n', 'Iss = ', Iss, '\n', 'Lss = ', Lss,
              '\n', 'Debt service = ', debt_service_ss)
        print('D/Y:', debt_ss / Yss, 'T/Y:', T_Hss / Yss, 'G/Y:',
              Gss / Yss, 'Rev/Y:', total_revenue_ss / Yss, 'business rev/Y: ',
              business_revenue / Yss, 'Int payments to GDP:',
              (rss * debt_ss) / Yss)
        print('Check SS budget: ', Gss - (np.exp(p.g_y) *
                                          (1 + p.g_n_ss) - 1 - rss) *
              debt_ss - total_revenue_ss + T_Hss)
        print('resource constraint: ', resource_constraint)

    if Gss < 0:
        print('Steady state government spending is negative to satisfy'
              + ' budget')

    if ENFORCE_SOLUTION_CHECKS and (np.absolute(resource_constraint) >
                                    p.mindist_SS):
        print('Resource Constraint Difference:', resource_constraint)
        err = 'Steady state aggregate resource constraint not satisfied'
        # raise RuntimeError(err)
        print(err)
        euler_savings = euler_errors[:p.S, :]
        euler_labor_leisure = euler_errors[p.S:, :]
        output = {'Kss': Kss, 'Bss': Bss, 'Lss': Lss, 'Css': Css, 'Iss': Iss,
              'Iss_total': Iss_total, 'nssmat': nssmat, 'Yss': Yss,
              'Dss': debt_ss, 'wss': wss, 'rss': rss, 'theta': theta,
              'BQss': BQss, 'factor_ss': factor_ss, 'bssmat_s': bssmat_s,
              'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
              'bqssmat': bqssmat, 'T_Hss': T_Hss, 'Gss': Gss,
              'total_revenue_ss': total_revenue_ss,
              'business_revenue': business_revenue,
              'IITpayroll_revenue': T_Iss,
              'T_Pss': T_Pss, 'T_BQss': T_BQss, 'T_Wss': T_Wss,
              'T_Css': T_Css, 'euler_savings': euler_savings,
              'euler_labor_leisure': euler_labor_leisure,
              'resource_constraint_error': resource_constraint,
              'etr_ss': etr_ss, 'mtrx_ss': mtrx_ss, 'mtry_ss': mtry_ss}
        #print('Exception results:')
        #print(output)
        with open("output.txt", "a") as text_file:
            text_file.write('\nException results:\n' + str(output))
        return output
    # check constraints
    household.constraint_checker_SS(bssmat_splus1, nssmat, cssmat, p.ltilde)

    euler_savings = euler_errors[:p.S, :]
    euler_labor_leisure = euler_errors[p.S:, :]
    print('Maximum error in labor FOC = ',
          np.absolute(euler_labor_leisure).max())
    print('Maximum error in savings FOC = ',
          np.absolute(euler_savings).max())

    '''
    ------------------------------------------------------------------------
        Return dictionary of SS results
    ------------------------------------------------------------------------
    '''
    output = {'Kss': Kss, 'Bss': Bss, 'Lss': Lss, 'Css': Css, 'Iss': Iss,
              'Iss_total': Iss_total, 'nssmat': nssmat, 'Yss': Yss,
              'Dss': debt_ss, 'wss': wss, 'rss': rss, 'theta': theta,
              'BQss': BQss, 'factor_ss': factor_ss, 'bssmat_s': bssmat_s,
              'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
              'bqssmat': bqssmat, 'T_Hss': T_Hss, 'Gss': Gss,
              'total_revenue_ss': total_revenue_ss,
              'business_revenue': business_revenue,
              'IITpayroll_revenue': T_Iss,
              'T_Pss': T_Pss, 'T_BQss': T_BQss, 'T_Wss': T_Wss,
              'T_Css': T_Css, 'euler_savings': euler_savings,
              'euler_labor_leisure': euler_labor_leisure,
              'resource_constraint_error': resource_constraint,
              'etr_ss': etr_ss, 'mtrx_ss': mtrx_ss, 'mtry_ss': mtry_ss}

    with open("output.txt", "a") as text_file:
            text_file.write('\nSuccessful results:\n' + str(output))
    return output


def SS_fsolve(guesses, *args):
    '''
    Solves for the steady state distribution of capital, labor, as well as
    w, r, T_H and the scaling factor, using a root finder.
    Inputs:
        b_guess_init = guesses for b (SxJ array)
        n_guess_init = guesses for n (SxJ array)
        wguess = guess for wage rate (scalar)
        rguess = guess for rental rate (scalar)
        T_Hguess = guess for lump sum tax (scalar)
        factorguess = guess for scaling factor to dollars (scalar)
        chi_n = chi^n_s (Sx1 array)
        chi_b = chi^b_j (Jx1 array)
        params = list of parameters (list)
        iterative_params = list of parameters that determine the convergence
                           of the while loop (list)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        omega_SS = population weights (Sx1 array)
        e = ability levels (SxJ array)
    Outputs:
        solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)
    '''
    (bssmat, nssmat, T_Hss, factor_ss, p, client) = args
    if (nssmat > 0.3).any():
        errors = [1e9] * 10
        return errors
    # Rename the inputs
    r = guesses[0]
    if p.baseline:
        BQ = guesses[1:-2]
        T_H = guesses[-2]
        factor = guesses[-1]
    else:
        BQ = guesses[1:-1]
        if p.baseline_spending:
            Y = guesses[-1]
        else:
            T_H = guesses[-1]
    # Create tuples of outler loop vars
    if p.baseline:
        if p.budget_balance:
            outer_loop_vars = (bssmat, nssmat, r, BQ, T_H, factor)
        else:
            Y = T_H / p.alpha_T[-1]
            outer_loop_vars = (bssmat, nssmat, r, BQ, Y, T_H, factor)
    else:
        if p.baseline_spending:
            outer_loop_vars = (bssmat, nssmat, r, BQ, Y, T_Hss, factor_ss)
        else:
            if p.budget_balance:
                outer_loop_vars = (bssmat, nssmat, r, BQ, T_H, factor_ss)
            else:
                Y = T_H / p.alpha_T[-1]
                outer_loop_vars = (bssmat, nssmat, r, BQ, Y, T_H, factor_ss)

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    print('------------------------------------')
    print("Made it to fsolve 0!")
    print('------------------------------------')
    if np.isnan(bssmat).any():
        print('bssmat has nan')
        print(bssmat)
        stop
    with open("output.txt", "a") as text_file:
            text_file.write('\nOuter_loop vars (SS_fsolve):\n' + str(outer_loop_vars))
<<<<<<< HEAD
    try:
        (euler_errors, bssmat, nssmat, new_r, new_w, new_T_H, new_Y,
            new_factor, new_BQ, average_income_model) =\
            inner_loop(outer_loop_vars, p, client)
    except:
        return [1e9] * 10
=======
    (euler_errors, bssmat, nssmat, new_r, new_w, new_T_H, new_Y,
     new_factor, new_BQ, average_income_model) =\
        inner_loop(outer_loop_vars, p, client)
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3
    print('------------------------------------')
    print("Made it to fsolve 1!")
    print('------------------------------------')
    #Lss = aggr.get_L(nssmat, p, 'SS')
    nssmat = {'nssmat': nssmat}
    from ogusa import calibrate
    nssmat = calibrate.calc_moments(nssmat, p.omega_SS, p.lambdas, p.S, p.J)
    print(nssmat)

    # Create list of errors in general equilibrium variables
    error1 = new_r - r
    error2 = new_BQ - BQ
    if p.baseline:
        if p.budget_balance:
            error3 = new_T_H - T_H
        else:
            error3 = new_Y - Y
        error4 = new_factor / 1000000 - factor / 1000000
        print('GE loop errors = ', error1, error2, error3, error4)
        # Check and punish violations of the factor
        if factor <= 0:
            error4 = 1e9
        errors = [error1] + list(error2) + [error3, error4]
    else:
        if p.baseline_spending:
            error3 = new_Y - Y
        else:
            if p.budget_balance:
                error3 = new_T_H - T_H
            else:
                error3 = new_Y - Y
        errors = [error1] + list(error2) + [error3]
        print('GE loop errors = ', error1, error2, error3)
    # Check and punish violations of the bounds on the interest rate
    if r + p.delta <= 0:
        errors[0] = 1e9

    print('------------------------------------')
    print("Made it to fsolve 2!")
    print('------------------------------------')
    return errors


def run_SS(p, client=None):
    '''
    --------------------------------------------------------------------
    Solve for SS of OG-USA.
    --------------------------------------------------------------------

    INPUTS:
    p = Specifications class with parameterization of model
    income_tax_parameters = length 5 tuple, (tax_func_type,
                            analytical_mtrs, etr_params,
                            mtrx_params, mtry_params)
    ss_parameters = length 21 tuple, (J, S, T, BW, beta, sigma, alpha,
                    gamma, epsilon, Z, delta, ltilde, nu, g_y, g_n_ss,
                    tau_payroll, retire, mean_income_data, h_wealth,
                    p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params  = [2,] vector, vector with max iterations and
                        tolerance for SS solution
    baseline = boolean, =True if run is for baseline tax policy
    calibrate_model = boolean, =True if run calibration of chi parameters
    output_dir = string, path to save output from current model run
    baseline_dir = string, path where baseline results located


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    SS_fsolve()
    SS_fsolve_reform()
    SS_solver

    OBJECTS CREATED WITHIN FUNCTION:
    chi_params = [J+S,] vector, chi_b and chi_n stacked together
    b_guess = [S,J] array, initial guess at savings
    n_guess = [S,J] array, initial guess at labor supply
    wguess = scalar, initial guess at SS real wage rate
    rguess = scalar, initial guess at SS real interest rate
    T_Hguess = scalar, initial guess at SS lump sum transfers
    factorguess = scalar, initial guess at SS factor adjustment (to
                  scale model units to dollars)

    output


    RETURNS: output

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # For initial guesses of w, r, T_H, and factor, we use values that
    # are close to some steady state values.
<<<<<<< HEAD
    print('--------------------------------------------------------------------')
    print('CHI N VALUES:')
    print(p.chi_n)
    print('--------------------------------------------------------------------')
    if p.baseline:
        b_guess = np.ones((p.S, p.J)) * 0.5   ### hard coded
        n_guess = np.ones((p.S, p.J)) * .8 * p.ltilde  ### hard coded
        rguess = 0.09 # initially 0.09
        T_Hguess = 0.12
        factorguess = 7.7 # convert it to yen and account that unit of income in Millions 
=======
    if p.baseline:
        #b_guess = np.ones((p.S, p.J)) * 0.07   ### hard coded
        b_guess = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [5.50629592e-02, 3.95758163e-02, 1.93254326e-02, 4.00533164e-02,
        4.39348194e-02, 3.53941816e-02, 1.04932979e-01],
       [9.38263523e-02, 6.48348714e-02, 3.68390137e-02, 6.40723795e-02,
        7.00032020e-02, 7.38459707e-02, 1.64673872e-01],
       [1.25964670e-01, 8.41581135e-02, 5.73906565e-02, 8.21144720e-02,
        8.97195960e-02, 1.14762768e-01, 2.10225631e-01],
       [1.54799159e-01, 1.00360235e-01, 8.07463108e-02, 9.74617795e-02,
        1.06757260e-01, 1.56240302e-01, 2.50670537e-01],
       [1.81877190e-01, 1.14507799e-01, 1.07004357e-01, 1.10906982e-01,
        1.21859084e-01, 1.98469219e-01, 2.86913534e-01],
       [2.09130189e-01, 1.28888750e-01, 1.36330839e-01, 1.25679513e-01,
        1.38744194e-01, 2.41894289e-01, 3.28548453e-01],
       [2.37167666e-01, 1.44046586e-01, 1.68773603e-01, 1.42102526e-01,
        1.57662833e-01, 2.84692480e-01, 3.75328880e-01],
       [2.66598874e-01, 1.60955557e-01, 2.04162315e-01, 1.61598720e-01,
        1.80236816e-01, 3.26113418e-01, 4.31182700e-01],
       [2.97538692e-01, 1.79936040e-01, 2.42248239e-01, 1.84519163e-01,
        2.06818914e-01, 3.67753041e-01, 4.96518396e-01],
       [3.30094785e-01, 2.01618962e-01, 2.82507223e-01, 2.12155365e-01,
        2.38923636e-01, 4.00830233e-01, 5.75334503e-01],
       [3.64079855e-01, 2.26133507e-01, 3.24798277e-01, 2.44951695e-01,
        2.77056438e-01, 4.13403569e-01, 6.68962025e-01],
       [3.98945247e-01, 2.52689822e-01, 3.68667765e-01, 2.81392308e-01,
        3.19368349e-01, 4.18873000e-01, 7.72106737e-01],
       [4.34600206e-01, 2.81623365e-01, 4.13191610e-01, 3.22446825e-01,
        3.66994826e-01, 4.13074713e-01, 8.88520954e-01],
       [4.71080387e-01, 3.13506854e-01, 4.57038203e-01, 3.69584602e-01,
        4.21658240e-01, 3.71066104e-01, 1.02397379e+00],
       [5.08210511e-01, 3.48163112e-01, 4.99343952e-01, 4.22554029e-01,
        4.83015288e-01, 3.46275101e-01, 1.17830892e+00],
       [5.45974287e-01, 3.85658110e-01, 5.39077525e-01, 4.81536899e-01,
        5.51239171e-01, 3.27155700e-01, 1.35307313e+00],
       [5.84588473e-01, 4.26524969e-01, 5.74601905e-01, 5.47588887e-01,
        6.27541722e-01, 3.09935505e-01, 1.55292759e+00],
       [6.24171998e-01, 4.70887122e-01, 6.04929545e-01, 6.20798484e-01,
        7.11981180e-01, 2.72526714e-01, 1.77896387e+00],
       [6.65122076e-01, 5.19499040e-01, 6.27929745e-01, 7.02426120e-01,
        8.06011219e-01, 3.26150425e-01, 2.03615357e+00],
       [7.07510788e-01, 5.72186233e-01, 6.44197423e-01, 7.91820560e-01,
        9.08813766e-01, 2.77397929e-01, 2.32249308e+00],
       [7.51738888e-01, 6.29563412e-01, 6.53133178e-01, 8.89794128e-01,
        1.02131587e+00, 2.24256463e-01, 2.64067825e+00],
       [7.98039622e-01, 6.91795811e-01, 6.55145182e-01, 9.96262429e-01,
        1.14338169e+00, 1.69134822e-01, 2.98992191e+00],
       [8.46735633e-01, 7.59251899e-01, 6.45404460e-01, 1.11147953e+00,
        1.27528343e+00, 1.55790057e-01, 3.37013464e+00],
       [8.98245738e-01, 8.32515311e-01, 6.17050514e-01, 1.23604012e+00,
        1.41770494e+00, 1.33944529e-01, 3.78181635e+00],
       [9.52960563e-01, 9.12101310e-01, 5.75847661e-01, 1.37037931e+00,
        1.57114712e+00, 1.14961181e-01, 4.22440424e+00],
       [1.01111545e+00, 9.98196266e-01, 5.06893751e-01, 1.51435069e+00,
        1.73542566e+00, 1.04991584e-01, 4.69504307e+00],
       [1.07304075e+00, 1.09120490e+00, 4.75245288e-01, 1.66815894e+00,
        1.91078703e+00, 9.80920238e-02, 5.19156963e+00],
       [1.13905036e+00, 1.19150894e+00, 4.00635031e-01, 1.83195049e+00,
        2.09742236e+00, 7.77130160e-02, 5.71128515e+00],
       [1.20855042e+00, 1.29774489e+00, 3.33822962e-01, 2.00305342e+00,
        2.29215891e+00, 9.22938738e-02, 6.24297274e+00],
       [1.28245912e+00, 1.41147042e+00, 2.58979115e-01, 2.18350168e+00,
        2.49749949e+00, 7.74281513e-02, 6.78947368e+00],
       [1.36008528e+00, 1.53124424e+00, 2.03797437e-01, 2.37053263e+00,
        2.71020118e+00, 7.35701118e-02, 7.33948191e+00],
       [1.44093570e+00, 1.65603768e+00, 1.70478189e-01, 2.56208751e+00,
        2.92789787e+00, 6.94672208e-02, 7.88431302e+00],
       [1.52413899e+00, 1.78418991e+00, 1.46999449e-01, 2.75515994e+00,
        3.14710837e+00, 6.53297237e-02, 8.41306224e+00],
       [1.60861728e+00, 1.91371491e+00, 1.31542455e-01, 2.94630199e+00,
        3.36384414e+00, 6.10883421e-02, 8.91411493e+00],
       [1.69308427e+00, 2.04229639e+00, 1.19455969e-01, 3.13163248e+00,
        3.57360992e+00, 5.68406967e-02, 9.37498334e+00],
       [1.77594412e+00, 2.16713110e+00, 1.09968816e-01, 3.30659771e+00,
        3.77109448e+00, 5.58221431e-02, 9.78116889e+00],
       [1.85552811e+00, 2.28532544e+00, 1.02273234e-01, 3.46660163e+00,
        3.95091489e+00, 5.39889179e-02, 1.01176179e+01],
       [1.93011774e+00, 2.39394686e+00, 9.80665303e-02, 3.60706890e+00,
        4.10766727e+00, 5.40539421e-02, 1.03684027e+01],
       [1.99842567e+00, 2.49078247e+00, 9.39854224e-02, 3.72465542e+00,
        4.23739715e+00, 5.33914401e-02, 1.05206132e+01],
       [2.05931492e+00, 2.57391191e+00, 8.91216326e-02, 3.81651424e+00,
        4.33668524e+00, 5.19585580e-02, 1.05618445e+01],
       [2.11174425e+00, 2.64163330e+00, 8.54141458e-02, 3.88011640e+00,
        4.40241881e+00, 5.12825595e-02, 1.04795621e+01],
       [2.15508519e+00, 2.69293848e+00, 8.15735597e-02, 3.91402507e+00,
        4.43276467e+00, 5.03824161e-02, 1.02644143e+01],
       [2.18883402e+00, 2.72706067e+00, 7.69864345e-02, 3.91708687e+00,
        4.42617013e+00, 4.95544388e-02, 9.90710114e+00],
       [2.21262022e+00, 2.74346599e+00, 6.73193553e-02, 3.88838738e+00,
        4.38131549e+00, 4.70458660e-02, 9.39790710e+00],
       [2.37673960e+00, 2.98749382e+00, 7.23515539e-02, 4.30835443e+00,
        4.89868250e+00, 4.93015190e-02, 1.13925960e+01],
       [2.53723852e+00, 3.22368870e+00, 7.02172821e-02, 4.71338610e+00,
        5.39727513e+00, 4.80367476e-02, 1.32735928e+01],
       [2.69468879e+00, 3.45321108e+00, 6.83728745e-02, 5.10586204e+00,
        5.88009149e+00, 4.74153177e-02, 1.50566701e+01],
       [2.84972204e+00, 3.67724560e+00, 6.49767996e-02, 5.48816785e+00,
        6.35013846e+00, 4.58401485e-02, 1.67564496e+01],
       [3.00187164e+00, 3.89535596e+00, 6.63022306e-02, 5.85969911e+00,
        6.80669090e+00, 4.65808681e-02, 1.83747378e+01],
       [3.15247215e+00, 4.10963088e+00, 6.64727096e-02, 6.22438170e+00,
        7.25465230e+00, 4.66699927e-02, 1.99303323e+01],
       [3.30216156e+00, 4.32111930e+00, 6.44828008e-02, 6.58420627e+00,
        7.69650182e+00, 4.57338206e-02, 2.14335477e+01],
       [3.45081205e+00, 4.52977751e+00, 6.36058352e-02, 6.93917594e+00,
        8.13226699e+00, 4.54106326e-02, 2.28871316e+01],
       [3.59866475e+00, 4.73605450e+00, 6.26546224e-02, 7.29014315e+00,
        8.56301982e+00, 4.50054736e-02, 2.42965275e+01],
       [3.74582340e+00, 4.94019278e+00, 6.17358568e-02, 7.63756665e+00,
        8.98935130e+00, 4.46097766e-02, 2.56655267e+01],
       [3.89233403e+00, 5.14234203e+00, 6.09396736e-02, 7.98171806e+00,
        9.41162826e+00, 4.42585491e-02, 2.69970436e+01],
       [4.03820788e+00, 5.34259251e+00, 6.07186895e-02, 8.32274461e+00,
        9.83007281e+00, 4.41704468e-02, 2.82933561e+01],
       [4.18378120e+00, 5.54145642e+00, 6.01980152e-02, 8.66151174e+00,
        1.02457758e+01, 4.39334025e-02, 2.95587235e+01],
       [4.32917690e+00, 5.73914677e+00, 5.93974423e-02, 8.99834422e+00,
        1.06591815e+01, 4.35732424e-02, 3.07954348e+01],
       [4.47425156e+00, 5.93551387e+00, 5.89326119e-02, 9.33292593e+00,
        1.10699813e+01, 4.33645977e-02, 3.20036188e+01],
       [4.61907013e+00, 6.13067598e+00, 5.83726992e-02, 9.66539896e+00,
        1.14784272e+01, 4.30915165e-02, 3.31846101e+01],
       [4.76356658e+00, 6.32455355e+00, 5.81138925e-02, 9.99534691e+00,
        1.18794221e+01, 4.29789144e-02, 3.43424084e+01],
       [4.90801914e+00, 6.51750615e+00, 5.77572520e-02, 1.03229492e+01,
        1.22733458e+01, 4.28199650e-02, 3.54770795e+01],
       [5.05247561e+00, 6.70959965e+00, 5.73907210e-02, 1.06479511e+01,
        1.26600756e+01, 4.26577706e-02, 3.65880049e+01],
       [5.19683529e+00, 6.90071283e+00, 5.71324220e-02, 1.09698408e+01,
        1.30392087e+01, 4.25422580e-02, 3.76738231e+01],
       [5.34107295e+00, 7.09082634e+00, 5.69344773e-02, 1.12883295e+01,
        1.34106137e+01, 4.24524723e-02, 3.87338520e+01],
       [5.48520975e+00, 7.27998204e+00, 5.67419660e-02, 1.16032689e+01,
        1.37743333e+01, 4.23651997e-02, 3.97678478e+01],
       [5.62920338e+00, 7.46814049e+00, 5.65584850e-02, 1.19144046e+01,
        1.41303112e+01, 4.22820111e-02, 4.07753087e+01],
       [5.77294825e+00, 7.65518153e+00, 5.63922262e-02, 1.22213735e+01,
        1.44783954e+01, 4.22067441e-02, 4.17554915e+01],
       [5.91629225e+00, 7.84092645e+00, 5.62522698e-02, 1.25237361e+01,
        1.48183794e+01, 4.21429828e-02, 4.27075270e+01],
       [6.05908401e+00, 8.02519845e+00, 5.61285449e-02, 1.28210680e+01,
        1.51501064e+01, 4.20877422e-02, 4.36306960e+01],
       [6.20113823e+00, 8.20777874e+00, 5.60481839e-02, 1.31128815e+01,
        1.54733880e+01, 4.20500377e-02, 4.45242281e+01],
       [6.34246219e+00, 8.38869244e+00, 5.59580118e-02, 1.33990616e+01,
        1.57884812e+01, 4.20099386e-02, 4.53885215e+01],
       [6.48293968e+00, 8.56780817e+00, 5.58890812e-02, 1.36792162e+01,
        1.60953960e+01, 4.19789850e-02, 4.62233729e+01],
       [6.62257208e+00, 8.74514220e+00, 5.58335226e-02, 1.39531258e+01,
        1.63944011e+01, 4.19487700e-02, 4.70292718e+01],
       [6.76151602e+00, 8.92090346e+00, 5.57254608e-02, 1.42207672e+01,
        1.66860776e+01, 4.19101359e-02, 4.78075181e+01],
       [6.89918133e+00, 9.09436808e+00, 5.56299458e-02, 1.44806043e+01,
        1.69695070e+01, 4.18988198e-02, 4.85557082e+01],
       [7.03618530e+00, 9.26631778e+00, 5.57971655e-02, 1.47331537e+01,
        1.72461783e+01, 4.18786822e-02, 4.92774930e+01],
       [7.17295672e+00, 9.43729178e+00, 5.56218175e-02, 1.49784067e+01,
        1.75171469e+01, 4.18558684e-02, 4.99753970e+01],
       [7.31010569e+00, 9.60804711e+00, 5.55434803e-02, 1.52167569e+01,
        1.77837645e+01, 4.17509027e-02, 5.06526276e+01]])
        #n_guess = np.ones((p.S, p.J)) * .4 * p.ltilde  ### hard coded
        n_guess = np.array([[2.80726420e-01, 2.52287187e-01, 8.17927149e-02, 2.61552329e-01,
        2.71334288e-01, 8.82999451e-02, 2.94885081e-01],
       [2.78504823e-01, 2.53162942e-01, 8.23679603e-02, 2.64038327e-01,
        2.72849806e-01, 9.21134562e-02, 2.92040085e-01],
       [2.86444886e-01, 2.64364067e-01, 8.90467776e-02, 2.77257578e-01,
        2.85379276e-01, 9.62790468e-02, 3.00688353e-01],
       [2.97788261e-01, 2.78791138e-01, 9.54065489e-02, 2.93362550e-01,
        3.00783949e-01, 9.87332842e-02, 3.12370759e-01],
       [3.10536976e-01, 2.94202166e-01, 1.01973306e-01, 3.09695793e-01,
        3.16322032e-01, 1.00679313e-01, 3.24290973e-01],
       [3.24651843e-01, 3.11236858e-01, 1.08470343e-01, 3.27528888e-01,
        3.33398162e-01, 9.81297800e-02, 3.38070749e-01],
       [3.38485385e-01, 3.27718059e-01, 1.14389368e-01, 3.44126866e-01,
        3.49196856e-01, 9.57679050e-02, 3.50818318e-01],
       [3.51598794e-01, 3.43553495e-01, 1.19791339e-01, 3.59823370e-01,
        3.64136703e-01, 9.08054293e-02, 3.63089054e-01],
       [3.63128601e-01, 3.57706842e-01, 1.24262890e-01, 3.73497983e-01,
        3.77093888e-01, 8.35229443e-02, 3.73754336e-01],
       [3.72780282e-01, 3.70155819e-01, 1.27453188e-01, 3.85519074e-01,
        3.88484810e-01, 7.16780058e-02, 3.83306083e-01],
       [3.80136246e-01, 3.80308081e-01, 1.28345648e-01, 3.95243075e-01,
        3.97655149e-01, 5.37364626e-02, 3.91073297e-01],
       [3.84892896e-01, 3.87435004e-01, 1.29006526e-01, 4.01644224e-01,
        4.03546947e-01, 4.57553997e-02, 3.95905391e-01],
       [3.87698662e-01, 3.92752279e-01, 1.28110902e-01, 4.06486747e-01,
        4.07961512e-01, 3.19879449e-02, 3.99702184e-01],
       [3.88977239e-01, 3.96746431e-01, 1.25668111e-01, 4.10316789e-01,
        4.11436287e-01, 3.41317187e-03, 4.03013269e-01],
       [3.88884401e-01, 3.99149756e-01, 1.22050649e-01, 4.12542653e-01,
        4.13347837e-01, 1.89592345e-02, 4.05109400e-01],
       [3.87943665e-01, 4.00549568e-01, 1.17345783e-01, 4.13801059e-01,
        4.14329909e-01, 2.54154212e-02, 4.06620940e-01],
       [3.86711026e-01, 4.01653458e-01, 1.11567359e-01, 4.14876742e-01,
        4.15166642e-01, 2.59625052e-02, 4.08342685e-01],
       [3.85377694e-01, 4.02424253e-01, 1.05073712e-01, 4.15569824e-01,
        4.15644089e-01, 1.34022667e-02, 4.09972073e-01],
       [3.84429390e-01, 4.03574590e-01, 9.78539366e-02, 4.16688658e-01,
        4.16574544e-01, 7.06581086e-02, 4.12332006e-01],
       [3.83737071e-01, 4.04585080e-01, 9.09249726e-02, 4.17512248e-01,
        4.17221315e-01, 6.19343843e-03, 4.14563303e-01],
       [3.83728349e-01, 4.06184853e-01, 8.48912039e-02, 4.18887438e-01,
        4.18439016e-01, 6.78220708e-03, 4.17532834e-01],
       [3.84327485e-01, 4.08133469e-01, 7.89206045e-02, 4.20503519e-01,
        4.19910145e-01, 4.86292143e-03, 4.20852598e-01],
       [3.85657645e-01, 4.10636844e-01, 6.70134134e-02, 4.22589062e-01,
        4.21865150e-01, 2.81436880e-02, 4.24720608e-01],
       [3.87808562e-01, 4.13862501e-01, 4.85050784e-02, 4.25324766e-01,
        4.24487504e-01, 2.39811436e-02, 4.29281347e-01],
       [3.90722588e-01, 4.17741083e-01, 3.95669906e-02, 4.28633478e-01,
        4.27700444e-01, 2.56025006e-02, 4.34408305e-01],
       [3.94196213e-01, 4.21996780e-01, 1.17041637e-02, 4.32229445e-01,
        4.31216148e-01, 2.99360714e-02, 4.39761052e-01],
       [3.98209384e-01, 4.26695182e-01, 5.70011138e-02, 4.36192576e-01,
        4.35120501e-01, 3.11175390e-02, 4.45374591e-01],
       [4.02592436e-01, 4.31684624e-01, 1.43146049e-02, 4.40375221e-01,
        4.39270085e-01, 2.47583878e-02, 4.51047108e-01],
       [4.06343438e-01, 4.35620110e-01, 1.52772941e-02, 4.43429535e-01,
        4.42296023e-01, 3.97560673e-02, 4.55351604e-01],
       [4.10530562e-01, 4.40224444e-01, 1.42502923e-02, 4.47103075e-01,
        4.45999452e-01, 2.53840156e-02, 4.60015957e-01],
       [4.13393252e-01, 4.43178571e-01, 3.36304991e-02, 4.49084621e-01,
        4.48028451e-01, 2.78537969e-02, 4.62627662e-01],
       [4.14734926e-01, 4.44427411e-01, 5.64753823e-02, 4.49338295e-01,
        4.48363857e-01, 2.48878558e-02, 4.63099047e-01],
       [4.13821205e-01, 4.43187898e-01, 6.49650907e-02, 4.47104808e-01,
        4.46244886e-01, 2.06817216e-02, 4.60605507e-01],
       [4.10091984e-01, 4.38900262e-01, 7.17000552e-02, 4.41850516e-01,
        4.41141353e-01, 1.87891391e-02, 4.54565470e-01],
       [4.03070590e-01, 4.31078195e-01, 7.42349918e-02, 4.33118945e-01,
        4.32597662e-01, 1.99976057e-02, 4.44491226e-01],
       [3.92357327e-01, 4.19288666e-01, 7.54712212e-02, 4.20508441e-01,
        4.20204258e-01, 2.64210841e-02, 4.29958443e-01],
       [3.77975543e-01, 4.03562520e-01, 7.56887940e-02, 4.04079286e-01,
        4.04015529e-01, 2.89791627e-02, 4.11047928e-01],
       [3.60208756e-01, 3.84195359e-01, 7.66281106e-02, 3.84149264e-01,
        3.84336343e-01, 3.17798094e-02, 3.88121961e-01],
       [3.39915426e-01, 3.62124479e-01, 7.51131008e-02, 3.61678727e-01,
        3.62127814e-01, 3.20680657e-02, 3.62280581e-01],
       [3.17840857e-01, 3.38141920e-01, 7.28542777e-02, 3.37462643e-01,
        3.38169282e-01, 3.19265687e-02, 3.34425886e-01],
       [2.94730711e-01, 3.13043812e-01, 7.16434378e-02, 3.12279094e-01,
        3.13219082e-01, 3.24826003e-02, 3.05421758e-01],
       [2.71491663e-01, 2.87820551e-01, 6.94467283e-02, 2.87106797e-01,
        2.88253834e-01, 3.26232597e-02, 2.76390041e-01],
       [2.48710392e-01, 2.63106638e-01, 6.67006175e-02, 2.62544710e-01,
        2.63860003e-01, 3.30091785e-02, 2.47979747e-01],
       [2.26848365e-01, 2.39403298e-01, 6.14817052e-02, 2.39057226e-01,
        2.40494498e-01, 3.32862067e-02, 2.20677191e-01],
       [2.06201745e-01, 2.17030151e-01, 3.78045775e-02, 2.16924052e-01,
        2.18430107e-01, 3.69296042e-02, 1.94727332e-01],
       [1.89418794e-01, 1.98879669e-01, 3.28163868e-02, 1.99227626e-01,
        2.00870901e-01, 3.41357328e-02, 1.74114936e-01],
       [1.75136993e-01, 1.83473296e-01, 3.15266353e-02, 1.84409679e-01,
        1.86219555e-01, 3.26868469e-02, 1.56794040e-01],
       [1.62815138e-01, 1.70232415e-01, 2.97494786e-02, 1.71886519e-01,
        1.73891961e-01, 3.00688248e-02, 1.42042917e-01],
       [1.51633304e-01, 1.58263213e-01, 3.05802136e-02, 1.60688245e-01,
        1.62885019e-01, 3.07728784e-02, 1.28790489e-01],
       [1.42008219e-01, 1.48024151e-01, 2.91729934e-02, 1.51367218e-01,
        1.53795584e-01, 2.88954068e-02, 1.17531735e-01],
       [1.33419054e-01, 1.38943883e-01, 2.75562019e-02, 1.43300247e-01,
        1.45978134e-01, 2.68875238e-02, 1.07587476e-01],
       [1.25445787e-01, 1.30569234e-01, 2.72037202e-02, 1.35991124e-01,
        1.38920712e-01, 2.62776964e-02, 9.84880884e-02],
       [1.18101288e-01, 1.22911034e-01, 2.64835984e-02, 1.29462651e-01,
        1.32652858e-01, 2.51207313e-02, 9.02268650e-02],
       [1.11251589e-01, 1.15823467e-01, 2.58549296e-02, 1.23557688e-01,
        1.27015159e-01, 2.40762773e-02, 8.26548963e-02],
       [1.04812558e-01, 1.09213826e-01, 2.53014417e-02, 1.18176039e-01,
        1.21906872e-01, 2.30734170e-02, 7.56783138e-02],
       [9.87235329e-02, 1.03014165e-01, 2.49657536e-02, 1.13243288e-01,
        1.17253585e-01, 2.25131016e-02, 6.92286675e-02],
       [9.30198219e-02, 9.72563714e-02, 2.43359653e-02, 1.08800658e-01,
        1.13100656e-01, 2.12897129e-02, 6.33105363e-02],
       [8.76076302e-02, 9.18381543e-02, 2.37331228e-02, 1.04732495e-01,
        1.09330574e-01, 1.99303166e-02, 5.78346963e-02],
       [8.24033758e-02, 8.66680461e-02, 2.34066671e-02, 1.00930935e-01,
        1.05833827e-01, 1.91807791e-02, 5.27292327e-02],
       [7.74352718e-02, 8.17703531e-02, 2.29132822e-02, 9.74251306e-02,
        1.02642612e-01, 1.77734322e-02, 4.79983858e-02],
       [7.27796901e-02, 7.71720398e-02, 2.26418288e-02, 9.41111321e-02,
        9.81868971e-02, 1.79321950e-02, 4.53487829e-02],
       [6.85788622e-02, 7.29831407e-02, 2.21729367e-02, 9.10119546e-02,
        9.39717784e-02, 1.72575335e-02, 4.27520781e-02],
       [6.47436998e-02, 6.91254590e-02, 2.17389436e-02, 8.80629543e-02,
        8.99306805e-02, 1.66607873e-02, 4.03909348e-02],
       [6.12133985e-02, 6.55467675e-02, 2.13882137e-02, 8.52252251e-02,
        8.60264540e-02, 1.64356581e-02, 3.82264615e-02],
       [5.79685644e-02, 6.22353672e-02, 2.10178431e-02, 8.25078086e-02,
        8.22706035e-02, 1.61624136e-02, 3.62441175e-02],
       [5.49864186e-02, 5.91741808e-02, 2.05931078e-02, 7.99095590e-02,
        7.86656520e-02, 1.56618234e-02, 3.44278375e-02],
       [5.22323550e-02, 5.63323902e-02, 2.01500433e-02, 7.74071028e-02,
        7.51951611e-02, 1.51119722e-02, 3.27557255e-02],
       [4.96768943e-02, 5.36833729e-02, 1.96896409e-02, 7.49781852e-02,
        7.18466452e-02, 1.45419466e-02, 3.12093569e-02],
       [4.72967488e-02, 5.12060699e-02, 1.92208963e-02, 7.26035842e-02,
        6.86134399e-02, 1.40247239e-02, 2.97741383e-02],
       [4.50767516e-02, 4.88871817e-02, 1.87019054e-02, 7.02699103e-02,
        6.54972913e-02, 1.33740775e-02, 2.84400609e-02],
       [4.29999281e-02, 4.67110173e-02, 1.82642835e-02, 6.79547528e-02,
        6.24959164e-02, 1.32469578e-02, 2.71965101e-02],
       [4.10722708e-02, 4.46852657e-02, 1.76299527e-02, 6.56605694e-02,
        5.96333490e-02, 1.22754539e-02, 2.60443917e-02],
       [3.92691837e-02, 4.27856166e-02, 1.70878401e-02, 6.33347774e-02,
        5.68942065e-02, 1.18214612e-02, 2.49697491e-02],
       [3.75899469e-02, 4.10122154e-02, 1.65740530e-02, 6.09423417e-02,
        5.42902204e-02, 1.14262911e-02, 2.39704191e-02],
       [3.60355072e-02, 3.93667108e-02, 1.63700405e-02, 5.84250378e-02,
        5.18326688e-02, 8.07917982e-03, 2.30449821e-02],
       [3.45387708e-02, 3.77803542e-02, 1.61001034e-02, 5.55921443e-02,
        4.94527341e-02, 1.02947060e-02, 2.21609926e-02],
       [3.31832206e-02, 3.63396765e-02, 7.77600888e-02, 5.24088573e-02,
        4.72527743e-02, 8.47937339e-03, 2.13549615e-02],
       [3.19404399e-02, 3.50156013e-02, 3.95073455e-03, 4.86400452e-02,
        4.52040437e-02, 7.34641795e-03, 2.06121581e-02],
       [3.08104041e-02, 3.38082602e-02, 3.04287132e-04, 4.41486463e-02,
        4.33086682e-02, 8.24131289e-03, 1.99308467e-02],
       [2.98235157e-02, 3.27490296e-02, 1.53466573e-02, 3.91432459e-02,
        4.15998829e-02, 4.66960917e-02, 1.93224271e-02]])
        rguess = 0.21152280845328936 # initially 0.09
        T_Hguess = 0.12
        factorguess = 6.568366845182593#7.7 # convert it to yen and account that unit of income in Millions 
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3
        BQguess = aggr.get_BQ(rguess, b_guess, None, p, 'SS', False)
        ss_params_baseline = (b_guess, n_guess, None, None, p, client)
        print('------------------------------------')
        print("Made it to -2!")
        print('------------------------------------')
        if p.use_zeta:
            guesses = [rguess] + list([BQguess]) + [T_Hguess, factorguess]
            print('------------------------------------')
            print("Made it to -1!")
            print('------------------------------------')
        else:
            guesses = [rguess] + list(BQguess) + [T_Hguess, factorguess]
            print('------------------------------------')
            print("Made it to 0!")
            print('------------------------------------')
        [solutions_fsolve, infodict, ier, message] =\
            opt.fsolve(SS_fsolve, guesses, args=ss_params_baseline,
                       xtol=10, full_output=True)#xtol=p.mindist_SS, full_output=True)
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            from ogusa import calibrate
            print('------------------------------------')
            print('LABOR MODEL MOMENTS', calibrate.calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J))
            print('------------------------------------')
            #raise RuntimeError('Steady state equilibrium not found')
            print('Steady state equilibrium not found')
            #print('Exception results:')
            #print(output)
            with open("output.txt", "a") as text_file:
                text_file.write('\nSteady state equilibrium not found. Exception results:\n' + str(output))
                text_file.write('\nEXCEPTION LABOR MODEL MOMENTS:\n' + str(calibrate.calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J)))
        print('------------------------------------')
        print("Made it to 1!")
        print('------------------------------------')
        rss = solutions_fsolve[0]
        BQss = solutions_fsolve[1:-2]
        T_Hss = solutions_fsolve[-2]
        factor_ss = solutions_fsolve[-1]
        Yss = T_Hss/p.alpha_T[-1]  # may not be right - if budget_balance
        # = True, but that's ok - will be fixed in SS_solver
        fsolve_flag = True
        # Return SS values of variables
        print('------------------------------------')
        print("Made it to 2!")
<<<<<<< HEAD
        if (n_guess < 0).any():
            print('2 HAS NEGATIVE N')
=======
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3
        print('------------------------------------')
        output = SS_solver(b_guess, n_guess, rss, BQss, T_Hss,
                           factor_ss, Yss, p, client, fsolve_flag)
        print('------------------------------------')
        print("Made it to 3!")
        print('------------------------------------')
    else:
        # Use the baseline solution to get starting values for the reform
        baseline_ss_dir = os.path.join(p.baseline_dir, 'SS/SS_vars.pkl')
        ss_solutions = pickle.load(open(baseline_ss_dir, 'rb'),
                                   encoding='latin1')
        (b_guess, n_guess, rguess, BQguess, T_Hguess, Yguess, factor) =\
            (ss_solutions['bssmat_splus1'], ss_solutions['nssmat'],
             ss_solutions['rss'], ss_solutions['BQss'],
             ss_solutions['T_Hss'], ss_solutions['Yss'],
             ss_solutions['factor_ss'])
        if p.baseline_spending:
            T_Hss = T_Hguess
            ss_params_reform = (b_guess, n_guess, T_Hss, factor, p, client)
            if p.use_zeta:
                guesses = [rguess] + list([BQguess]) + [Yguess]
            else:
                guesses = [rguess] + list(BQguess) + [Yguess]
            [solutions_fsolve, infodict, ier, message] =\
                opt.fsolve(SS_fsolve, guesses,
                           args=ss_params_reform, xtol=p.mindist_SS,
                           full_output=True)
            rss = solutions_fsolve[0]
            BQss = solutions_fsolve[1:-1]
            Yss = solutions_fsolve[-1]
        else:
            ss_params_reform = (b_guess, n_guess, None, factor, p, client)
            if p.use_zeta:
                guesses = [rguess] + list([BQguess]) + [T_Hguess]
            else:
                guesses = [rguess] + list(BQguess) + [T_Hguess]
            [solutions_fsolve, infodict, ier, message] =\
                opt.fsolve(SS_fsolve, guesses,
                           args=ss_params_reform, xtol=p.mindist_SS,
                           full_output=True)
            rss = solutions_fsolve[0]
            BQss = solutions_fsolve[1:-1]
            T_Hss = solutions_fsolve[-1]
            Yss = T_Hss/p.alpha_T[-1]  # may not be right - if
            # budget_balance = True, but that's ok - will be fixed in
            # SS_solver
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            from ogusa import calibrate
            print('------------------------------------')
            print('LABOR MODEL MOMENTS', calibrate.calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J))
            print('------------------------------------')
            #raise RuntimeError('Steady state equilibrium not found')  # want to save these variables so commented out !!!!
            print('Steady state equilibrium not found')
            #print('Exception results:')
            #print(output)
            with open("output.txt", "a") as text_file:
                text_file.write('\nSteady state equilibrium not found. Exception results:\n' + str(output))
                text_file.write('\nEXCEPTION LABOR MODEL MOMENTS:\n' + str(calibrate.calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J)))

        # Return SS values of variables
        fsolve_flag = True
        # Return SS values of variables
<<<<<<< HEAD
        if (n_guess < 0).any():
            print('N GUESS < 0 AT SS 894')
=======
>>>>>>> e45bda2b32d217b3ef4ed14b3dd87cf484ab68d3
        output = SS_solver(b_guess, n_guess, rss, BQss, T_Hss, factor,
                           Yss, p, client, fsolve_flag)
        if output['Gss'] < 0.:
            warnings.warn('Warning: The combination of the tax policy '
                          + 'you specified and your target debt-to-GDP '
                          + 'ratio results in an infeasible amount of '
                          + 'government spending in order to close the '
                          + 'budget (i.e., G < 0)')
    
    from ogusa import calibrate
    print('------------------------------------')
    print('LABOR MODEL MOMENTS', calibrate.calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J))
    print('------------------------------------')
    with open("output.txt", "a") as text_file:
        text_file.write('\nSteady state found. Equilibrium results:\n' + str(output))
        text_file.write('\nSUCCEESS LABOR MODEL MOMENTS:\n' + str(calibrate.calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J)))
    return output

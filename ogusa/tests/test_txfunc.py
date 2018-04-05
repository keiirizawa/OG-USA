from ogusa import txfunc

import pytest
import numpy as np
import pickle
import os

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize('tax_func_type,expected',
                         [('DEP', 0.032749763), ('GS', 0.007952744)],
                         ids=['DEP', 'GS'])
def test_wsumsq(tax_func_type, expected):
    rate_type = 'etr'
    A = 0.01
    B = 0.02
    C = 0.1
    D = 1.1
    max_x = 0.55
    min_x = 0.17
    max_y = 0.46
    min_y = 0.04
    shift_x = 0.1
    shift_y = 0.2
    shift = 0.04
    share = 0.8
    phi0 = 0.396
    phi1 = 0.7
    phi2 = 0.9
    X = np.array([32.0, 44.0, 1.6, 0.4])
    Y = np.array([32.0, 55.0, 0.9, 0.03])
    txrates = np.array([0.6, 0.5, 0.3, 0.25])
    wgts = np.array([0.1, 0.25, 0.55, 0.1])
    if tax_func_type == 'DEP':
        params = A, B, C, D, max_x, max_y, share
        args = ((min_x, min_y, shift), X, Y, txrates, wgts,
                tax_func_type, rate_type)
    elif tax_func_type == 'GS':
        params = phi0, phi1, phi2
        args = None, X, Y, txrates, wgts, tax_func_type, rate_type
    test_val = txfunc.wsumsq(params, *args)

    assert(np.allclose(test_val, expected))


@pytest.mark.parametrize('se_mult,expected_mat',
                         [(2,
                           np.array([[False, False], [False, False],
                                     [False, True]])),
                          (8,
                           np.array([[False, False], [False, False],
                                     [False, False]]))],
                         ids=['2', '8'])
def test_find_outliers(se_mult, expected_mat):
    # Test the find outliers function
    sse_mat = np.array([[21.0, 22.0], [20.0, 32.0], [20.0, 100.0]])
    age_vec = np.array([40, 41])
    start_year = 2018
    varstr = 'MTRy'
    test_mat = txfunc.find_outliers(sse_mat, age_vec, se_mult,
                                    start_year, varstr, False)

    assert(np.allclose(test_mat, expected_mat))


def test_replace_outliers():
    """
    4 cases:
        s is an outlier and is 0
        s is an outlier and is in the interior (s > 0 and s < S)
        s is not an outlier but the first s - 1 ages were (s = 1 in our case)
        s is an outlier and is the max age
    """
    S = 10
    BW = 2
    numparams = 3
    random_state = np.random.RandomState(10)
    param_arr = random_state.rand(S * BW * numparams).reshape(S, BW, numparams)
    sse_big_mat = ~ np.ones((S, BW), dtype=bool)
    sse_big_mat[0, 0] = True
    sse_big_mat[1, 0] = True
    sse_big_mat[S-4, 0] = True
    sse_big_mat[S-5, 0] = True
    sse_big_mat[S - 2, 0] = True
    sse_big_mat[S - 1, 0] = True

    act = txfunc.replace_outliers(param_arr, sse_big_mat)

    exp = [[[0.00394827, 0.51219226, 0.81262096],
            [0.74880388, 0.49850701, 0.22479665]],

           [[0.00394827, 0.51219226, 0.81262096],
            [0.08833981, 0.68535982, 0.95339335]],

           [[0.00394827, 0.51219226, 0.81262096],
            [0.61252607, 0.72175532, 0.29187607]],

           [[0.91777412, 0.71457578, 0.54254437],
            [0.14217005, 0.37334076, 0.67413362]],

           [[0.44183317, 0.43401399, 0.61776698],
            [0.51313824, 0.65039718, 0.60103895]],

           [[0.3608713, 0.57495943, 0.5290622],
            [0.31923609, 0.09045935, 0.30070006]],

           [[0.27990942, 0.71590487, 0.44035742],
            [0.62628715, 0.54758616, 0.819287]],

           [[0.19894754, 0.8568503, 0.35165264],
            [0.75464769, 0.29596171, 0.88393648]],

           [[0.19894754, 0.8568503, 0.35165264],
            [0.09346037, 0.82110566, 0.15115202]],

           [[0.19894754, 0.8568503, 0.35165264],
            [0.45630455, 0.82612284, 0.25137413]]]

    assert np.allclose(act, exp)


# def test_txfunc_est():


# def test_tax_func_loop():
    # Test txfunc.tax_func_loop() function.  The test is that given
    # inputs from previous run, the outputs are unchanged.
    with open(os.path.join(CUR_PATH,
                           'test_io_data/tax_func_loop_inputs.pkl'),
              'rb') as f:
        input_tuple = pickle.load(f)
    guesses, params = input_tuple
    params = params + (None, 1)
    test_list = SS.SS_fsolve(guesses, params)

    with open(os.path.join(CUR_PATH,
                           'test_io_data/tax_func_loop_outputs.pkl'),
              'rb') as f:
        expected_list = pickle.load(f)
    print 'outputs = ', np.absolute(np.array(test_list) - np.array(expected_list)).max()
    assert(np.allclose(np.array(test_list), np.array(expected_list)))

A = 0.02
B = 0.01
C = 0.003
D = 3.2
max_x = 0.6
min_x = 0.05
max_y = 0.8
min_y = 0.05
shift = 0.03
share = 0.7
phi0 = 0.6
phi1 = 0.5
phi2 = 0.6
@pytest.mark.parametrize('tax_func_type,rate_type,params,for_estimation,expected',
                         [('DEP', 'etr',
                           np.array([A, B, C, D, max_x, max_y, share,
                                     min_x, min_y, shift]), True,
                           np.array([0.1894527, 0.216354953,
                                     0.107391574, 0.087371974])),
                          ('DEP', 'etr',
                           np.array([A, B, C, D, max_x, max_y, share,
                                     min_x, min_y, shift]), False,
                           np.array([0.669061481, 0.678657921,
                                     0.190301075, 0.103958946])),
                          ('GS', 'etr',
                           np.array([phi0, phi1, phi2]), False,
                           np.array([0.58216409, 0.5876492, 0.441995766,
                                     0.290991255])),
                          ('GS', 'mtrx',
                           np.array([phi0, phi1, phi2]), False,
                           np.array([0.596924843, 0.598227987,
                                     0.518917438, 0.37824137])),
                          ('DEP_totalinc', 'etr',
                           np.array([A, B, max_x, min_x, shift]), True,
                           np.array([0.110821747, 0.134980034,
                                     0.085945843, 0.085573318])),
                          ('DEP_totalinc', 'etr',
                           np.array([A, B, max_x, min_x, shift]),
                           False,
                           np.array([0.628917903, 0.632722363,
                                     0.15723913, 0.089863997]))],
                         ids=['DEP for estimation',
                              'DEP not for estimation', 'GS, etr',
                              'GS, mtr', 'DEP_totalinc for estimation',
                              'DEP_totalinc not for estimation'])
def test_get_tax_rates(tax_func_type, rate_type, params, for_estimation,
                       expected):
    '''
    Teset of txfunc.get_tax_rates() function.  There are 6 cases to
    test:
    1) DEP function, for estimation
    2) DEP function, not for estimation
    3) GS function, etr
    4) GS function, mtr
    5) DEP_totalinc function, for estimation
    6) DEP_totalinc function, not for estimation
    '''
    wgts = np.array([0.1, 0.25, 0.55, 0.1])
    X = np.array([32.0, 44.0, 1.6, 0.4])
    Y = np.array([32.0, 55.0, 0.9, 0.03])
    test_txrates = txfunc.get_tax_rates(params, X, Y, wgts,
                                        tax_func_type, rate_type,
                                        for_estimation)

    assert np.allclose(test_txrates, expected)


# def test_tax_func_estimate():


# def test_get_tax_func_estimate():
# saves a pickle file

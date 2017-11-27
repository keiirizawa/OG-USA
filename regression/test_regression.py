from ogusa.macro_output import dump_diff_output
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pytest
import os


CURDIR = os.path.abspath(os.path.dirname(__file__))
REG_BASELINE = os.path.join(CURDIR, 'regression_results/REG_OUTPUT_BASELINE')
REG_REFORM = os.path.join(CURDIR, 'regression_results/REG_OUTPUT_REFORM_{ref_idx}')
REF_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

BASELINE = os.path.join(CURDIR, 'OUTPUT_BASELINE')
REFORM = os.path.join(CURDIR, 'OUTPUT_REFORM_{ref_idx}')

@pytest.fixture(scope="module", params=REF_IDXS)
def macro_outputs(request):
    (pct_changes,
        baseline_macros,
        policy_macros) = dump_diff_output(BASELINE,
                                          REFORM.format(ref_idx=request.param))
    (reg_pct_changes,
        reg_baseline_macros,
        reg_policy_macros) = dump_diff_output(REG_BASELINE,
                                              REG_REFORM.format(ref_idx=request.param))

    return {"new":{
                   "pct_changes": pct_changes,
                   "baseline_macros": baseline_macros,
                   "policy_macros": policy_macros
                  },
            "reg": {
                    "pct_changes": reg_pct_changes,
                    "baseline_macros": reg_baseline_macros,
                    "policy_macros": reg_policy_macros
                    },
            }


MACRO_VARS = ["Y", "C", "I", "L", "w", "r", "Revenue"]


@pytest.mark.regression
@pytest.mark.parametrize("macro_var_idx", np.arange(len(MACRO_VARS)))
def test_macro_output(macro_outputs, macro_var_idx):
    """
    Compare macro output

    macro_outputs: output as read from TPI/TPI_macro_vars.pkl using
                   macro_output.dump_diff_output
                   created in TPI.run_TPI
    macro_var_idx: index of TPI macro ouptut variable from OUTPUT_VARS
    """
    assert np.allclose(
        macro_outputs["new"]["pct_changes"][macro_var_idx, :],
        macro_outputs["reg"]["pct_changes"][macro_var_idx, :],
        atol=0.0, rtol=0.001
    )


@pytest.fixture(scope="module", params=REF_IDXS + ["baseline"])
def tpi_output(request):
    def get_tpi_output(path):
        with open(path + "/TPI/TPI_vars.pkl", 'rb') as f:
            return pickle.load(f)

    ref_idx = request.param
    if ref_idx == "baseline":
        return (get_tpi_output(REG_BASELINE), get_tpi_output(BASELINE))
    else:
        return (get_tpi_output(path=REG_REFORM.format(ref_idx=request.param)),
                get_tpi_output(path=REFORM.format(ref_idx=request.param)))


TPI_VARS = ['C', 'D', 'G', 'REVENUE', 'I', 'K', 'tax_path', 'L',
            'eul_laborleisure', 'T_H', 'r', 'n_mat', 'BQ', 'w', 'Y',
            'eul_savings', 'c_path', 'b_mat']


@pytest.mark.regression
@pytest.mark.parametrize("tpi_var", TPI_VARS)
def test_tpi_vars(tpi_output, tpi_var):
    """
    Compare TPI_vars

    tpi_output: output read from TPI/TPI_vars.pkl
                created in TPI.run_TPI
    tpi_var: TPI output variable from TPI_VARS
    """
    reg = tpi_output[0][tpi_var]
    new = tpi_output[1][tpi_var]
    assert np.allclose(reg, new, atol=0.0, rtol=0.001)


@pytest.fixture(scope="module", params=REF_IDXS + ["baseline"])
def txfunc_output(request):
    def get_txfunc_output(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    ref_idx = request.param
    if ref_idx == "baseline":
        reg_path = REG_BASELINE + "/TxFuncEst_{idx}.pkl".format(idx=ref_idx)
        path = BASELINE + "/TxFuncEst_{idx}.pkl".format(idx=ref_idx)

        return (get_txfunc_output(reg_path), get_txfunc_output(path))

    else:
        reg_path = (REG_REFORM.format(ref_idx=ref_idx) +
                    "/TxFuncEst_{idx}.pkl".format(idx=ref_idx))
        path = (REFORM.format(ref_idx=ref_idx) +
                "/TxFuncEst_{idx}.pkl".format(idx=ref_idx))

        return (get_txfunc_output(reg_path), get_txfunc_output(path))


TXFUNC_VARS = ['tfunc_mtrx_params_S', 'tfunc_avg_etr',
               'tfunc_avg_mtry', 'tfunc_mtrx_obs', 'tfunc_avg_mtrx',
               'tfunc_mtry_obs', 'tfunc_etr_sumsq', 'tfunc_mtrx_sumsq',
               'tfunc_avginc', 'tfunc_etr_obs', 'tfunc_etr_params_S',
               'tfunc_mtry_sumsq', 'tfunc_mtry_params_S']


@pytest.mark.regression
@pytest.mark.parametrize("txfunc_var", TXFUNC_VARS)
def test_txfunc_vars(txfunc_output, txfunc_var):
    """
    Compare tax function variables

    txfunc_output: output read from TxFuncEst_*.pkl
                   created in txfunc.tax_func_estimate
    txfunc_var: tax function output variable from TXFUNC_VARS
    """
    reg = txfunc_output[0][txfunc_var]
    new = txfunc_output[1][txfunc_var]
    assert np.allclose(reg, new, atol=0.0, rtol=0.001)

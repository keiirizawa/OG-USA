import json
import os
import collections as collect
import six
import re
import numpy as np

# import ogusa
from ogusa.parametersbase import ParametersBase
# from ogusa import elliptical_u_est

class Specifications(ParametersBase):
    """
    Inherits ParametersBase. Implements the PolicyBrain API for OG-USA
    """
    DEFAULTS_FILENAME = 'default_parameters.json'

    def __init__(self,
                 get_micro=False):
        super(Specifications, self).__init__()

        # reads in default data
        self._vals = self._params_dict_from_json_file()

        # does cheap calculations such as growth
        self.initialize(get_micro=get_micro)

        self.parameter_warnings = ''
        self.parameter_errors = ''
        self._ignore_errors = False

    def initialize(self, get_micro=False):
        """
        ParametersBase reads JSON file and sets attributes to self
        Next call self.compute_default_params for further initialization
        If estimate_params is true, then run long running estimation routines
        Parameters:
        -----------
        get_micro: boolean that indicates whether to estimate tax funtions
                   from microsim model
        """
        for name, data in self._vals.items():
            intg_val = data.get('integer_value', None)
            bool_val = data.get('boolean_value', None)
            values = data.get('value', None)
            if values:
                setattr(self, name,
                        self._expand_array(values, intg_val, bool_val))
        self.compute_default_params()
        if get_micro:
            self.get_micro_parameters()

    def compute_default_params(self):
        """
        Does cheap calculations to return parameter values
        """
        # self.b_ellipse, self.upsilon = elliptical_u_est.estimation(
        #     self.frisch[0],
        #     self.ltilde[0]
        # )
        #call some more functions
        pass

    def get_micro_parameters(self, data=None, reform={}):
        """
        Runs long running parameter estimatation routines such as estimating
        tax function parameters
        Parameters:
        ------------
        data: not sure what this is yet...
        reform: Tax-Calculator Policy reform
        Returns:
        --------
        nothing: void
        """
        # self.tax_func_estimate = tax_func_estimate(self.BW, self.S, self.starting_age, self.ending_age,
        #                                 self.start_year, self.baseline,
        #                                 self.analytical_mtrs, self.age_specific,
        #                                 reform=None, data=data)
        pass

    def default_parameters(self):
        """
        Return Policy object same as self except with current-law policy.
        Returns
        -------
        Specifications: Specifications instance with the default configuration
        """
        dp = Specifications()
        return dp

    def update_specifications(self, revision, raise_errors=True):
        """
        Updates parameter specification with values in revision dictionary
        Parameters
        ----------
        reform: dictionary of one or more PARAM:VALUE pairs
        raise_errors: boolean
            if True (the default), raises ValueError when parameter_errors
                    exists;
            if False, does not raise ValueError when parameter_errors exists
                    and leaves error handling to caller of
                    update_specifications.
        Raises
        ------
        ValueError:
            if raise_errors is True AND
              _validate_parameter_names_types generates errors OR
              _validate_parameter_values generates errors.
        Returns
        -------
        nothing: void
        Notes
        -----
        Given a reform dictionary, typical usage of the Policy class
        is as follows::
            specs = Specifications()
            specs.update_specifications(reform)
        An example of a multi-parameter specification is as follows::
            spec = {
                frisch: [0.03]
            }
        This method was adapted from the Tax-Calculator
        behavior.py-update_behavior method.
        """
        # check that all revisions dictionary keys are integers
        if not isinstance(revision, dict):
            raise ValueError('ERROR: revision is not a dictionary')
        if not revision:
            return  # no revision to implement
        revision_years = sorted(list(revision.keys()))
        # check range of remaining revision_years
        # validate revision parameter names and types
        self.parameter_errors = ''
        self.parameter_warnings = ''
        self._validate_parameter_names_types(revision)
        if not self._ignore_errors and self.parameter_errors:
            raise ValueError(self.parameter_errors)
        # implement the revision
        revision_parameters = set()
        revision_parameters.update(revision.keys())
        self._update(revision)
        # validate revision parameter values
        self._validate_parameter_values(revision_parameters)
        if self.parameter_errors and raise_errors:
            raise ValueError('\n' + self.parameter_errors)

    @staticmethod
    def read_json_param_objects(revision):
        """
        Read JSON file and convert to dictionary
        Returns
        -------
        rev_dict: formatted dictionary
        """
        # next process first reform parameter
        if revision is None:
            rev_dict = dict()
        elif isinstance(revision, six.string_types):
            if os.path.isfile(revision):
                txt = open(revision, 'r').read()
            else:
                txt = revision
            # strip out //-comments without changing line numbers
            json_str = re.sub('//.*', ' ', txt)
            # convert JSON text into a Python dictionary
            try:
                rev_dict = json.loads(json_str)
            except ValueError as valerr:
                msg = 'Policy reform text below contains invalid JSON:\n'
                msg += str(valerr) + '\n'
                msg += 'Above location of the first error may be approximate.\n'
                msg += 'The invalid JSON reform text is between the lines:\n'
                bline = 'XX----.----1----.----2----.----3----.----4'
                bline += '----.----5----.----6----.----7'
                msg += bline + '\n'
                linenum = 0
                for line in json_str.split('\n'):
                    linenum += 1
                    msg += '{:02d}{}'.format(linenum, line) + '\n'
                msg += bline + '\n'
                raise ValueError(msg)
        else:
            raise ValueError('reform is neither None nor string')

        return rev_dict

    def _validate_parameter_names_types(self, revision):
        """
        Check validity of parameter names and parameter types used
        in the specified revision dictionary.
        Parameters
        ----------
        revision: parameter dictionary of form {parameter_name: [value]}
        Returns:
        --------
        nothing: void
        Notes
        -----
        copied from taxcalc.Behavior._validate_parameter_names_types
        """
        param_names = set(self._vals.keys())
        revision_param_names = list(revision.keys())
        for param_name in revision_param_names:
            if param_name not in param_names:
                msg = '{} unknown parameter name'
                self.parameter_errors += (
                    'ERROR: ' + msg.format(param_name) + '\n'
                )
            else:
                # check parameter value type avoiding use of isinstance
                # because isinstance(True, (int,float)) is True, which
                # makes it impossible to check float parameters
                bool_param_type = self._vals[param_name]['boolean_value']
                int_param_type = self._vals[param_name]['integer_value']
                if isinstance(revision[param_name], list):
                    param_value = revision[param_name]
                else:
                    param_value = [revision[param_name]]
                for idx in range(0, len(param_value)):
                    pval = param_value[idx]
                    pval_is_bool = type(pval) == bool
                    pval_is_int = type(pval) == int
                    pval_is_float = type(pval) == float
                    if bool_param_type:
                        if not pval_is_bool:
                            msg = '{} value {} is not boolean'
                            self.parameter_errors += (
                                'ERROR: ' +
                                msg.format(param_name, pval) +
                                '\n'
                            )
                    elif int_param_type:
                        if not pval_is_int:  # pragma: no cover
                            msg = '{} value {} is not integer'
                            self.parameter_errors += (
                                'ERROR: ' +
                                msg.format(param_name, pval) +
                                '\n'
                            )
                    else:  # param is float type
                        if not (pval_is_int or pval_is_float):
                            msg = '{} value {} is not a number'
                            self.parameter_errors += (
                                'ERROR: ' +
                                msg.format(param_name, pval) +
                                '\n'
                            )
        del param_names


    def _validate_parameter_values(self, parameters_set):
        """
        Check values of parameters in specified parameter_set using
        range information from the current_law_policy.json file.
        Parameters:
        -----------
        parameters_set: set of parameters whose values need to be validated
        Returns:
        --------
        nothing: void
        Notes
        -----
        copied from taxcalc.Policy._validate_parameter_values
        """
        rounding_error = 100.0
        # above handles non-rounding of inflation-indexed parameter values
        dp = self.default_parameters()
        parameters = sorted(parameters_set)
        for param_name in parameters:
            param_value = getattr(self, param_name)
            if not hasattr(param_value, 'shape'):  # value is not a numpy array
                param_value = np.array([param_value])
            for validation_op, validation_value in self._vals[param_name]['range'].items():
                if isinstance(validation_value, six.string_types):
                    validation_value = self.simple_eval(validation_value)
                else:
                    validation_value = np.full(param_value.shape, validation_value)
                assert param_value.shape == validation_value.shape
                for idx in np.ndindex(param_value.shape):
                    out_of_range = False
                    if validation_op == 'min' and param_value[idx] < validation_value[idx]:
                        out_of_range = True
                        msg = '{} value {} < min value {}'
                        extra = self._vals[param_name]['out_of_range_minmsg']
                        if extra:
                            msg += ' {}'.format(extra)
                    if validation_op == 'max' and param_value[idx] > validation_value[idx]:
                        out_of_range = True
                        msg = '{} value {} > max value {}'
                        extra = self._vals[param_name]['out_of_range_maxmsg']
                        if extra:
                            msg += ' {}'.format(extra)
                    if out_of_range:
                        self.parameter_errors += (
                            'ERROR: ' + msg.format(param_name,
                                                   param_value[idx],
                                                   validation_value[idx]) + '\n'
                        )
        del dp
        del parameters

# copied from taxcalc.tbi.tbi.reform_errors_warnings--probably needs further
# changes
def reform_warnings_errors(user_mods):
    """
    Generate warnings and errors for OG-USA parameter specifications
    Parameters:
    -----------
    user_mods : dict created by read_json_param_objects
    Return
    ------
    rtn_dict : dict with endpoint specific warning and error messages
    """
    rtn_dict = {'ogusa': {'warnings': '', 'errors': ''}}

    # create Specifications object and implement reform
    specs = Specifications(2017)
    specs._ignore_errors = True
    try:
        specs.update_specifications(user_mods['ogusa'], raise_errors=False)
        rtn_dict['ogusa']['warnings'] = specs.parameter_warnings
        rtn_dict['ogusa']['errors'] = specs.parameter_errors
    except ValueError as valerr_msg:
        rtn_dict['ogusa']['errors'] = valerr_msg.__str__()
    return rtn_dict

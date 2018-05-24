import os
import tempfile
import pytest

from ogusa.pb_api import Specifications, reform_warnings_errors


JSON_REVISION_FILE = """{
    "revision": {
        "frisch": 0.03
    }
}"""

@pytest.fixture(scope='module')
def revision_file():
    f = tempfile.NamedTemporaryFile(mode="a", delete=False)
    f.write(JSON_REVISION_FILE)
    f.close()
    # Must close and then yield for Windows platform
    yield f
    os.remove(f.name)


def test_create_specs_object():
    specs = Specifications(2017)
    assert specs


def test_read_json_params_objects(revision_file):
    exp = {"revision": {"frisch": 0.03}}
    act1 = Specifications.read_json_param_objects(JSON_REVISION_FILE)
    assert exp == act1
    act2 = Specifications.read_json_param_objects(JSON_REVISION_FILE)
    assert exp == act2


def test_implement_reform():
    specs = Specifications(2017)
    new_specs = {
        'tG1': 30,
        'T': 80,
        'frisch': 0.03
    }

    specs.update_specifications(new_specs)
    assert specs.frisch == 0.03
    assert specs.tG1 == 30
    assert specs.T == 80
    assert len(specs.parameter_errors) == 0
    assert len(specs.parameter_warnings) == 0


def test_implement_bad_reform():
    specs = Specifications(2017)
    # tG1 has an upper bound at T / 2
    new_specs = {
        'tG1': 50,
        'T': 80
    }

    specs.update_specifications(new_specs, raise_errors=False)

    assert len(specs.parameter_errors) > 0
    assert specs.parameter_errors == 'ERROR: tG1 value 50 > max value 40.0\n'
    assert len(specs.parameter_warnings) == 0


def test_reform_warnings_errors():
    user_mods = {'ogusa': {'frisch': 0.03}}

    ew = reform_warnings_errors(user_mods)
    assert len(ew['ogusa']['errors']) == 0
    assert len(ew['ogusa']['warnings']) == 0

    user_mods = {'ogusa': {'frisch': 0.1}}

    bad_ew = reform_warnings_errors(user_mods)
    assert len(bad_ew['ogusa']['errors']) > 0
    assert len(bad_ew['ogusa']['warnings']) == 0


def test_simple_eval():
    specs = Specifications(2017)
    specs.T = 100
    assert specs.simple_eval('T / 2') == 50
    assert specs.simple_eval('T * 2') == 200
    assert specs.simple_eval('T - 2') == 98
    assert specs.simple_eval('T + 2') == 102

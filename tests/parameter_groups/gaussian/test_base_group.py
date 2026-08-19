import pytest
from pypolymix.parameter_groups import ParameterGroup

def test_ParameterGroup_is_abstract_base_class():
    with pytest.raises(TypeError):
        ParameterGroup()
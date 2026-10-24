import pytest

from pypolymix.likelihoods import Likelihood


def test_likelihood_is_abstract_base_class():
    with pytest.raises(TypeError):
        Likelihood()

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from triqs.gf import MeshReTime

from tddt.util import simpsons_weights


class test_util(unittest.TestCase):
    """Utility functions"""

    def test_simpsons_weights(self):
        mesh = MeshReTime(1.0, 3.0, 11)
        w = simpsons_weights(mesh)
        w_ref = 0.2 * np.array([1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1]) / 3
        assert_array_almost_equal(w, w_ref)

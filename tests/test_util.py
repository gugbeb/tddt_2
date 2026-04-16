import unittest
import numpy as np

from tddt.util import fermi


class TestUtil(unittest.TestCase):
    """Utilities"""

    def test_fermi(self):
        self.assertAlmostEqual(fermi(1), 1.0 / (1.0 + np.exp(1)))
        self.assertAlmostEqual(fermi(-1), 1.0 / (1.0 + np.exp(-1)))
        self.assertAlmostEqual(fermi(0), 0.5)
        self.assertAlmostEqual(fermi(-np.inf), 1)
        self.assertAlmostEqual(fermi(np.inf), 0)


if __name__ == '__main__':
    unittest.main()

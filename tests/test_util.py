# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# Copyright (C) 2021-2026, I. Krivenko, V. Harkov, V. Valmispild
#
# tddt is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# tddt is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# tddt. If not, see <http://www.gnu.org/licenses/>.
#
# ##############################################################################

import unittest
import numpy as np

from tddt.util import mapsum, fermi


class TestUtil(unittest.TestCase):
    """Utilities"""

    def test_mapsum(self):
        self.assertEqual(mapsum(lambda x: x.upper(), "abcd"), "ABCD")

    def test_fermi(self):
        self.assertAlmostEqual(fermi(1), 1.0 / (1.0 + np.exp(1)))
        self.assertAlmostEqual(fermi(-1), 1.0 / (1.0 + np.exp(-1)))
        self.assertAlmostEqual(fermi(0), 0.5)
        self.assertAlmostEqual(fermi(-np.inf), 1)
        self.assertAlmostEqual(fermi(np.inf), 0)


if __name__ == '__main__':
    unittest.main()

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

from setuptools import setup

setup(
    name='tddt',
    version='0.3.3',
    url='https://github.com/krivenko/tddt.git',
    author='Igor Krivenko, Viktor Harkov, Viktor Valmispild',
    author_email='iskrivenko@proton.me',
    description='Implementation of the time-dependent dual TRILEX theory',
    packages=['tddt'],
    install_requires=['numpy >= 1.12.0', 'scipy >= 1.10.0'],
)

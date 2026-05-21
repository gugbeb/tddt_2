# Implementation of the time-dependent dual TRILEX theory

Copyright (C) 2021-2026, I. Krivenko, V. Harkov, V. Valmispild

## Installation

Clone / unpack source code of the package and run

```bash
$ pip install .
```

in the root directory.
Verify proper functioning of the package by also running

```bash
$ pip install pytest pytest-mpi
$ pytest -v             # Skip the unit tests depending on MPI
$ pytest -v --with-mpi  # Run the MPI tests as well
```

## Documentation

Jupyter notebook at [`doc/example.ipynb`](doc/example.ipynb) contains usage
examples covering all major parts of this package's API.

## Acknowledgements

Development of this software was funded by the European Research Council (ERC)
under the European Union's Horizon 2020 research and innovation programme
(Grant Agreement No.
[854843-FASTCORR](https://cordis.europa.eu/project/id/854843)).

## License

tddt is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

tddt is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
tddt (in the file [LICENSE.txt](LICENSE.txt) in this directory). If not, see
<http://www.gnu.org/licenses/>.

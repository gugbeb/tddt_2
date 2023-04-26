Implementation of the time-dependent dual TRILEX theory
=======================================================

Verify proper functioning of the package by running

```bash
    pip install pytest pytest-mpi
    pytest -v             # Skip the realevol MPI tests
    pytest -v --with-mpi  # Run the realevol MPI tests as well
```

in its root directory.

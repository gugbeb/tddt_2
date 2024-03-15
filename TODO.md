TODO list
=========

* Re-implement `KeldyshGF.__matmul__()` using the component-wise treatment as
  described in Section 11 of the NESSi paper [1].

* Add a version of `compute_keldysh_vertex3()` that returns vertices with
  non-trivial `arg_index_shapes`.

* Embed a writable persistent overlay into the Apptainer image file.

* Try to implement a new class `KeldyshFunction` that stores an N-point contour
  function as N! real time components (see [2] for details).

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0010465520302277
[2]: https://iopscience.iop.org/article/10.1088/1751-8121/ab165d

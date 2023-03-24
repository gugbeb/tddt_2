#
# Volterra integral equations of the 2nd kind
#

from itertools import product
import numpy as np

from triqs.gf import MeshReTime

from .integration import GregoryIntegrator


class VIE2Solver:
    r"""
    Solve the Volterra integral equation of the second kind

      y_a(t) + \sum_b \int_0^t ds k_{ab}(t, s) y_b(s) = q_b(t)

    w.r.t. y_a(t) on a uniform real time mesh using start-up and time-stepping
    procedures based on the Gregory quadrature rule. The solution y_a(t) is,
    in general, tensor-valued with an arbitrary shape and 'a' is the
    corresponding multi-index.
    """

    def __init__(self, mesh: MeshReTime, solution_shape, /, gregory_order=5):
        self.N = len(mesh)
        assert self.N >= gregory_order + 1

        self.dt = mesh.delta
        self.solution_shape = solution_shape

        self.integrator = GregoryIntegrator(gregory_order)
        self.y_shape = (self.N,) + solution_shape
        self.k_shape = (self.N,) + (self.N,) + solution_shape + solution_shape

        self.y = np.zeros(self.y_shape)
        self.w = self.integrator.weights(mesh)

        self.startup_shape = (gregory_order, *solution_shape)
        self.startup_mat = np.zeros(self.startup_shape + self.startup_shape)
        self.startup_rhs = np.zeros(self.startup_shape)

        if solution_shape:
            self.stepping_mat = np.zeros(solution_shape + solution_shape)
            self.stepping_rhs = np.zeros(solution_shape)
        else:
            self.stepping_mat = np.zeros((1, 1))
            self.stepping_rhs = np.zeros(1)

    def startup(self, k: np.ndarray, q: np.ndarray):
        self.y[0, ...] = q[0, ...]

        order = self.integrator.order
        sol_slice = (slice(None),) * len(self.solution_shape)

        s_size = np.prod(self.startup_shape)

        self.startup_mat = np.eye(s_size).reshape(self.startup_mat.shape)
        for n, l in product(range(1, order + 1), repeat=2):
            self.startup_mat[(n - 1, *sol_slice, l - 1, *sol_slice)] += \
                self.dt * self.w[n, l] * k[n, l, ...]

        self.startup_rhs[:, ...] = q[1:(order + 1), ...]
        for n in range(1, order + 1):
            self.startup_rhs[n - 1, ...] -= \
                self.dt * self.w[n, 0] * \
                np.tensordot(k[n, 0, ...], self.y[0, ...], axes=self.y.ndim - 1)

        self.y[1:(order + 1), ...] = \
            np.linalg.tensorsolve(self.startup_mat, self.startup_rhs)

    def step(self, k, q, n):
        size = int(np.prod(self.solution_shape))

        self.stepping_mat = np.eye(size).reshape(self.stepping_mat.shape)
        self.stepping_mat += self.dt * self.w[n, n] * k[n, n, ...]

        self.stepping_rhs = q[n, ...]
        for l in range(n):  # noqa: E741
            self.stepping_rhs -= self.dt * self.w[n, l] \
                * np.tensordot(k[n, l, ...], self.y[l, ...],
                               axes=self.y.ndim - 1)

        self.y[n, ...] = np.linalg.tensorsolve(self.stepping_mat,
                                               self.stepping_rhs)

    def __call__(self, k: np.ndarray, q: np.ndarray):
        r"""
        Solve the integral equation.

        The array `q` is the right hand side. Its first axis corresponds to the
        time argument and the rest of axes (if any) correspond to the
        multi-index 'a' with dimensions specified by the `solution_shape` tuple
        passed to the constructor.

        The array `k` is the integral kernel. Its layout must be such that

        * Its first 2 axes correspond to the time arguments t and s.
        * The next `len(solution_shape)` axes correspond to the multi-index 'a'.
        * The final `len(solution_shape)` axes correspond to the multi-index
          'b'.

        The returned solution y_a(t) has the same layout as `q`.
        """
        assert k.shape == self.k_shape
        assert q.shape == self.y_shape

        self.startup(k, q)
        for n in range(self.integrator.order + 1, self.N):
            self.step(k, q, n)

        return self.y

#
# Dual TRILEX theory
#

from enum import Enum
from itertools import product
from typing import Union

from triqs.gf import MeshReTime, MeshProduct

import realevol.operators_tinterp as op

from .keldysh import Branch, KeldyshGF, conv
from .models import FiniteCluster
from .realevol import (
    compute_keldysh_gf,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)


IndicesType = tuple[Union[int, str], Union[int, str]]


class Channel(Enum):
    "Bosonic channel."
    CHARGE = 0
    "Charge channel"
    SPIN = 1
    "Spin channel"


class DualTRILEX:
    r"""
    Implementation of the "single-shot" dual TRILEX theory.
    """

    def __init__(self, system_ref: FiniteCluster,
                 t_mesh: MeshReTime,
                 *,
                 imp_states_up: list[IndicesType],
                 imp_states_dn: list[IndicesType]):
        r"""
        Initialize a dual TRILEX calculation.

        ref_system: Model object representing the reference system.
        t_mesh: Real time mesh used to define correlation functions.
        imp_states_up: A list of (block index, inner index) pairs corresponding
                       to the spin-up impurity manifold of the reference system.
        imp_states_dn: A list of (block index, inner index) pairs corresponding
                       to the spin-down impurity manifold of the reference
                       system.
        """
        self.system_ref = system_ref
        self.t_mesh = t_mesh
        self.tt_mesh = MeshProduct(t_mesh, t_mesh)
        self.ttt_mesh = MeshProduct(t_mesh, t_mesh, t_mesh)

        assert len(imp_states_up) == len(imp_states_dn), \
            "The numbers of spin-up and spin-down impurity states must be equal"
        self.imp_states_up = imp_states_up
        self.imp_states_dn = imp_states_dn

    def compute_ref_init_state(self, T: float, /, **kwargs):
        r"""
        Call realevol to compute the initial thermal state of the reference
        system.

        T: Temperature.
        kwargs: Parameters to be passed to make_equilibrium_init_state().
        """

        # Compute the initial state
        self.init_state_ref = self.system_ref.equilibrium_init_state(
            T, **kwargs
        )

    def compute_ref_correlators(self, **kwargs):
        r"""
        Call realevol to compute correlation functions of the reference system.
        The computed functions are stored as object attributes:

        self.g_ref: Single-particle Green's function of the reference system.
        self.g_imp: Single-particle Green's function of the impurity subsystem.
        self.chi_imp: Susceptibility of the impurity subsystem.
        self.Lambda: Three-point vertex of the impurity subsystem.

        kwargs: Parameters to be passed to compute_*() methods of realevol.
        """
        N = self.system_ref.N
        Nimp = len(self.imp_states_up)
        imp_states = [self.imp_states_up, self.imp_states_dn]

        def make_n_imp(spin, iimp1, iimp2):
            return op.c_dag(*imp_states[spin][iimp1]) \
                * op.c(*imp_states[spin][iimp2])

        #
        # Reference system GF
        #

        g_ref = compute_keldysh_gf(self.system_ref.gf_struct,
                                   self.init_state_ref,
                                   self.system_ref.hamiltonian,
                                   self.t_mesh,
                                   params=kwargs)

        self.g_ref = KeldyshGF(mesh=self.tt_mesh,
                               arg_index_shapes=((2, N), (2, N)))
        # Impurity GF
        self.g_imp = KeldyshGF(mesh=self.tt_mesh,
                               arg_index_shapes=((2, Nimp), (2, Nimp)))

        for spin1, spin2 in product(range(2), repeat=2):
            indices1 = self.system_ref.spin_states(spin1)
            indices2 = self.system_ref.spin_states(spin2)
            for (n1, ind1), (n2, ind2) in product(enumerate(indices1),
                                                  enumerate(indices2)):
                bl1, i1 = ind1
                bl2, i2 = ind2
                # Skip elements corresponding to different blocks of g_ref
                if bl1 != bl2:
                    continue

                for br in product(Branch, repeat=2):
                    self.g_ref[br][spin1, n1, spin2, n2] \
                        << g_ref[bl1][br][i1, i2]

                if (ind1 in imp_states[spin1]) and (ind2 in imp_states[spin2]):
                    iimp1 = imp_states[spin1].index(ind1)
                    iimp2 = imp_states[spin2].index(ind2)
                    for br in product(Branch, repeat=2):
                        self.g_imp[br][spin1, iimp1, spin2, iimp2] << \
                            self.g_ref[br][spin1, n1, spin2, n2]

        #
        # Impurity susceptibility
        #

        # Generator of scalar-valued elements of impurity susceptibility
        def generator_chi_imp(ind1, ind2):
            chan1, iimp1, iimp2 = ind1
            chan2, iimp3, iimp4 = ind2

            s1 = 1 if (chan1 == Channel.CHARGE.value) else -1
            op1 = make_n_imp(0, iimp1, iimp2) + s1 * make_n_imp(1, iimp1, iimp2)
            s2 = 1 if (chan2 == Channel.CHARGE.value) else -1
            op2 = make_n_imp(0, iimp3, iimp4) + s2 * make_n_imp(1, iimp3, iimp4)

            return -1j * compute_keldysh_conn_correlator_2t(
                op1, op2,
                self.init_state_ref, self.system_ref.hamiltonian, self.t_mesh,
                params=kwargs
            )

        self.chi_imp = KeldyshGF.from_arg_index_gen(
            generator_chi_imp,
            mesh=self.tt_mesh,
            arg_index_shapes=((2, Nimp, Nimp), (2, Nimp, Nimp))
        )

        #
        # Three-point vertex of the impurity
        #

        # Generator of scalar-valued elements of the vertex
        def generator_Lambda_imp(ind1, ind2, ind3):
            spin1, iimp1 = ind1
            spin2, iimp2 = ind2
            chan, iimp3, iimp4 = ind3

            c_index = imp_states[spin1][iimp1]
            c_dag_index = imp_states[spin2][iimp2]

            s = 1 if (chan == Channel.CHARGE.value) else -1
            n_op = make_n_imp(0, iimp3, iimp4) + s * make_n_imp(1, iimp3, iimp4)

            return compute_keldysh_vertex3(
                c_index, c_dag_index, n_op,
                self.init_state_ref,
                self.system_ref.hamiltonian,
                self.t_mesh,
                params=kwargs
            )

        self.Lambda = KeldyshGF.from_arg_index_gen(
            generator_Lambda_imp,
            mesh=self.ttt_mesh,
            arg_index_shapes=((2, Nimp), (2, Nimp), (2, Nimp, Nimp))
        )


def polarization_2nd_order(Lambda: KeldyshGF, g: KeldyshGF):
    r"""
    2nd order contribution to the polarization function.

    Lambda: 3-point vertex.
    g: Fermionic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert g.n_args == 2, "g must be a 2-point GF"

    # f(z_0, z_1, z_2) = \int_C d\bar z \Lambda(z_0, \bar z, z_2) g(\bar z, z_1)
    f = conv(Lambda, g, [(1, 0)], free_args=([0, 2], [1]))
    # \Pi(z_1, z_2) = -i \int_C dz' dz'' f(z', z'', z_1) f(z'', z', z_2)
    return -1j * conv(f, f, [(0, 1), (1, 0)])


def selfenergy_2nd_order(Lambda: KeldyshGF, g: KeldyshGF, w: KeldyshGF):
    r"""
    2nd order contribution to the self-energy function.

    Lambda: 3-point vertex.
    g: Fermionic line.
    w: Bosonic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert g.n_args == 2, "g must be a 2-point GF"
    assert w.n_args == 2, "w must be a 2-point GF"

    # f1(z_1, z'''', z'') = \int_C dz' \Lambda(z_1, z', z'') g(z', z'''')
    f1 = conv(Lambda, g, [(1, 0)], free_args=([0, 2], [1]))
    # f2(z'''', z_2, z'') = \int_C dz''' \Lambda(z'''', z_2, z''') w(z'', z''')
    f2 = conv(Lambda, w, [(2, 1)], free_args=([0, 1], [2]))
    # \Sigma(z_1, z_2) = i \int_C dz'' dz'''' f1(z_1, z'''', z'')
    #                                         f2(z'''', z_2, z'')
    return 1j * conv(f1, f2, [(1, 0), (2, 2)])


def selfenergy_2nd_order_hf(Lambda: KeldyshGF, g: KeldyshGF, w: KeldyshGF):
    r"""
    Hartree-Fock contribution to the self-energy function.

    Lambda: 3-point vertex.
    g: Fermionic line.
    w: Bosonic line.
    """
    assert Lambda.n_args == 3, "Lambda must be a 3-point vertex"
    assert g.n_args == 2, "g must be a 2-point GF"
    assert w.n_args == 2, "w must be a 2-point GF"

    # n(z'') = \int_C dz''' dz'''' \Lambda(z''', z'''', z'') g(z'''', z''')
    n = conv(Lambda, g, [(0, 1), (1, 0)])
    # wn(z') = \int_C dz'' w(z', z'') n(z'')
    wn = conv(w, n, [(1, 0)])
    # \Sigma_{HF}(z_1, z_2) = -i \int_C dz' \Lambda(z_1, z_2, z') wn(z')
    return -1j * conv(Lambda, wn, [(2, 0)])

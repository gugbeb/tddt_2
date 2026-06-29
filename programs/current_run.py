"""
D-TRILEX driver for the biased-junction (current) experiment.

Two independent Mickey-Mouse 1 units serve as reference impurities, with
bath potentials shifted by ±V_bias/2.  The system is a Mickey-Mouse² pair
where each unit can have its own bath parameters.

All dual propagators carry an explicit 2×2 site-space structure
(target_shape=(2,2)); the [[G_∥, G_⊥],[G_⊥, G_∥]] block symmetry of the
unbiased case is broken and is not exploited here.

Usage
-----
    mpirun -n N python current_run.py param_current.in

Parameters (param file or CLI)
-------------------------------
    --U0              float  non-interacting interaction (quench initial)
    --U               float  reference interaction
    --U1              float  system interaction
    --n_t             int    number of real-time grid points
    --t_max           float  maximum real time
    --T               float  temperature 1/β
    --V_bias          float  bias voltage (bath1 at +V_bias/2, bath2 at -V_bias/2)
    --t_ref_array     float+ reference bath hoppings (same for both units)
    --t_sys_array     float+ sys bath hoppings for unit 1
    --t_sys_array_2   float+ sys bath hoppings for unit 2 (default: t_sys_array)
    --t_perp          float  inter-site hopping t_⊥
    --V               float  inter-site Coulomb interaction V
    --hartree_on             include Hartree-Fock tadpole diagram
    --output_dir      str    output directory (default: programs/data/current)
    --output_name     str    HDF5 base name  (default: current_results)
    --simple_output          write plain-text .dat files instead of HDF5
"""

import argparse
import shlex
import sys
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------
_has_mpi = False
_comm    = None
_rank    = 0
_size    = 1

try:
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    _has_mpi = True
    _comm    = MPI.COMM_WORLD
    _rank    = _comm.Get_rank()
    _size    = _comm.Get_size()
except Exception:
    pass


def get_time():
    if _has_mpi:
        return MPI.Wtime()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
from triqs.gf import MeshReTime, MeshProduct, Gf
from itertools import product as iproduct

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

if __package__ is None:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root  = os.path.dirname(_script_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import h5py
from programs.utilities import save_keldysh_gf_2pt, save_keldysh_gf_3pt, print_GF
from programs.dual_quantities import (
    DualQuantities,
    KeldyshGF_2_components,
    herm_viol,
)
from programs.mickey_mouse import MickeyMouseModel, MickeyMouseModel2
from tddt.keldysh import KeldyshGF, Branch, Singular2PKeldyshGF


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="D-TRILEX biased-junction (current) driver",
)
_parser.add_argument("--U0",             type=float)
_parser.add_argument("--U",              type=float)
_parser.add_argument("--U1",             type=float)
_parser.add_argument("--n_t",            type=int)
_parser.add_argument("--t_max",          type=float)
_parser.add_argument("--T",              type=float)
_parser.add_argument("--V_bias",         type=float, default=0.0,
                     help="Bias voltage: bath1 at +V_bias/2, bath2 at -V_bias/2")
_parser.add_argument("--t_ref_array",    type=float, nargs="+")
_parser.add_argument("--t_sys_array",    type=float, nargs="+")
_parser.add_argument("--t_sys_array_2",  type=float, nargs="+", default=None,
                     help="Sys bath hoppings for unit 2 (default: t_sys_array)")
_parser.add_argument("--t_perp",         type=float, default=0.0)
_parser.add_argument("--V",              type=float, default=0.0)
_parser.add_argument("--hartree_on",     action="store_true")
_parser.add_argument("--output_dir",     type=str, default="programs/data/current")
_parser.add_argument("--output_name",    type=str, default="current_results")
_parser.add_argument("--simple_output",  action="store_true")

# ---------------------------------------------------------------------------
# MPI-safe argument parsing: rank 0 reads param file, broadcasts to all
# ---------------------------------------------------------------------------
if _rank == 0:
    with open(sys.argv[1]) as f:
        lines = []
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    _arg_list = shlex.split(" ".join(lines))
    _args = _parser.parse_args(_arg_list)
else:
    _args = None

if _has_mpi:
    _args = _comm.bcast(_args, root=0)

# ---------------------------------------------------------------------------
# Unpack args
# ---------------------------------------------------------------------------
U0             = _args.U0
U              = _args.U
U1             = _args.U1
n_t            = _args.n_t
t_max          = _args.t_max
T              = _args.T
V_bias         = _args.V_bias
t_ref_array    = np.asarray(_args.t_ref_array, dtype=float)
t_sys_array    = np.asarray(_args.t_sys_array, dtype=float)
t_sys_array_2  = np.asarray(_args.t_sys_array_2 if _args.t_sys_array_2 is not None
                             else _args.t_sys_array, dtype=float)
t_perp         = _args.t_perp
V              = _args.V
output_dir     = _args.output_dir
output_name    = _args.output_name

t_mesh   = MeshReTime(0, t_max, n_t)
tt_mesh  = MeshProduct(t_mesh, t_mesh)
ttt_mesh = MeshProduct(t_mesh, t_mesh, t_mesh)

# Reference: single correlated site with both leads (symmetric).
eps_ref = np.array([+0.5 * V_bias, -0.5 * V_bias])

ref_params = {
    "U0": U0, "U": U, "mu": 0.5 * U,
    "eps_array": eps_ref, "t_array": t_ref_array, "T": T,
}
sys_params = {
    "U0": U0, "U": U1, "mu": 0.5 * U1,
    "eps_array":   np.array([+0.5 * V_bias]),  # unit 1: left lead only
    "t_array":     t_sys_array,
    "eps_array_2": np.array([-0.5 * V_bias]),  # unit 2: right lead only
    "t_array_2":   t_sys_array_2,
    "T": T,
    "t_perp": t_perp, "V": V,
}
# dual_params entries are 2×2 matrices: diagonal U values + V off-diagonal for the
# system charge channel.  DualQuantities.setW_bare does tmp[t] = U[ch] which works
# identically for scalars and numpy arrays when target_shape matches.
_I2 = np.eye(2, dtype=complex)
_J2 = np.array([[0, 1], [1, 0]], dtype=complex)   # off-diagonal coupling matrix
dual_params = {
    "U_ch":      0.5  * U  * _I2,
    "U_sp":     -0.5  * U  * _I2,
    "U_ch_sys":  0.5  * U1 * _I2 + V * _J2,   # V enters as off-diagonal inter-site Coulomb
    "U_sp_sys": -0.5  * U1 * _I2,
}
solver_params = {
    "verbosity":               2,
    "hbar":                    1.0,
    "hamiltonian_interpol":    "Trapezoid",
    "lanczos_min_matrix_size": 40,
}

if _rank == 0:
    print(f"Running with MPI: {_has_mpi}, size: {_size}, rank: {_rank}")
    print(f"V_bias = {V_bias},  t_perp = {t_perp},  V = {V}")

# ---------------------------------------------------------------------------
# Step 1: Reference impurity models (MPI-collective)
# ---------------------------------------------------------------------------
# Both refs are identical — compute once.
Mickey_ref = MickeyMouseModel(ref_params, t_mesh)
Mickey_sys = MickeyMouseModel2(sys_params,  t_mesh)

t_start = get_time()

if _rank == 0:
    print("Computing G_ref (biased bath: ε = [+V_bias/2, -V_bias/2]) ...")
Mickey_ref.computeGandGimp(solver_params)

if _rank == 0:
    print("Computing Chi, Pi, Lambda for ref ...")
Mickey_ref.computeChiAndPi(t_mesh, solver_params)
Mickey_ref.computeLambda(t_mesh, solver_params)

t_impurity_done = get_time() - t_start
if _rank == 0:
    print(f"Reference impurity done  [{t_impurity_done:.3f} s]")
    print("Computing Mickey2 system GF ...")

t_sys_start = get_time()
Mickey_sys.computeGandGimp(solver_params)

# Non-rank-0 processes exit after the impurity calculations.
if _has_mpi and _size > 1:
    _comm.Barrier()
    if _rank != 0:
        sys.exit(0)

# ---------------------------------------------------------------------------
# Step 2: Build 2×2 dual inputs from the two independent references
# ---------------------------------------------------------------------------
target_shape = (2, 2)

# g_imp_2x2: diagonal 2×2, ref1 on site 0 and ref2 on site 1.
g_imp_2x2 = KeldyshGF(mesh=tt_mesh, target_shape=target_shape)
for b1, b2 in iproduct(Branch, repeat=2):
    g_imp_2x2[b1, b2].data[..., 0, 0] = Mickey_ref.g_imp[b1, b2].data[..., 0, 0]
    g_imp_2x2[b1, b2].data[..., 1, 1] = Mickey_ref.g_imp[b1, b2].data[..., 0, 0]

g_imp_2comp = KeldyshGF_2_components(t_mesh, target_shape, is_reg=True, reg=g_imp_2x2)

# pi_imp_2x2: diagonal 2×2 per channel.
pi_imp_2x2 = {}
for ch in ["ch", "sp"]:
    pi = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
    for b1, b2 in iproduct(Branch, repeat=2):
        pi[b1, b2].data[..., 0, 0] = Mickey_ref.pi_imp[ch][b1, b2].data[..., 0, 0]
        pi[b1, b2].data[..., 1, 1] = Mickey_ref.pi_imp[ch][b1, b2].data[..., 0, 0]
    pi_imp_2x2[ch] = pi

# Lambda_2x2: block-diagonal per channel.
Lambda_2x2 = {}
for ch in ["ch", "sp"]:
    L = KeldyshGF(mesh=ttt_mesh, arg_index_shapes=((2,), (2,), (2,)))
    for b1, b2, b3 in iproduct(Branch, repeat=3):
        L[b1, b2, b3].data[..., 0, 0, 0] = Mickey_ref.Lambda[ch][b1, b2, b3].data[..., 0, 0, 0]
        L[b1, b2, b3].data[..., 1, 1, 1] = Mickey_ref.Lambda[ch][b1, b2, b3].data[..., 0, 0, 0]
    Lambda_2x2[ch] = L

# delta_tilde: diagonal regular part + off-diagonal singular t_perp part.
# Regular: (Δ_sys1 - Δ_ref1, Δ_sys2 - Δ_ref2) on the diagonal.
delta_hyb_reg = KeldyshGF(mesh=tt_mesh, target_shape=target_shape)
for b1, b2 in iproduct(Branch, repeat=2):
    diff1 = (Mickey_sys.hyb1 - Mickey_ref.hyb)[b1, b2].data[..., 0, 0]
    diff2 = (Mickey_sys.hyb2 - Mickey_ref.hyb)[b1, b2].data[..., 0, 0]
    delta_hyb_reg[b1, b2].data[..., 0, 0] = diff1
    delta_hyb_reg[b1, b2].data[..., 1, 1] = diff2

# Singular: t_perp in the off-diagonal, representing the direct inter-site coupling.
t_perp_mat = np.zeros((2, 2), dtype=complex)
t_perp_mat[0, 1] = t_perp_mat[1, 0] = t_perp
t_perp_gf = Gf(mesh=t_mesh, target_shape=target_shape)
for t in t_mesh:
    t_perp_gf[t] = t_perp_mat
t_perp_sing = Singular2PKeldyshGF.from_retime(t_perp_gf)

delta_tilde = KeldyshGF_2_components(
    t_mesh, target_shape, is_reg=False,
    sing=t_perp_sing, reg=delta_hyb_reg,
)

# ---------------------------------------------------------------------------
# Step 3: Dual quantities
# ---------------------------------------------------------------------------
if _rank == 0:
    print("Setting up dual quantities (2×2 biased-junction) ...")
t_dual_start = get_time()

dual = DualQuantities(
    t_mesh, dual_params, delta_tilde, g_imp_2comp, pi_imp_2x2, Lambda_2x2,
    target_shape=(2, 2),
)

t_dual_setup = get_time() - t_dual_start
if _rank == 0:
    print(f"Dual quantities setup done  [{t_dual_setup:.3f} s]")

# ---------------------------------------------------------------------------
# Step 4: Dual self-energy Σ̃
# ---------------------------------------------------------------------------
if _rank == 0:
    print("Computing dual self-energy ...")
t_se_start = get_time()

plots_dir = os.path.join(output_dir, "plots")
dual.computeFermionSelfEnergy(Lambda_2x2, hartree_on=_args.hartree_on, plots_dir=plots_dir)

t_se = get_time() - t_se_start
if _rank == 0:
    print(f"Dual self-energy done  [{t_se:.3f} s]")

# ---------------------------------------------------------------------------
# Step 5: Physical Green's functions
# ---------------------------------------------------------------------------
# Dyson equation: (1 - (g_imp + Σ̃) Δ̃) G = g_imp + Σ̃
# Dyson equation solved via dual.solveVie2 — works for any target_shape.
K_dtrilex = g_imp_2comp + dual.sigma
G_dtrilex  = dual.solveVie2(-(K_dtrilex @ delta_tilde), K_dtrilex)

K_cpt = g_imp_2comp
G_cpt = dual.solveVie2(-(K_cpt @ delta_tilde), K_cpt)

# System GF (full 2×2 from Mickey2)
G_sys_2x2 = Mickey_sys.computeGimp_matrix()

t_sys   = get_time() - t_sys_start
t_total = get_time() - t_start

if _rank == 0:
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Reference impurities (G, Chi, Pi, Lambda × 2): {t_impurity_done:.3f} s")
    print(f"Dual quantities setup:                          {t_dual_setup:.3f} s")
    print(f"Dual self-energy:                               {t_se:.3f} s")
    print(f"TOTAL:                                          {t_total:.3f} s")
    print(f"======================\n")

# ---------------------------------------------------------------------------
# Step 6: Output
# ---------------------------------------------------------------------------
brf, brb = Branch.FORWARD, Branch.BACKWARD

if _rank == 0:
    os.makedirs(output_dir, exist_ok=True)

    if _args.simple_output:
        print_GF(os.path.join(output_dir, "G_ref.dat"),        Mickey_ref.g_imp,  brf, brb, t_mesh, idx0=0)
        print_GF(os.path.join(output_dir, "G_sys_00.dat"),     G_sys_2x2,          brf, brb, t_mesh, idx0=0)
        print_GF(os.path.join(output_dir, "G_sys_11.dat"),     G_sys_2x2,          brf, brb, t_mesh, idx0=1)
        print_GF(os.path.join(output_dir, "G_dtrilex_00.dat"), G_dtrilex.reg,      brf, brb, t_mesh, idx0=0)
        print_GF(os.path.join(output_dir, "G_dtrilex_11.dat"), G_dtrilex.reg,      brf, brb, t_mesh, idx0=1)
        print_GF(os.path.join(output_dir, "G_cpt_00.dat"),     G_cpt.reg,          brf, brb, t_mesh, idx0=0)
        print_GF(os.path.join(output_dir, "G_cpt_11.dat"),     G_cpt.reg,          brf, brb, t_mesh, idx0=1)
        print(f"Done — plain-text files written to {output_dir}/")
    else:
        h5_path = os.path.join(output_dir, output_name + ".h5")
        print(f"Writing results to {h5_path} ...")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("t_mesh",  data=np.linspace(0, t_max, n_t))
            f.create_dataset("V_bias",  data=V_bias)
            f.create_dataset("t_perp",  data=t_perp)
            f.create_dataset("V",       data=V)

            ref_grp = f.require_group("ref")
            save_keldysh_gf_2pt(ref_grp, "G_ref", Mickey_ref.g_imp)
            chi_grp = ref_grp.require_group("chi")
            lam_grp = ref_grp.require_group("Lambda")
            for ch in ["ch", "sp"]:
                save_keldysh_gf_2pt(chi_grp, ch, Mickey_ref.chi_imp[ch])
                save_keldysh_gf_3pt(lam_grp, ch, Mickey_ref.Lambda[ch])

            # Occupation n_i(t) = -i G^{FB}_{ii}(t,t) per site (corr + bath).
            # Bond current J_{j->0}(t) = 2 t_j Im[G^{FB}_{0,j}(t,t)].
            g_fb = Mickey_ref.g["up"][brf, brb].data  # (n_t, n_t, N, N)
            _tidx = np.arange(n_t)
            g_fb_tt = g_fb[_tidx, _tidx, :, :]        # (n_t, N, N) equal-time slice
            n_occ   = -1j * np.diagonal(g_fb_tt, axis1=-2, axis2=-1)  # (n_t, N)
            # currents from each bath site (index 1..n_bath) to corr site (index 0)
            J_to_imp = 2.0 * t_ref_array * np.real(g_fb_tt[:, 0, 1:])  # (n_t, n_bath)
            ref_grp.create_dataset("n_occ",    data=np.real(n_occ))
            ref_grp.create_dataset("J_to_imp", data=J_to_imp)
            ref_grp.create_dataset("eps_bath", data=eps_ref)

            dual_grp = f.require_group("dual")
            save_keldysh_gf_2pt(dual_grp, "g_dual",     dual.g0.reg)
            save_keldysh_gf_2pt(dual_grp, "sigma_dual", dual.sigma.reg)
            w_grp  = dual_grp.require_group("w_dual")
            pi_grp = dual_grp.require_group("pi_dual")
            for ch in ["ch", "sp"]:
                save_keldysh_gf_2pt(w_grp,  ch, dual.w0[ch].reg)
                save_keldysh_gf_2pt(pi_grp, ch, dual.pi[ch].reg)

            sys_grp = f.require_group("sys")
            save_keldysh_gf_2pt(sys_grp, "G_sys",     G_sys_2x2)
            save_keldysh_gf_2pt(sys_grp, "G_dtrilex", G_dtrilex.reg)
            save_keldysh_gf_2pt(sys_grp, "G_cpt",     G_cpt.reg)

        print(f"Done — {h5_path}")

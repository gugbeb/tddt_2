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

from itertools import product
import numpy as np

import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, MeshBrZone, MeshProduct, Gf
from triqs.lattice import BravaisLattice, BrillouinZone

from realevol.tinterp import TInterp as ti

from tddt.keldysh import Branch, KeldyshGF, herm_conj
from tddt.dtrilex import DualTRILEX, Channel

from tddt.lattice import local_part
from tddt.models import FiniteCluster
from tddt.vie2 import solve_vie2

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

############################ Time meshes #######################################

t_max = 10.0
n_t = 51
# Time mesh for correlation functions
t_mesh = MeshReTime(0, t_max, n_t)
# Time mesh used to construct interpolators
ti_mesh = MeshReTime(0, t_max, 5001)

########################## Lattice problem #####################################

n_k = 2             # Number of k-points along each dimension
t_nn = 1.0          # Nearest neighbor hopping
t_nnn = 0.0         # Next nearest neighbor hopping

# Vector potential: Amplitude and frequency
A = 0.0
Omega = 4.0

lat = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # 2D square lattice
bz_mesh = MeshBrZone(BrillouinZone(lat), n_k)       # k-mesh on 1BZ; 0 - 2pi

# Lattice dispersion \eps_{\sigma,l,\sigma',l'}(t, k)
eps_tk = Gf(
    mesh=MeshProduct(t_mesh, bz_mesh),
    target_shape=(2, 1, 2, 1)  # 2 spins, 1 orbital
)

for t, k in eps_tk.mesh:
    A_t = A * np.cos(Omega * t.value)
    for spin in range(2):
        eps_tk[t, k][spin, 0, spin, 0] = \
            2 * t_nn * (np.cos(k[0] - A_t) + np.cos(k[1] - A_t)) \
            + 4 * t_nnn * np.cos(2 * (k[0] - A_t)) * np.cos(2 * (k[1] - A_t))

# Interaction U^\varsigma_{l_1,l_2,l_3,l_4}(t, q)

U = 6.0             # Hubbard interaction at t=0
U1 = 4.0            # Hubbard interaction at t>0
V = 0.2             # Non-local interaction strength
Uch = U1 / 2
Usp = -U1 / 2

U_tq = Gf(
    mesh=MeshProduct(t_mesh, bz_mesh),
    target_shape=(2, 1, 1, 1, 1)  # 2 channels, 1 orbital
)

for t, q in U_tq.mesh:
    V_q = 2 * V * (np.cos(k[0]) + np.cos(k[1]))
    U_tq[t, q][Channel.CHARGE.value] = Uch + V_q
    U_tq[t, q][Channel.SPIN.value] = Usp + V_q

# Double-counting interaction shifts
U_dc = np.zeros((2, 1, 1, 1, 1))  # 2 channels, 1 orbital
U_dc[Channel.CHARGE.value] = Uch
U_dc[Channel.SPIN.value] = Usp

########################## Reference system ####################################

# Correlated plaquette site (0)
mu = 0.5 * U        # Chemical potential at t=0
mu1 = 0.5 * U1      # Chemical potential at t>0

# Energy levels of uncorrelated plaquette sites are +exx, -exx, ex
ex = 0.0
exx = 3.0

# Hopping amplitudes between site 0 and the uncorrelated sites are txx, txx, tx
tx = 0.7 * t_nn
txx = 1.0 * t_nn

# Time-dependent vector potential (x- and y-component)
Ax_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])
Ay_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])

# Temperature
T = 0.01

# Reference system
model_ref = FiniteCluster(
    # 2x2 plaquette
    [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
    # Hopping matrix
    hopping=[[-mu, txx, txx,  tx],                            # noqa: E202, E241
             [txx, exx, 0,    0 ],                            # noqa: E202, E241
             [txx, 0,   -exx, 0 ],                            # noqa: E202, E241
             [tx,  0,   0,    ex]],                           # noqa: E202, E241
    # Local interaction
    local_int=[U, 0, 0, 0],
    # Vector potential: Components along the two Cartesian axes
    vector_potential=(Ax_t, Ay_t, 0)
)

theory = DualTRILEX(model_ref, t_mesh,
                    imp_states_up=[('up', 0)],
                    imp_states_dn=[('dn', 0)])

# Compute initial thermal state of the reference system
theory.compute_ref_init_state(T, verbosity=1)

# Set local Hubbard interaction and energy level on site 0 after quench
model_ref.local_int[0] = U1
model_ref.hopping[0, 0] = -mu1

# Compute correlation functions of the reference system
print("Computing correlators of the reference system...")
theory.compute_ref_correlators(verbosity=2,
                               hamiltonian_interpol='Trapezoid',
                               lanczos_min_matrix_size=40)

###################### Construct a hybridization function ######################

# Hybridization of impurity site 0 with bath sites 1, 2, 3
print("Computing the hybridization function...")
Delta = model_ref.hybridization(theory.t_mesh, [0], [1, 2, 3], T=T)

########################### Solve D-TRILEX equations ###########################

print("Computing bare dual lines and the vertex...")
theory.compute_bare_lines_vertex(eps_tk, Delta, U_tq, U_dc)

print("Computing dual diagrams...")
theory.compute_diagrams()

# TODO
exit()

K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1, br2 in product(Branch, Branch):
    for k in bz_mesh:
        for time1, time2 in tt_mesh:
            K[br1,br2][time1,time2,k] =  sigma_dual_full[br1,br2][time1,time2,k][:,:] \
                                                    + gref[br1,br2][time1,time2][:,0,:,0]

L = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1, br2 in product(Branch, Branch):
    for k in bz_mesh:
        for time1, time2 in tt_mesh:
            L[br1,br2][time1,time2,k] = gref[br1,br2][time1,time2][:,0,:,0]


K_test = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
for br1, br2 in product(Branch, Branch):
    for k in bz_mesh:
        for time1, time2 in tt_mesh:
            K_test[br1,br2][time1,time2] = gref[br1,br2][time1,time2][:,0,:,0]


Keps = K @ eps_s2p_K
Leps = L @ eps_s2p_K

Kdelta = K @ delta
Ldelta = L @ delta

FG = Kdelta - Keps
FG = 0.5 * (FG + herm_conj(FG)) # for now to circumvent the hermicity check !!!!
LG = Ldelta - Leps
LG = 0.5 * (LG + herm_conj(LG)) # for now to circumvent the hermicity check !!!!

K = 0.5 * (K + herm_conj(K)) # for now to circumvent the hermicity check !!!!
G_latt = solve_vie2(FG, K)
G_latt_CPT = solve_vie2(LG, L) # check because sigma_dual is in F!!!!!

del Keps, Kdelta, FG, L #, K


Gd0_K_full = K_test @ Gd0_reg + K_test @ eps_s2p_K
Gd0_K_full = Gd0_K_full @ K_test


G_latt_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
G_latt_CPT_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
G_latt_mR = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Gd0_full_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1, br2 in product(Branch, Branch):
        for sigm in range(2): # spin up and dn
            GR = G_latt[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
            GR_CPT  = G_latt_CPT[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)

            G_latt_CPT_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(GR_CPT, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R

            G_latt_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(GR, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R
            G_latt_mR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.fftn(GR, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> -R

            Gd0_full_K = Gd0_K_full[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
            Gd0_full_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(Gd0_full_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R

######################## Write results into text files #########################

def write_keldysh_gf_file(filename, g, k_point=None, target_indices=()):
    """
    Write (FW, FW) and (FW, BW) components of a 2-point KeldyshGF
    object to a text file.
    """
    FW, BW = Branch
    t_mesh = g.time_mesh.components[1]
    t0 = next(iter(t_mesh))
    with open(filename, 'w') as file:
        file.write("# Re (FW, FW) Im (FW, FW) Re (FW, BW) Im (FW, BW)\n")
        for t in t_mesh:
            mesh_point = (t0, t) if (k_point is None) else (t0, t, k_point)
            col_data = (g[FW, FW][*mesh_point][*target_indices].real,
                        g[FW, FW][*mesh_point][*target_indices].imag,
                        g[FW, BW][*mesh_point][*target_indices].real,
                        g[FW, BW][*mesh_point][*target_indices].imag)
            file.write("{} {} {} {}\n".format(*col_data))

write_keldysh_gf_file('data/tddt_ref_sys_t0_00.txt',
                      theory.g_ref, target_indices=(0, 0, 0, 0))
write_keldysh_gf_file('data/tddt_ref_sys_t0_01.txt',
                      theory.g_ref, target_indices=(0, 0, 0, 1))

k0, k1 = list(bz_mesh)[:2]
write_keldysh_gf_file("data/tddt_Gd0_R_k0.txt",
                      Gd0_full_R, k_point=k0, target_indices=(0, 0))
write_keldysh_gf_file("data/tddt_Gd0_R_k1.txt",
                      Gd0_full_R, k_point=k1, target_indices=(0, 0))

write_keldysh_gf_file("data/tddt_t0_loc.txt",
                      local_part(G_latt), target_indices=(0, 0))
write_keldysh_gf_file("data/tddt_CPT.txt",
                      local_part(G_latt_CPT), target_indices=(0, 0))
write_keldysh_gf_file("data/K.txt", local_part(K), target_indices=(0, 0))
write_keldysh_gf_file("data/sigma_dual.txt",
                      local_part(sigma_dual_full), target_indices=(0, 0))
write_keldysh_gf_file("data/tddt_01.txt",
                      G_latt_R, k_point=k1, target_indices=(0, 0))
write_keldysh_gf_file("data/tddt_CPT_01.txt",
                      G_latt_CPT_R, k_point=k1, target_indices=(0, 0))

for ki, k in enumerate(bz_mesh):
    write_keldysh_gf_file(f"data/tddt_CPT_k{ki}.txt",
                          G_latt_CPT, k_point=k, target_indices=(0, 0))
    write_keldysh_gf_file(f"data/tddt_T_t0_k{ki}.txt",
                          G_latt, k_point=k, target_indices=(0, 0))

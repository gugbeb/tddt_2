from enum import Enum
from itertools import product
import numpy as np

import triqs.utility.mpi  # noqa: F401
from triqs.gf import MeshReTime, Gf
from triqs.gf import MeshProduct # A direct product of 1D meshes

from realevol.tinterp import TInterp as ti
from realevol.init_state import make_equilibrium_init_state
from realevol.realevol import compute_expectval

#from tddt.keldysh import Branch, KeldyshGF, from_lesser_greater, conv
from tddt.keldysh import Branch, KeldyshGF, conv
from tddt.realevol import (
    compute_keldysh_gf,
    compute_keldysh_gf_element,
    compute_keldysh_correlator_2t,
    compute_keldysh_conn_correlator_2t,
    compute_keldysh_vertex3
)
from tddt.dtrilex import (polarization_2nd_order,
                          selfenergy_2nd_order,
                          selfenergy_2nd_order_hf)

#from tddt.keldysh import Branch, lesser, greater, retarded, advanced
from tddt.keldysh import ContourPoint
from tddt.keldysh import herm_conj
from tddt.keldysh import Singular2PKeldyshGF

# Import some TRIQS modules related to lattice
from triqs.lattice import BravaisLattice, BrillouinZone
from triqs.gf import MeshBrZone

from tddt.lattice import local_part
from tddt.models import SingleFermion, SquarePlaquette
from tddt.vie2 import solve_vie2

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

############################ Time meshes #######################################
t_max = 10.0
n_t = 51
t_mesh = MeshReTime(0, t_max, n_t)
# Time mesh used to construct interpolators
ti_mesh = MeshReTime(0, t_max, 5001)

########################## Lattice problem #####################################

t_nn = 1.0          # Nearest neighbor hopping
t_nnn = 0.0         # Next nearest neighbor hopping

# Vector potential: Amplitude and frequency
A = 0.0
Omega = 4.0
# Time-dependent vector potential (x- and y-component)
Ax_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])
Ay_t = ti(ti_mesh, [A * np.cos(Omega * t) for t in ti_mesh])

########################## Reference system ####################################

# Correlated plaquette site (0)
U = 6.0             # Hubbard interaction at t=0
U1 = 4.0            # Hubbard interaction at t>0
mu = 0.5 * U        # Chemical potential at t=0
mu1 = 0.5 * U1      # Chemical potential at t>0

# Energy levels of uncorrelated plaquette sites are +exx, -exx, ex
ex = 0.0
exx = 3.0

# Hopping amplitudes between site 0 and the uncorrelated sites are txx, txx, tx
tx = 0.7 * t_nn
txx = 1.0 * t_nn

# Temperature
T = 0.0

# Model object
model_ref = SquarePlaquette(
    # 2x2 plaquette
    N=2,
    # Hopping matrix
    hopping=[[-mu, txx, txx,  tx],
             [txx, exx, 0,    0 ],
             [txx, 0,   -exx, 0 ],
             [tx,  0,   0,    ex]],
    # Local interaction
    local_int=[U, 0, 0, 0],
    # Vector potential: Components along the two Cartesian axes
    vector_potential=(Ax_t, Ay_t)
)

# Initial thermal state of the reference system
init_state_ref = model_ref.equilibrium_init_state(T)

# Local Hubbard interaction and energy level on site 0 after quench
model_ref.U[0] = U1
model_ref.t[0, 0] = -mu1

# Reference system GF
params = {}
params['verbosity'] = 2
params['hamiltonian_interpol'] = 'Trapezoid'
params['lanczos_min_matrix_size'] = 40
gf_ref = compute_keldysh_gf(model_ref.gf_struct,
                            init_state_ref,
                            model_ref.hamiltonian,
                            t_mesh,
                            params)

def write_keldysh_gf_file(filename, g, k_point=None, target_indices=()):
    """
    Write (FW, FW) and (FW, BW) components of a 2-point KeldyshGF object to a text file.
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
                      gf_ref['up'], target_indices=(0, 0))
write_keldysh_gf_file('data/tddt_ref_sys_t0_01.txt',
                      gf_ref['up'], target_indices=(0, 1))

tt_mesh = MeshProduct(t_mesh, t_mesh)

# k-mesh
lat = BravaisLattice(units=[(1, 0, 0), (0, 1, 0)])  # 2D square lattice
bz = BrillouinZone(lat)  # Brillouin zone of the lattice
n_k = 2 # Number of k-points along each dimension
bz_mesh = MeshBrZone(bz, n_k) # k-mesh on 1BZ; 0 - 2pi
nkx = n_k
nky = n_k
nkz = 1

# mixed-mesh
tk_mesh = MeshProduct(t_mesh, bz_mesh)
ttk_mesh = MeshProduct(t_mesh, t_mesh, bz_mesh)
tttk_mesh = MeshProduct(t_mesh, t_mesh, t_mesh, bz_mesh)

# TODO: Move to tddt/dtrilex.py
#class Channel(Enum):
#    "Bosonic channel."
#    CHARGE = 0
#    "Charge channel"
#    SPIN = 1
#    "Spin channel"

Uch = U/2
Usp = -U/2
Uch = U1/2
Usp = -U1/2

# Lattice dispersion
#eps_tk = Gf(mesh=tk_mesh, target_shape=(2, 2))

#for t, k in tk_mesh:
#    #eps_tk[t, k] = -2 * np.array([[[1, 0], [0, -1]]]) * np.cos(t.value) * (np.cos(k[0]) + np.cos(k[1]))
#    eps_tk[t, k][0, 0] = 2.0*t1*(np.cos(k[0]-A*np.cos(Omega * t.value))+np.cos(k[1]-A*np.cos(Omega * t.value))) \
#                         +4.0*t2*np.cos(2*(k[0]-A*np.cos(Omega * t.value)))*np.cos(2*(k[1]-A*np.cos(Omega * t.value)))
#    eps_tk[t, k][1, 1] = 2.0*t1*(np.cos(k[0]-A*np.cos(Omega * t.value))+np.cos(k[1]-A*np.cos(Omega * t.value))) \
#                         +4.0*t2*np.cos(2*(k[0]-A*np.cos(Omega * t.value)))*np.cos(2*(k[1]-A*np.cos(Omega * t.value)))

#eps_loc = np.mean(eps_tk.data, axis = 1)

#for t, k in tk_mesh:
#    eps_tk[t, k] = eps_tk[t, k] - eps_loc[t.index, :]

#eps_s2p_K = Singular2PKeldyshGF.from_retime(eps_tk)


# Dispersion reference system
#TODO: compare to dt_pos/ dt_neg
#def V(tx2, axis, sign, t):
#    """ axis = x,y
#        sign = -1,+1
#        tx2  = tx,txx
#    """
#    V = tx2 * np.exp(sign * 1.j * A * np.cos(Omega * t))
#    return V
#for i in t_mesh:
#    print(V(tx,'x', +1, i))

# Non-local interaction
#def Vq(k, channel):
#    Vq = 0.0
#    return Vq


# Bare lattice interaction
#Uq = Gf(mesh=tk_mesh, target_shape=(2, 2))
#Uq_tilde = Gf(mesh=tk_mesh, target_shape=(2, 2))

#for t, k in tk_mesh:
#    Uq[t, k][0,0] = Uch + Vq(k, 0) # charge
#    Uq[t, k][1,1] = Usp + Vq(k, 1) # spin
#    Uq_tilde[t, k][0,0] = Uq[t, k][0,0] - 0.5*Uch
#    Uq_tilde[t, k][1,1] = Uq[t, k][1,1] - 0.5*Usp

#Uq_s2p_K = Singular2PKeldyshGF.from_retime(Uq)
#Uq_tilde_s2p_K = Singular2PKeldyshGF.from_retime(Uq_tilde)

# Bare impurity GF for 3 bath sites
gimp0 = SingleFermion(ex).gf(t_mesh, n=0.5)
gimp0p = SingleFermion(exx).gf(t_mesh, n=0.1)
gimp0m = SingleFermion(-exx).gf(t_mesh, n=0.9)

gimp = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,))) # GF for site 0 of the reference system
gref = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,4), (2,4))) # time, time, spin, site, spin, site for example: (11, 11, 2, 5, 2, 5)

for br1, br2 in product(Branch, Branch):
    gimp[br1,br2][0,0].data[...] = gf_ref['up'][br1,br2].data[...,0,0]
    gimp[br1,br2][1,1].data[...] = gf_ref['dn'][br1,br2].data[...,0,0]
    for i in range(4): # sites
        gref[br1,br2].data[...,0,i,0,i] = gf_ref['up'][br1,br2].data[...,i,i]
        gref[br1,br2].data[...,1,i,1,i] = gf_ref['dn'][br1,br2].data[...,i,i]


# Hybridisation function delta and Dual_reg GF0
delta = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))

for sp in (0,1):
    for br1, br2 in product(Branch, Branch):
        for time1, time2 in tt_mesh:
            z1 = ContourPoint(br1, time1)
            z2 = ContourPoint(br2, time2)
            print('V: ', V(tx,  'x',+1,time1.linear_index))
            print('gimp0: ', gimp0[z1,z2])
            delta[z1,z2][sp,sp] = V(tx,'x',+1,time1.linear_index) * gimp0[z1,z2] * V(tx,'x',-1,time2.linear_index) \
                                    + V(txx,'x',+1,time1.linear_index) * gimp0p[z1,z2] * V(txx,'x',-1,time2.linear_index) \
                                    + V(txx,'x',-1,time1.linear_index) * gimp0m[z1,z2] * V(txx,'x',+1,time2.linear_index)


eps_gimp = eps_s2p_K @ gimp
eps_gimp_eps = eps_gimp @ eps_s2p_K
gimp_eps = gimp @ eps_s2p_K

eps_gimp_delta = eps_gimp @ delta
delta_gimp_eps = delta @ gimp_eps
delta_gimp = delta @ gimp
delta_gimp_delta = delta_gimp @ delta


Q = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1, br2 in product(Branch, Branch):
    for k in bz_mesh:
        #print('k: ',k[0])
        for time1, time2 in tt_mesh:
            Q[br1,br2][time1,time2,k] = - delta[br1,br2][time1,time2] \
                                        + eps_gimp_eps[br1,br2][time1,time2,k] \
                                        - delta_gimp_eps[br1,br2][time1,time2,k]


#  Test print
Q_herm = 0.5 * (Q - herm_conj(Q))

write_keldysh_gf_file("data/Q.txt", local_part(Q), target_indices=(0, 0))
write_keldysh_gf_file("data/Qherm.txt", local_part(Q_herm), target_indices=(0, 0))

Q = 0.5 * (Q + herm_conj(Q)) # for now to circumvent the hermicity check !!!!


F = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1, br2 in product(Branch, Branch):
    for k in bz_mesh:
        for time1, time2 in tt_mesh:
            F[br1,br2][time1,time2,k] = - eps_gimp[br1,br2][time1,time2,k] \
                                        + delta_gimp[br1,br2][time1,time2]


#print('F is Hermitian:', F.is_hermitian())

#F = 0.5 * (F + herm_conj(F)) # for now to circumvent the check !!!!
Gd0_reg = solve_vie2(F, Q)
del F


arg_index_shapes = ((2,), # ch = 0, sp = 1
                    (2,))

# Generator of scalar-valued elements
def generator_susc_imp(ind1, ind2):
    channel1 = ind1
    channel2 = ind2

    def get_operator(ind1, ind2):
        if ind1 == ind2 and ind1 == (0,):
            return (n('up',0) + n('dn',0))
        elif ind1 == ind2 and ind1 == (1,):
            return (n('up',0) - n('dn',0))
        else:
            return None

    operator = get_operator(ind1, ind2)
    #print(operator)
    if operator:
        #print('operator after if: ', operator)
        g_el = -1j * compute_keldysh_conn_correlator_2t(operator,
                                        operator,
                                        init_state,
                                        h,
                                        t_mesh,
                                        params)
        return g_el
    else:
        g_el = KeldyshGF(mesh=tt_mesh)
        return g_el

susc_imp = KeldyshGF.from_arg_index_gen(generator_susc_imp, mesh=tt_mesh, arg_index_shapes=arg_index_shapes)

susc_imp_U = KeldyshGF(mesh=tt_mesh, arg_index_shapes=arg_index_shapes)
for br1, br2 in product(Branch, Branch):
    susc_imp_U[br1,br2].data[:,:,0,0] = susc_imp[br1,br2].data[:,:,0,0]*Uch
    susc_imp_U[br1,br2].data[:,:,1,1] = susc_imp[br1,br2].data[:,:,1,1]*Usp


# Impurity polarization

pi_imp = solve_vie2(susc_imp_U, susc_imp)


# Vertex
arg_index_shapes = ((2,), #((2, 3),    # \sigma_1, l_1
                    (2,), #(2, 3),    # \sigma_2, l_2
                    (2,)) #(4, 3, 3)) # \varsigma, l_3, l_4


# Generator of scalar-valued elements
def generator_three_point_vertex(ind1, ind2, ind3):
    spin1 = ind1
    spin2 = ind2
    channel = ind3

    #def get_operator(ind1, ind2, ind3):
    c_index = ('up', 0) if spin1 == (0,) else ('dn', 0)
    c_dag_index = ('up', 0) if spin2 == (0,) else ('dn', 0)
    n_op = n('up', 0) + n('dn', 0) if channel == (0,) else n('up', 0) - n('dn', 0)

    g_el = compute_keldysh_vertex3(c_index, # c_indices
                                   c_dag_index , # c_dag_indices
                                   n_op, # n_op
                                   init_state,
                                   h,
                                   t_mesh,
                                   params)
    return g_el


three_point_corr = KeldyshGF.from_arg_index_gen(generator_three_point_vertex, mesh=MeshProduct(t_mesh, t_mesh, t_mesh), arg_index_shapes=arg_index_shapes)


U_pi_imp = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,),(2,)))
for br1, br2 in product(Branch, Branch):
    U_pi_imp[br1,br2].data[:,:,0,0] = Uch*pi_imp[br1,br2].data[:,:,0,0]
    U_pi_imp[br1,br2].data[:,:,1,1] = Usp*pi_imp[br1,br2].data[:,:,1,1]

three_point_corr_U_pi_imp = conv(three_point_corr, U_pi_imp,
              [(2, 0)])


Lambda = three_point_corr - three_point_corr_U_pi_imp
del three_point_corr, three_point_corr_U_pi_imp


Uq_piimp = Uq_s2p_K @ pi_imp
Uq_piimp_Uq = Uq_piimp @ Uq_s2p_K

#Uq_piimp = 0.5 * (Uq_piimp + herm_conj(Uq_piimp)) # for now to circumvent the hermicity check !!!!
Uq_piimp_Uq = 0.5 * (Uq_piimp_Uq + herm_conj(Uq_piimp_Uq))
W0prime = solve_vie2(-Uq_piimp, Uq_piimp_Uq)

########################### DIAGRAMS #####################################

########################### prepare diagrams #####################################

#self-energy

eps_s2p_R = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
eps_s2p_mR = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
eps_s2p_loc = Singular2PKeldyshGF(mesh=t_mesh, arg_index_shapes=((2,), (2,)))


for time in t_mesh:
    for sigm in range(2):
        for br in Branch:
            eps_s2p_loc[br][time][sigm,sigm] = np.mean(eps_s2p_K[br1][time,:][sigm,sigm].data)
            eps_K = eps_s2p_K[br][time,:][sigm,sigm].data.reshape(nkx,nky,nkz)

            eps_s2p_R[br].data[time.linear_index,:,sigm,sigm] = np.fft.ifftn(eps_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k -> R
            eps_s2p_mR[br].data[time.linear_index,:,sigm,sigm] = np.fft.fftn(eps_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k+q -> mR

print(eps_s2p_R[Branch.FORWARD].data[0,:,0,0].reshape(nkx,nky,nkz))

k_points = list(bz_mesh)
q0 = k_points[0]


Uq_tilde_s2p_R = Singular2PKeldyshGF(mesh=tk_mesh, arg_index_shapes=((2,), (2,)))
Uq0_tilde = Singular2PKeldyshGF(mesh=t_mesh, arg_index_shapes=((2,), (2,)))
for time in t_mesh:
    for ch in range(2):
        for br in Branch:
            Uq_tilde_K = Uq_tilde_s2p_K[br][time,:][ch,ch].data.reshape(nkx,nky,nkz)
            Uq0_tilde[br][time][ch,ch] = Uq_tilde_s2p_K[br][time,q0][ch,ch]

            Uq_tilde_s2p_R[br].data[time.linear_index,:,ch,ch] = np.fft.ifftn(Uq_tilde_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R


Gd0_reg_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Gd0_reg_mR = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
Gd0_reg_loc = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1, br2 in product(Branch, Branch):
        for sigm in range(2): # spin up and dn
            Gd0_reg_loc[br1,br2][time1,time2][sigm,sigm] = np.mean(Gd0_reg[br1,br2][time1,time2,:][sigm,sigm].data)
            Gd0_reg_K = Gd0_reg[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
            Gd0_reg_R[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(Gd0_reg_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R
            Gd0_reg_mR[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.fftn(Gd0_reg_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> -R


# TODO this won't work if k-mesh does not start from Gamma point
# TODO check what to use for W and Uq np.fft.ifftn or np.fft.fftn (check also for Gd and eps) !!!!!!

########################### diagrams  prepare  END  #####################################

########################### Polarization #####################

Lambdaeps_s2p_R = conv(Lambda, eps_s2p_R,
             [(1, 0)])

eps_s2p_mRLambda = conv(eps_s2p_mR, Lambda,
             [(0, 1)])


LambdaGd0_reg_R = conv(Lambda, Gd0_reg_R,
             [(1, 0)])
Gd0_reg_mRLambda = conv(Gd0_reg_mR, Lambda,
             [(0, 1)])



Pi_R_1 = conv(LambdaGd0_reg_R, Gd0_reg_mRLambda,
             [(0, 0), (2, 1)])

Pi_R_2 = conv(Lambdaeps_s2p_R, Gd0_reg_mRLambda,
             [(0, 0),(2, 1)])

Pi_R_3 = conv(LambdaGd0_reg_R, eps_s2p_mRLambda,
             [(0, 0), (2, 1)])

Pi_R_4 = conv(Lambdaeps_s2p_R, eps_s2p_mRLambda,
             [(0, 0), (2, 1)])



Pi_R = -1j*(Pi_R_1 + Pi_R_2 + Pi_R_3 + Pi_R_4)

del LambdaGd0_reg_R, Lambdaeps_s2p_R, Gd0_reg_mRLambda, eps_s2p_mRLambda
del Pi_R_1, Pi_R_2, Pi_R_3, Pi_R_4

Pi_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1, br2 in product(Branch, Branch):
        for sigm in range(2): # spin up and dn
            Pi = Pi_R[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
            Pi_K[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(Pi, axes=(0,1,2)).reshape(nkx*nky*nkz) # K -> R

########################### Polarization END #####################

########################### Full W #####################
W0prime_Pi_K = W0prime @ Pi_K
Uq_tilde_s2p_K_Pi_K = Uq_tilde_s2p_K @ Pi_K

mFW = Uq_tilde_s2p_K_Pi_K
mQW = mFW @ Uq_tilde_s2p_K
QW = W0prime - mQW

QW = -0.5 * (QW + herm_conj(QW)) # for now to circumvent the hermicity check !!!!
mFW = 0.5 * (mFW + herm_conj(mFW)) # for now to circumvent the hermicity check !!!!

Wprime = solve_vie2(-mFW, QW)

del mFW, mQW, QW, W0prime_Pi_K, Uq_tilde_s2p_K_Pi_K


Wprime_R = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
W0prime_q0 = KeldyshGF(mesh=tt_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1, br2 in product(Branch, Branch):
        for ch in range(2): # channel charge and spin
            W0prime_q0[br1,br2][time1,time2][ch,ch] = W0prime[br1,br2][time1,time2,q0][ch,ch]
            Wprime_K = Wprime[br1,br2][time1,time2,:][ch,ch].data.reshape(nkx,nky,nkz)
            Wprime_R[br1,br2].data[time1.linear_index,time2.linear_index,:,ch,ch] = np.fft.ifftn(Wprime_K, axes=(0,1,2)).reshape(nkx*nky*nkz) # k+q -> R

########################### Full W END #####################
########################### Self-Energy #####################################

LambdaGd0_reg_mR = conv(Lambda, Gd0_reg_mR,
             [(1, 0)])

Wprime_RLambda = conv(Wprime_R, Lambda,
             [(1, 2)])

Lambdaeps_s2p_mR = conv(Lambda, eps_s2p_mR,
             [(1, 0)])
Uq_tilde_s2p_RLambda = conv(Uq_tilde_s2p_R, Lambda,
             [(1, 2)])

sigma_R_1 = conv(LambdaGd0_reg_mR, Wprime_RLambda,
             [(1, 0), (2, 1)])

sigma_R_2 = conv(Lambdaeps_s2p_mR, Wprime_RLambda,
               [(1, 0), (2, 1)])

sigma_R_3 = conv(LambdaGd0_reg_mR, Uq_tilde_s2p_RLambda,
               [(1, 0), (2, 1)])

sigma_R_4 = conv(Lambdaeps_s2p_mR, Uq_tilde_s2p_RLambda,
               [(1, 0), (2, 1)])

sigma_R = 1j * (sigma_R_1 + sigma_R_2 + sigma_R_3 + sigma_R_4)

del LambdaGd0_reg_mR, Wprime_RLambda, Lambdaeps_s2p_mR, Uq_tilde_s2p_RLambda
del sigma_R_1, sigma_R_2, sigma_R_3,sigma_R_4

sigma_dual_K = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for time1, time2 in tt_mesh:
    for br1, br2 in product(Branch, Branch):
        for sigm in range(2): # spin up and dn
            sigma = sigma_R[br1,br2][time1,time2,:][sigm,sigm].data.reshape(nkx,nky,nkz)
            sigma_dual_K[br1,br2].data[time1.linear_index,time2.linear_index,:,sigm,sigm] = np.fft.ifftn(sigma, axes=(0,1,2)).reshape(nkx*nky*nkz) # R -> K

del sigma_R

############# Tadpole #####################

#Lambdaeps_s2p_loc = conv(Lambda, eps_s2p_loc,
#             [(0, 1),(1, 0)])

LambdaUq0_tilde = conv(Lambda, Uq0_tilde,
             [(2, 0)])

LambdaW0prime_q0 = conv(Lambda, W0prime_q0,
             [(2, 0)])

LambdaGd0_reg_loc = conv(Lambda, Gd0_reg_loc,
             [(0, 1),(1, 0)])


sigma_tadpole_1 = conv(LambdaW0prime_q0, LambdaGd0_reg_loc,
             [(2, 0)])

#sigma_tadpole_2 = conv(LambdaWprime_q0, Lambdaeps_s2p_loc,
#             [(2, 0)])

sigma_tadpole_3 = conv(LambdaUq0_tilde, LambdaGd0_reg_loc,
             [(2, 0)])

#sigma_tadpole_4 = conv(LambdaUq0_tilde, Lambdaeps_s2p_loc,
#             [(2, 0)])

sigma_tadpole = -1j*(sigma_tadpole_1 + sigma_tadpole_3)
#sigma_tadpole = -1j*(sigma_tadpole_1 + sigma_tadpole_2 + sigma_tadpole_3 + sigma_tadpole_4)

del LambdaW0prime_q0, LambdaUq0_tilde, LambdaGd0_reg_loc #, Lambdaeps_s2p_loc

############# Tadpole END #####################

############# Full Self-Energy #####################

print(sigma_dual_K.mesh)
print(sigma_tadpole.mesh)

sigma_dual_full = KeldyshGF(mesh=ttk_mesh, arg_index_shapes=((2,), (2,)))
for br1, br2 in product(Branch, Branch):
    for sigm in range(2): # spin up and dn
        for time1, time2 in tt_mesh:
            for k in bz_mesh:
                sigma_dual_full[br1,br2][time1,time2,k][sigm,sigm] = sigma_dual_K[br1,br2][time1,time2,k][sigm,sigm] \
                                                                        + 0.0 * sigma_tadpole[br1,br2][time1,time2][sigm,sigm] #!!!!! tadpole is set to zero !!!!!

del sigma_dual_K, sigma_tadpole
############# Full Self-Energy END #####################

########################### Self-Energy END #####################################

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


k0 = list(bz_mesh)[0]
k1 = list(bz_mesh)[1]
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

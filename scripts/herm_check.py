# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# ##############################################################################

r"""
Keldysh hermiticity diagnostic, same single check as
programs/debug_hermiticity.py.

The hermiticity condition for a 2-point Keldysh GF (NESSi convention, the one
implemented by tddt.keldysh.herm_conj) is

    G = herm_conj(G)    <=>    G^{ab}_{ij}(t,t') = -[G^{ba}_{ji}(t',t)]^*

Note the MINUS sign and the transposition of the two target-index groups.
Because herm_conj() rebuilds its result from the lesser/greater components,
comparing all four Keldysh blocks against it in one shot covers both the
G^<, G^> relations and the G^FF <-> G^BB one.

Violation is measured as

    viol(G) = max_{a,b,t,t'} |G^{ab}(t,t') - herm_conj(G)^{ab}(t,t')|

For a quantity that never went through a contour convolution (pure ED output)
this is ~1e-14 or below. Anything built with conv() / solve_vie2() carries
Gregory quadrature error on top, which converges as O(dt^5..dt^6).

For a function carrying an extra momentum argument the relation holds k by k,
so this may only be used on k-space (or purely local) quantities -- in real
space it connects r and -r.
"""

import numpy as np

from tddt.keldysh import Branch, KeldyshGF, Singular2PKeldyshGF, herm_conj

# (name, absolute violation, relative violation), filled by herm_viol()
_RECORDS = []


def herm_viol(G, name: str) -> float:
    """
    Print and return max|G - herm_conj(G)| over all four Keldysh blocks.

    The relative violation (divided by max|G|) is printed alongside, since the
    absolute number alone is not comparable between quantities whose scales
    differ by orders of magnitude.
    """
    if isinstance(G, Singular2PKeldyshGF) or not isinstance(G, KeldyshGF) \
            or G.n_args != 2 \
            or G.time_mesh.components[0] != G.time_mesh.components[1] \
            or G.arg_index_shapes[0] != G.arg_index_shapes[1]:
        return float('nan')

    G_hc = herm_conj(G)
    mx = 0.0
    scale = 0.0
    for b1, b2 in ((Branch.FORWARD, Branch.FORWARD),
                   (Branch.FORWARD, Branch.BACKWARD),
                   (Branch.BACKWARD, Branch.FORWARD),
                   (Branch.BACKWARD, Branch.BACKWARD)):
        mx = max(mx, float(np.max(np.abs(G[b1, b2].data - G_hc[b1, b2].data))))
        scale = max(scale, float(np.max(np.abs(G[b1, b2].data))))

    rel = mx / scale if scale > 0 else float('nan')
    # flush: tddt.vie2 emits UserWarnings on stderr, which otherwise splice
    # themselves into the middle of these lines when both are redirected to
    # the same file.
    print(f"  {name:<24s}  max|G - G†| = {mx:.3e}   (rel {rel:.3e})",
          flush=True)

    _RECORDS.append((name, mx, rel))
    return mx


def print_summary():
    """Print all violations recorded so far, worst (relative) first."""
    if not _RECORDS:
        return
    print()
    print("-" * 60)
    print("Hermiticity summary  (worst first)")
    print("-" * 60)
    for name, mx, rel in sorted(_RECORDS, key=lambda r: -r[2]):
        print(f"  {name:<24s}  {mx:.3e}   (rel {rel:.3e})")
    print("-" * 60)


def reset():
    _RECORDS.clear()

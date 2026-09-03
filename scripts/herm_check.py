# ##############################################################################
#
# tddt - Implementation of the time-dependent dual TRILEX theory
#
# ##############################################################################

r"""
Diagnostics: Keldysh hermiticity ("conjugation symmetry") checks.

Conventions used in `tddt.keldysh`
----------------------------------
`KeldyshGF.lesser()`  == G[FORWARD,  BACKWARD]   (Aoki RMP Eq. (17), G^{12})
`KeldyshGF.greater()` == G[BACKWARD, FORWARD]    (                   G^{21})

For a 2-point function with a prefactor -i in front of the correlator (the
convention of this code, see `herm_conj()` / `KeldyshGF.is_hermitian()`), the
exact relations are ANTI-hermitian, i.e. they carry a MINUS sign and a
transposition of the left/right target-index groups:

    G^<_{ab}(t, t') = - [ G^<_{ba}(t', t) ]^*        (L)
    G^>_{ab}(t, t') = - [ G^>_{ba}(t', t) ]^*        (G)
    G^{FF}_{ab}(t, t') = - [ G^{BB}_{ba}(t', t) ]^*  (T)

(T) follows from (L)+(G) and the definitions
    G^{FF} = theta(t-t') G^> + theta(t'-t) G^<   (time ordered),
    G^{BB} = theta(t'-t) G^> + theta(t-t') G^<   (anti-time ordered).

`is_hermitian()` in tddt/keldysh.py checks (L) and (G) only; (T) is an
independent statement at coinciding times t = t', where a single stored number
cannot represent both one-sided limits of the G^> - G^< jump.

For a function carrying an extra momentum argument k the relations hold
k-by-k; for a real-space argument r they connect r and -r, so this checker
must only be used on k-space (or purely local) quantities.

Caveat for bosonic 3-index groups (channel, i, j): the adjoint of the density
operator n_{ij} = c^+_i c_j is n_{ji}, so for Nimp > 1 the exact relation for
chi/Pi/W additionally swaps i <-> j inside each index group.  For Nimp == 1
(the 2x2-plaquette runs) that is a no-op and the generic check below applies
verbatim.
"""

import numpy as np

from tddt.keldysh import Branch, KeldyshGF, Singular2PKeldyshGF

# Collected results, filled by check_herm(); printed by print_summary()
_RECORDS = []


def _swap(data, nli, nri):
    """
    (t, t') -> (t', t) together with the swap of the left and right groups of
    target indices.  Non-time mesh axes (e.g. k) stay in place; that is why
    only negative target-axis indices are used.
    """
    axes_from = (0, 1, *range(-1, -nli - nri - 1, -1))
    axes_to = (1, 0,
               *range(-nli - 1, -nli - nri - 1, -1),
               *range(-1, -nli - 1, -1))
    return np.moveaxis(data, axes_from, axes_to)


def _loc(a, mesh_shape):
    """Human-readable position of the largest entry of |a|."""
    idx = tuple(int(i) for i in
                np.unravel_index(np.argmax(np.abs(a)), a.shape))
    nt = len(mesh_shape)
    return f"(it,it')=({idx[0]},{idx[1]}) mesh={idx[2:nt]} tgt={idx[nt:]}"


def _residuals(x, y_swapped):
    """
    Return (anti-hermitian residual, hermitian residual):
      anti:  x + conj(y_swapped)   -> zero if x_{ab}(t,t') = -[y_{ba}(t',t)]^*
      herm:  x - conj(y_swapped)   -> zero if x_{ab}(t,t') = +[y_{ba}(t',t)]^*
    """
    c = np.conj(y_swapped)
    return x + c, x - c


def check_herm(name, G, *, verbose=True, indent=""):
    r"""
    Report the violation of the Keldysh conjugation relations for a 2-point
    KeldyshGF `G`.  Returns a dict with the absolute/relative violations.
    """
    if isinstance(G, Singular2PKeldyshGF):
        if verbose:
            print(f"{indent}[herm] {name:<28} skipped "
                  f"(Singular2PKeldyshGF: delta_C(t,t') contact term)")
        return None
    if not isinstance(G, KeldyshGF):
        if verbose:
            print(f"{indent}[herm] {name:<28} skipped "
                  f"(not a KeldyshGF: {type(G).__name__})")
        return None
    if G.n_args != 2:
        if verbose:
            print(f"{indent}[herm] {name:<28} skipped "
                  f"({G.n_args}-point function)")
        return None
    if G.time_mesh.components[0] != G.time_mesh.components[1]:
        if verbose:
            print(f"{indent}[herm] {name:<28} skipped (non-square time mesh)")
        return None
    if G.arg_index_shapes[0] != G.arg_index_shapes[1]:
        if verbose:
            print(f"{indent}[herm] {name:<28} skipped "
                  f"(left/right index shapes differ: {G.arg_index_shapes})")
        return None

    nli, nri = map(len, G.arg_index_shapes)

    d_l = G.lesser().data
    d_g = G.greater().data
    d_ff = G[Branch.FORWARD, Branch.FORWARD].data
    d_bb = G[Branch.BACKWARD, Branch.BACKWARD].data

    scale = max(np.max(np.abs(d)) for d in (d_l, d_g, d_ff, d_bb))
    norm = scale if scale > 0 else 1.0

    mesh_shape = d_l.shape[:2 + len(G.non_time_mesh.components)]

    # Mask selecting the equal-time diagonal t = t' (broadcast over the
    # remaining axes).  The FF/BB relation is trivially broken there by the
    # G^> - G^< jump whenever the two blocks store different one-sided limits,
    # so the diagonal is reported separately from the genuine bulk violation.
    n_t = d_l.shape[0]
    diag = np.zeros(d_l.shape[:2], dtype=bool)
    diag[np.arange(n_t), np.arange(n_t)] = True
    diag = diag.reshape(diag.shape + (1,) * (d_l.ndim - 2))

    checks = (
        ("G^<(t,t') vs G^<(t',t)*", d_l, _swap(d_l, nli, nri), None),
        ("G^>(t,t') vs G^>(t',t)*", d_g, _swap(d_g, nli, nri), None),
        ("G^FF(t,t') vs G^BB(t',t)*", d_ff, _swap(d_bb, nli, nri), "offdiag"),
        ("  ^ same, on t=t' diagonal", d_ff, _swap(d_bb, nli, nri), "diag"),
    )

    res = {"name": name, "scale": scale}
    lines = []
    for label, x, y, sel in checks:
        anti, herm = _residuals(x, y)
        if sel == "offdiag":
            anti = np.where(diag, 0.0, anti)
            herm = np.where(diag, 0.0, herm)
        elif sel == "diag":
            anti = np.where(diag, anti, 0.0)
            herm = np.where(diag, herm, 0.0)
        a_abs, h_abs = np.max(np.abs(anti)), np.max(np.abs(herm))
        res[label] = (a_abs / norm, h_abs / norm, a_abs)
        which = "-" if a_abs <= h_abs else "+"
        lines.append(
            f"{indent}    {label:<28} "
            f"rel[-] = {a_abs / norm:9.3e}   rel[+] = {h_abs / norm:9.3e}"
            f"   sign={which}   abs[-] = {a_abs:9.3e}"
            f"   worst @ {_loc(anti, mesh_shape)}"
        )

    if verbose:
        print(f"{indent}[herm] {name:<28} "
              f"data{tuple(d_l.shape)}  max|G| = {scale:.4e}")
        for ln in lines:
            print(ln)

    _RECORDS.append(res)
    return res


def print_summary(title="HERMITICITY SUMMARY"):
    """Print a compact table of all checks performed so far, worst first."""
    if not _RECORDS:
        return
    keys = ("G^<(t,t') vs G^<(t',t)*",
            "G^>(t,t') vs G^>(t',t)*",
            "G^FF(t,t') vs G^BB(t',t)*",
            "  ^ same, on t=t' diagonal")
    W = 110
    print()
    print("=" * W)
    print(title + "   (relative violation of the MINUS-sign rule "
                  "X_ab(t,t') = -[Y_ba(t',t)]*)")
    print("=" * W)
    print(f"{'function':<30}{'max|G|':>12}"
          f"{'lesser':>13}{'greater':>13}{'FF/BB bulk':>13}"
          f"{'FF/BB diag':>13}   verdict")
    print("-" * W)

    def worst(r):
        return max(r[k][0] for k in keys[:3])

    for r in sorted(_RECORDS, key=worst, reverse=True):
        w = worst(r)
        verdict = ("OK" if w < 1e-10 else
                   "ok(1e-10..1e-8)" if w < 1e-8 else
                   "WARN" if w < 1e-4 else
                   "VIOLATED")
        print(f"{r['name']:<30}{r['scale']:>12.3e}"
              f"{r[keys[0]][0]:>13.3e}{r[keys[1]][0]:>13.3e}"
              f"{r[keys[2]][0]:>13.3e}{r[keys[3]][0]:>13.3e}   {verdict}")
    print("-" * W)
    print("'FF/BB diag' is a pure convention artifact of "
          "from_lesser_greater(): at t=t' the FF block stores one one-sided "
          "limit\nand the BB block the other, so the residual there equals "
          "the G^> - G^< jump. It is NOT an error.")
    print("=" * W)


def reset():
    _RECORDS.clear()

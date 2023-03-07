#
# Numerical integration tools
#

from numpy import ones

from triqs.gf import MeshReTime


def simpsons_weights(mesh: MeshReTime):
    """
    Make a list of Simpson’s rule weights for integration on a real time mesh.
    """
    w = ones(len(mesh))
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    w *= mesh.delta / 3
    return w

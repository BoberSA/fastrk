from numba import cfunc
import numpy as np


@cfunc('f8[:](f8, f8[:], f8[:])', nopython=True, nogil=True, fastmath=True, cache=True)
def crtbp(t, s, mc):
    """
    Circular Restricted Three Body Problem ODE right part
    :param t: time
    :param s: state
    :param mc: model constant mu2
    :return:
    """
    mu2 = mc[0]
    mu1 = 1 - mu2

    x, y, z, vx, vy, vz = s[:6]

    yz2 = y * y + z * z
    r13 = ((x + mu2) * (x + mu2) + yz2) ** (-1.5)
    r23 = ((x - mu1) * (x - mu1) + yz2) ** (-1.5)

    mu12r12 = (mu1 * r13 + mu2 * r23)

    ax = 2 * vy + x - (mu1 * (x + mu2) * r13 + mu2 * (x - mu1) * r23)
    ay = -2 * vx + y - mu12r12 * y
    az = - mu12r12 * z

    out = np.array([vx, vy, vz, ax, ay, az])
    return out

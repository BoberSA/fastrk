from numba import cfunc, jit
import numpy as np


@cfunc('f8[:](f8, f8[:], f8[:])', nopython=True, nogil=True, fastmath=True, cache=True, boundscheck=False)
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


@cfunc('f8[:](f8, f8[:], f8[:])', nopython=True, nogil=True, fastmath=True, cache=True, boundscheck=False)
def crtbp_stm(t, s, mc):
    ds = np.zeros(42)
    x = s[0]
    y = s[1]
    z = s[2]
    vx = s[3]
    vy = s[4]
    vz = s[5]
    stm = np.ascontiguousarray(s[6:42])
    ds[0] = s[3]
    ds[1] = s[4]
    ds[2] = s[5]
    mu2 = mc[0]
    mu1 = 1 - mu2

    yz2 = y * y + z * z
    xmu2 = x + mu2
    xmu1 = x - mu1
    r1 = xmu2 ** 2 + yz2
    r1s = r1 ** 0.5
    r2 = xmu1 ** 2 + yz2
    r2s = r2 ** 0.5
    r13 = 1.0 / (r1 * r1s)
    r23 = 1.0 / (r2 * r2s)

    mu12r12 = (mu1 * r13 + mu2 * r23)

    ds[3] += 2 * vy + x - (mu1 * xmu2 * r13 + mu2 * xmu1 * r23)
    ds[4] += -2 * vx + y - mu12r12 * y
    ds[5] += - mu12r12 * z

    r15 = r13 / r1
    r25 = r23 / r2

    Uxx = 1. - mu12r12 + 3 * mu1 * xmu2 ** 2 * r15 + 3 * mu2 * xmu1 ** 2 * r25
    Uxy = 3 * mu1 * xmu2 * y * r15 + 3 * mu2 * xmu1 * y * r25
    Uxz = 3 * mu1 * xmu2 * z * r15 + 3 * mu2 * xmu1 * z * r25
    Uyy = 1. - mu12r12 + 3 * mu1 * y ** 2 * r15 + 3 * mu2 * y ** 2 * r25
    Uyz = 3 * mu1 * y * z * r15 + 3 * mu2 * y * z * r25
    Uzz = -mu12r12 + 3 * mu1 * z ** 2 * r15 + 3 * mu2 * z ** 2 * r25

    stm1 = np.empty((6, 6))
    istm = stm.reshape((6, 6))
    stm1[:3] = istm[3:]

    # STM CALCULATION
    stm1[3, 0] = Uxx * istm[0, 0] + Uxy * istm[1, 0] + Uxz * istm[2, 0] + 2.0 * istm[4, 0]
    stm1[3, 1] = Uxx * istm[0, 1] + Uxy * istm[1, 1] + Uxz * istm[2, 1] + 2.0 * istm[4, 1]
    stm1[3, 2] = Uxx * istm[0, 2] + Uxy * istm[1, 2] + Uxz * istm[2, 2] + 2.0 * istm[4, 2]
    stm1[3, 3] = Uxx * istm[0, 3] + Uxy * istm[1, 3] + Uxz * istm[2, 3] + 2.0 * istm[4, 3]
    stm1[3, 4] = Uxx * istm[0, 4] + Uxy * istm[1, 4] + Uxz * istm[2, 4] + 2.0 * istm[4, 4]
    stm1[3, 5] = Uxx * istm[0, 5] + Uxy * istm[1, 5] + Uxz * istm[2, 5] + 2.0 * istm[4, 5]
    stm1[4, 0] = Uxy * istm[0, 0] + Uyy * istm[1, 0] + Uyz * istm[2, 0] - 2.0 * istm[3, 0]
    stm1[4, 1] = Uxy * istm[0, 1] + Uyy * istm[1, 1] + Uyz * istm[2, 1] - 2.0 * istm[3, 1]
    stm1[4, 2] = Uxy * istm[0, 2] + Uyy * istm[1, 2] + Uyz * istm[2, 2] - 2.0 * istm[3, 2]
    stm1[4, 3] = Uxy * istm[0, 3] + Uyy * istm[1, 3] + Uyz * istm[2, 3] - 2.0 * istm[3, 3]
    stm1[4, 4] = Uxy * istm[0, 4] + Uyy * istm[1, 4] + Uyz * istm[2, 4] - 2.0 * istm[3, 4]
    stm1[4, 5] = Uxy * istm[0, 5] + Uyy * istm[1, 5] + Uyz * istm[2, 5] - 2.0 * istm[3, 5]
    stm1[5, 0] = Uxz * istm[0, 0] + Uyz * istm[1, 0] + Uzz * istm[2, 0]
    stm1[5, 1] = Uxz * istm[0, 1] + Uyz * istm[1, 1] + Uzz * istm[2, 1]
    stm1[5, 2] = Uxz * istm[0, 2] + Uyz * istm[1, 2] + Uzz * istm[2, 2]
    stm1[5, 3] = Uxz * istm[0, 3] + Uyz * istm[1, 3] + Uzz * istm[2, 3]
    stm1[5, 4] = Uxz * istm[0, 4] + Uyz * istm[1, 4] + Uzz * istm[2, 4]
    stm1[5, 5] = Uxz * istm[0, 5] + Uyz * istm[1, 5] + Uzz * istm[2, 5]

    ds[6:42] = stm1.ravel()

    return ds
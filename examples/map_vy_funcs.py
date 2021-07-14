from numba import jit, prange
import numpy as np
from fastrk import BT8713M, RKCodeGen

# Generate/load module to propagate ODE and calculate events
rk_module = RKCodeGen(BT8713M, autonomous=True).save_and_import()
rk_prop_ev = rk_module.rk_prop_ev


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def get_border(x, z, vy, L, R, fode, call_event, rtol, atol):
    """propagate spacecraft up to x == L or x == R"""
    s = np.zeros(6, dtype=np.float64)
    s[0] = x
    s[2] = z
    s[4] = vy
    values = np.array([L, R], dtype=np.float64)
    directions = np.array([0, 0])
    terminals = np.array([True, True])
    accurates = np.array([False, False])
    counts = np.array([-1, -1])
    mc = np.array([3.001348389698916e-06])

    arr, ev = rk_prop_ev(fode, s, 0., 100., np.inf, rtol, atol,
                         values, terminals, directions, counts, accurates,
                         call_event, atol, rtol, 100, mc)

    return 1 if ev[-1, 3] < (L + R)*0.5 else -1


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def calc_vy(x, z, dv0, L, R, fode, call_event, rtol, atol):
    """calculate break point of get_border(vy) function"""
    dv = dv0
    # root separation
    for i in range(100):
        pa = get_border(x, z, -dv, L, R, fode, call_event, rtol, atol)
        pb = get_border(x, z, dv, L, R, fode, call_event, rtol, atol)
        if pa == pb:
            dv *= 10
        else:
            break

    # bisection
    a = -dv
    b = dv
    while abs(a - b) > atol:
        c = 0.5 * (a + b)
        pc = get_border(x, z, c, L, R, fode, call_event, rtol, atol)
        if pa == pc:
            a = c
        else:
            b = c

    return c

@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def calc_vy_map_parallel(x, z, dv0, L, R, fode, call_event, rtol, atol):
    """
    calculate vy for all (x, z) pairs (numba's openmp version)
    """

    nx = x.size
    nz = z.size
    res = np.empty((nx, nz), dtype=np.float64)

    for i in prange(nx):
        for j in range(nz):
            res[i, j] = calc_vy(x[i], z[j], dv0, L, R, fode, call_event, rtol, atol)

    return res

@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def calc_vy_map_sequential(x, z, dv0, L, R, fode, call_event, rtol, atol):
    """
    calculate vy for all (x, z) pairs (sequential version)
    """

    nx = x.size
    nz = z.size
    res = np.empty((nx, nz), dtype=np.float64)

    for i in range(nx):
        for j in range(nz):
            res[i, j] = calc_vy(x[i], z[j], dv0, L, R, fode, call_event, rtol, atol)

    return res

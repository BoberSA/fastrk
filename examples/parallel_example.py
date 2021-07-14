import numpy as np
import matplotlib.pyplot as plt
from fastrk import EventsCodeGen
from timeit import time

# CRTBP ODE, https://github.com/BoberSA/fastrk/tree/master/examples/model_crtbp.py
from model_crtbp import crtbp

# jitted functions that calculate map of Vy for each state
from map_vy_funcs import (calc_vy_map_parallel,
                          calc_vy_map_sequential)

#%%

# event function; becomes zero when x == value
def eventX(t, s, value):
    return s[0] - value

# two identical events that will be used with different [value] argument
events = [eventX, eventX]

# generate call_event compiled @cfunc callback
call_event = EventsCodeGen(events).save_and_import()

#%% constants
L1 = 0.9900289479871328
mu = 3.001348389698916e-06  # CRTBP constant
ER = 149600000.0
maxt = 100
left  = L1 - 1.4e6/ER
right = L1 + 1.4e6/ER

rtol, atol = 1e-12, 1e-12  # propagation tolerances

#%% Generate mesh of initial (x,z)-coordinates

nx = 50
nz = 50
x = np.linspace(L1 - 1.0e6/ER, L1 + 1.0e6/ER, nx)
z = np.linspace(0., 1.0e6/ER, nz)

#%%
print('compile sequential...')
calc_vy_map_sequential(x[:1], z[:1], 0.1, left, right, crtbp, call_event, rtol, atol)

print('compile parallel...')
calc_vy_map_parallel(x[:1], z[:1], 0.1, left, right, crtbp, call_event, rtol, atol)

#%% Run sequential calculations

print(f'states count: {nx * nz}')
print('sequential calculation started (Ms)')
t = time.time()
Ms = calc_vy_map_sequential(x, z, 0.1, left, right, crtbp, call_event, rtol, atol)
ts = time.time() - t
print(f'map calculation time: {ts:0.2f} s')

#%% Run parallel calculations

print(f'states count: {nx * nz}')
print('parallel calculation started (Mp)')
t = time.time()
Mp = calc_vy_map_parallel(x, z, 0.1, left, right, crtbp, call_event, rtol, atol)
tp = time.time() - t
print(f'map calculation time: {tp:0.2f} s')

print(f'speedup x {ts/tp:.2f}')

#%%

print('||Ms - Mp|| =', np.linalg.norm(Ms - Mp))

#%%

plt.pcolormesh(x, z, Mp.T, shading='auto', cmap='jet')
plt.colorbar()
plt.show();







# t0, t1 = 0., 3 * np.pi
# initial state for halo orbit
# s0 = np.zeros(6)
# s0[[0, 2, 4]] = 9.949942666080747733e-01, 4.732924802139452415e-03, -1.973768492871211949e-02

# vy = calc_vy(s0[0], s0[2], 0.1, left, right, 1e-16, crtbp, call_event)
# print(vy - s0[4])

# integrate CRTBP ODE from t0, s0 to t1
# arr = rk_prop(crtbp, s0, t0, t1, np.inf, rtol, atol, mc)
#
# plt.plot(arr[:, 1], arr[:, 2], 'r')
# plt.axis('equal')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from fastrk import BT8713M, RKCodeGen

# CRTBP ODE, https://github.com/BoberSA/fastrk/tree/master/examples/model_crtbp.py
from model_crtbp import crtbp

rk_module = RKCodeGen(BT8713M, autonomous=True).save_and_import()
rk_prop = rk_module.rk_prop

t0, t1 = 0., 3 * np.pi
# initial state for halo orbit
s0 = np.zeros(6)
s0[[0, 2, 4]] = 9.949942666080747733e-01, 4.732924802139452415e-03, -1.973768492871211949e-02
mc = np.array([3.001348389698916e-06])  # CRTBP constant
rtol, atol = 1e-12, 1e-12

# integrate CRTBP ODE from t0, s0 to t1
arr = rk_prop(crtbp, s0, t0, t1, np.inf, rtol, atol, mc)

plt.plot(arr[:, 1], arr[:, 2], 'r')
plt.axis('equal')
plt.show()
# FastRK

- developed as **fast alternative** for subset of [scipy.integrate.ode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) methods (i.e. DOP853);
- is a python code generator for Ordinary Differential Equations (ODE) propagation;
- uses explicit **embedded Runge-Kutta** (ERK) methods with adaptive step technique;
- calculates events using **event functions** (like [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)); 
- is **jit-compiled** by [numba](https://pypi.org/project/numba/);
  - compiled code **cached** on SSD/HDD to prevent unnecessary recompilation;
- reentry, i.e. can be used in **multithreaded** applications;
- OS-independent (to the same extent as [numba](https://pypi.org/project/numba/));
- contains Butcher Tables for several ERK methods:
    - Dormand and Prince 6(5)8M;
    - Dormand and Prince 8(7)13M;
      - **>2x faster** than `DOP853` from `scipy.integrate.ode`
    - Verner's 8(9)16;
- user-defined Butcher Tables also supported; 
- generated code is open and user-modifiable;

Butcher Tables was adapted from [TrackerComponentLibrary](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary).

## Installation

    pip install fastrk

## Fast example

```python
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

```

## Detailed examples

#### [Example 0: Propagate spacecraft motion in Circular Restricted Three Body Problem](https://github.com/BoberSA/fastrk/blob/master/examples/ex0_propagate_crtbp.ipynb)

#### [Example 1: Calculate events in Circular Restricted Three Body Problem](https://github.com/BoberSA/fastrk/blob/master/examples/ex1_calculate_events.ipynb)

Required modules:
- [model_crtbp.py](https://github.com/BoberSA/fastrk/blob/master/examples/model_crtbp.py)

#### [Parallel (OpenMP) example](https://github.com/BoberSA/fastrk/blob/master/examples/parallel_example.py)

Required modules:
- [model_crtbp.py](https://github.com/BoberSA/fastrk/blob/master/examples/model_crtbp.py)
- [map_vy_funcs.py](https://github.com/BoberSA/fastrk/blob/master/examples/map_vy_funcs.py)

Output for `Ryzen 7 4700U` CPU 

```    
    compiling sequential...
    compiling parallel...
    states count: 2500
    sequential calculation started (Ms)
    map calculation time: 46.31 s
    parallel calculation started (Mp)
    map calculation time: 22.19 s
    speedup x 2.09
    ||Ms - Mp|| = 0.0
```

## Core Developer
Stanislav Bober, [MIEM NRU HSE](https://miem.hse.ru/), [IKI RAS](http://iki.rssi.ru/)

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
      - **1.5x - 4.5x faster** than `DOP853` from `scipy.integrate.ode`
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

## Performance measurement

```python
import numpy as np
import pandas as pd
from scipy.integrate import ode
from numba import jit
from timeit import timeit
from fastrk import BT8713M, RKCodeGen, default_jitkwargs
from model_crtbp import crtbp
rk_prop = RKCodeGen(BT8713M, autonomous=False).save_and_import().rk_prop
#%%
# scipy.integrate.ode's fortran implementation of DOP853 works
# significantly faster with @jit-compiled function rather than @cfunc-compiled
# default_jitkwargs = {'nopython': True, 'nogil': True, 'fastmath': True, 'cache': True}
jit_crtbp = jit(**default_jitkwargs)(crtbp._pyfunc).compile('f8[:](f8, f8[:], f8[:])')

#%% integration parameters
rtol, atol = 1e-12, 1e-12
max_step = np.inf

#%% CTRBP constant
mc = np.array([3.001348389698916e-06])  # CRTBP constant

#%% initial states
s0 = [0.9919060293647325, 0., 0.0016194537148125807, 0., -0.010581643111837302, 0.]
s1 = [0.9966271059324971, 0., 0.0050402173579027045, 0., -0.024398902561093703, 0.]
s2 = [0.4999857344807682, -0.866005893551121, 0., 3.902066111769351e-05, 2.252789194673211e-05, 0.]
s3 = [0.4966615415563801, -0.8602481879501589, 0., 0.011597147217611577, 0.0066415463209149195, 0.]

#%% tests
#         name                 t          initial state
tests = {'halo (small)':      (4*np.pi,   np.array(s0)),
         'halo (big)':        (4*np.pi,   np.array(s1)),
         'stable L5 (small)': (100*np.pi, np.array(s2)),
         'stable L5 (big)':   (100*np.pi, np.array(s3)),
         }
#%%
# define DOP853 integrator with same parameters
dop853_prop = ode(jit_crtbp)
dop853_prop.set_integrator('DOP853', max_step=np.inf, rtol=rtol, atol=atol, nsteps=100000)
dop853_prop.set_f_params(mc)

# to retrieve all integrator steps a callback with side effect needed
lst = []
def solout(t, s):
    lst.append([t, *s])
dop853_prop.set_solout(solout)

#%% measure execution time

loops = 1000
res = []
for i, name in enumerate(tests):
    print(name)
    t, s = tests[name]
    lst = []
    dop853_prop.set_initial_value(s, 0.).integrate(t)
    steps0 = len(lst)
    steps1 = rk_prop(crtbp, s, 0., t, max_step, rtol, atol, mc).shape[0]
    r0 = timeit("dop853_prop.set_initial_value(s, 0.).integrate(t)",
                number=loops, globals=globals())
    r1 = timeit("rk_prop(crtbp, s, 0., t, max_step, rtol, atol, mc)",
                number=loops, globals=globals())
    res.append([t, steps0, r0, steps1, r1])

#%% print results

columns = pd.MultiIndex.from_tuples([('integration', 'time'),
                                     ('dop853', 'steps'),
                                     ('dop853', 'time'),
                                     ('fastrk', 'steps'),
                                     ('fastrk', 'time')], names=['', ''])
df = pd.DataFrame(res, columns=columns, index=tests.keys())
df['speedup'] = df[('dop853', 'time')] / df[('fastrk', 'time')]
print(df)
```

Output for `AMD Ryzen 7 4700U @ 4GHz`:

                      integration dop853            fastrk             speedup
                             time  steps       time  steps      time          
    halo (small)        12.566371     96   1.422070    235  0.964003  1.475171
    halo (big)          12.566371    145   2.212505    151  0.611676  3.617117
    stable L5 (small)  314.159265    452   5.898220    422  1.333944  4.421639
    stable L5 (big)    314.159265    916  12.021953    856  2.701954  4.449355

## Detailed examples

#### [Example 0: Propagate spacecraft motion in Circular Restricted Three Body Problem](https://github.com/BoberSA/fastrk/blob/master/examples/ex0_propagate_crtbp.ipynb)

#### [Example 1: Calculate events in Circular Restricted Three Body Problem](https://github.com/BoberSA/fastrk/blob/master/examples/ex1_calculate_events.ipynb)

Required modules:
- [model_crtbp.py](https://github.com/BoberSA/fastrk/blob/master/examples/model_crtbp.py)

#### [Example 2: Custom Embedded Runge-Kutta Method](https://github.com/BoberSA/fastrk/blob/master/examples/ex2_custom_erk_method.ipynb)


#### [Parallel (OpenMP) example](https://github.com/BoberSA/fastrk/blob/master/examples/parallel_example.py)

Required modules:
- [model_crtbp.py](https://github.com/BoberSA/fastrk/blob/master/examples/model_crtbp.py)
- [map_vy_funcs.py](https://github.com/BoberSA/fastrk/blob/master/examples/map_vy_funcs.py)

Output for `AMD Ryzen 7 4700U @ 4GHz`:

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

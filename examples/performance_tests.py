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
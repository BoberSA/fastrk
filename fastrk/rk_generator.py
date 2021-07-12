import inspect
import importlib
import os
import sympy as sp
from .bt import ButcherTable
from . import rk_base

default_jitkwargs = {'nopython': True, 'nogil': True, 'fastmath': True, 'cache': True}

def _py_func(f):
    """Get python function from numba jitted or cfunced function, return f otherwise"""
    return getattr(f, 'py_func', getattr(f, '_pyfunc', f))


def _fun_call_descr(t, s, *args, fname='f'):
    return f"{fname}({t}, {s}, {', '.join(str(a) for a in args)})"


def dict2str(d, sep=', '):
    return ', '.join([f'{key}={value}' for key, value in d.items()])


class RKCodeGen:
    '''
    Code generator for Runge-Kutta methods defined by Butcher Table
    '''
    def __init__(self, bt, autonomous=False, **jitkwargs):
        assert isinstance(bt, ButcherTable)
        self.bt = bt
        self.step_source = None
        self.source = None
        self.autonomous = autonomous
        self.jitkwargs = jitkwargs if jitkwargs else default_jitkwargs
        self.fname = f'rk_{bt.name}.py'

        if not os.path.exists(self.fname):
            self._gen_pre_code()
            self._gen_rk_step_code()
            self._gen_full_code()

    def _gen_pre_code(self):
        stages = self.bt.stages
        A = self.bt.A
        b_main = self.bt.b_main
        b_sub = self.bt.b_sub

        k = ['s']
        for i in range(1, stages):
            tmp = '+'.join(f'{A[i][j]} * k{j}' for j in range(0, i))
            tmps = f's + h * ({tmp})'
            #print(tmps)
            k.append(str(sp.collect(tmps, 'h')))

        tmp = ' + '.join(f"{b_main[i]} * k{i}" for i in range(stages))
        ds_main = sp.collect(f"h * ({tmp})", 'h')
        tmp = ' + '.join(f"{b_main[i]-b_sub[i]} * k{i}" for i in range(stages))
        ds_err = sp.collect(f"h * ({tmp})", 'h')
        self.k = k
        self.ds_main = ds_main
        self.ds_err = ds_err
        #return k, ds_main, ds_subs

    def _gen_rk_step_code(self):

        f_alias = 'fode'
        stages = self.bt.stages
        c = self.bt.c

        text = ''
        fargs = '*args'
        _args = (f_alias, 't', 's', 'h', fargs)

        text += 'def ' + _fun_call_descr(*_args, fname='rk_step') + ':\n'
        text += f'''    \'\'\' One step of Embedded Runge-Kutta method {self.bt.name} \'\'\'\n'''

        fk = []
        for i in range(0, stages):
            t = '0' if self.autonomous else str(sp.collect(f't + {c[i]} * h', 'h'))
            fk.append(_fun_call_descr(t, self.k[i], fargs, fname=f_alias))
            text += f'    k{i} = {fk[-1]}\n'

        text += '\n'

        text += '    res = np.empty((s.shape[0], 2), dtype=s.dtype)\n'
        text += f'    res[:, 0] = s + {self.ds_main}\n'
        text += f'    res[:, 1] = {self.ds_err}\n'

        text += '\n'
        text += '    return res'

        text += '\n\n'
        text += f'RK_ORDER = {self.bt.order}\n\n'

        self.step_source = text

        return self

    def _gen_full_code(self):
        source = 'from numba import jit_module\n'
        source += 'import numpy as np\n\n\n'
        source += self.step_source
        source += inspect.getsource(rk_base)

        source += f"\n\njit_module({dict2str(self.jitkwargs)})\n"

        self.source = source
        return self

    def save_and_import(self, overwrite=False):
        if overwrite:
            self._gen_pre_code()
            self._gen_rk_step_code()
            self._gen_full_code()

        if overwrite or not os.path.exists(self.fname):
            with open(self.fname, 'wt') as f:
                f.write(self.source)

        module = importlib.import_module(f'{self.fname[:-3]}', '')
        return module


#%%




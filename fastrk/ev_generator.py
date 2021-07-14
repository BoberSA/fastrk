import hashlib
import importlib
import inspect
import os
import textwrap

default_jitkwargs = {'nopython': True, 'nogil': True, 'fastmath': True, 'cache': True}


def _py_func(f):
    """Get python function from numba jitted or cfunc'ed function, return f otherwise"""
    return getattr(f, 'py_func', getattr(f, '_pyfunc', f))


def make_hash(s, length):
    return hashlib.shake_128(s.encode()).hexdigest(length)


def dict2str(d, sep=', '):
    return ', '.join([f'{key}={value}' for key, value in d.items()])


def generate_call_event(n, **jitkwargs):

    if n < 1:
        raise ValueError('Value of n should be >= 1')

    lst = []
    lst.append(f"@cfunc('f8(f8, f8[:], f8[:], i8)', {dict2str(jitkwargs)})")
    lst.append('def call_event(t, s, values, idx):')
    lst.append('    if idx == 0:')
    lst.append('        return event_0(t, s, values[0])')

    for i in range(1, n):
        lst.append(f'    elif idx == {i}:')
        lst.append(f'        return event_{i}(t, s, values[{i}])')

    lst.append('    return 0')

    return '\n'.join(lst)


def extract_function_code(fun):
    try:
        text = inspect.getsource(_py_func(fun))
        return text
    except:
        return ''


def extract_method_code(obj, method):
    try:
        text = inspect.getsource(getattr(obj, method))
        return text
    except:
        return ''


def extract_closure_vars(obj):
    try:
        cvars = inspect.getclosurevars(obj)
        return cvars
    except:
        return None


def extract_code(obj, exclude_vars=tuple(), method='__call__'):
    """
    Here obj can be:
    - object which method's code should be extracted with dependencies
    - function which code should be extracted with dependencies
    - builtin which code can't be extracted therefore import this builtin
    """
    obj_ = _py_func(obj)

    if inspect.isbuiltin(obj_):
        return f'from {obj_.__module__} import {obj_.__name__}'
    elif inspect.isfunction(obj_):
        code = '\n' + extract_function_code(obj_).split(':\n')[1]
        deps = extract_deps_code(obj_)
        return deps, code

    code = extract_method_code(obj_, method)
    cvars = extract_closure_vars(getattr(obj_, method))

    deps = ''
    text = ''
    for name, g in cvars.globals.items():
        if hasattr(g, '__call__') or inspect.ismodule(obj_):
            deps += extract_deps_code(g, name=name)
        else:
            text += f'    {name} = {repr(g)}\n'

    for name in cvars.unbound:
        if name in exclude_vars:
            continue
        try:
            attr = _py_func(getattr(obj_, name))
        except:
            continue
            #raise TypeError(f'Attribute {name} is too complicated')
        if hasattr(attr, '__call__') or inspect.ismodule(attr):
            deps += extract_deps_code(attr, name=name)
        else:
            text += f'    {name} = {repr(attr)}\n'

    text += ''.join(textwrap.dedent(code).split(':\n')[1:])
    text = '\n' + text.replace('self.', '').replace('array', 'np.array')

    return deps, text

def extract_deps_code(obj, name=''):
    """
    Here obj can be:
    - module that should be imported
    - function which dependencies should be extracted (recursion)
    - builtin which code can't be extracted therefore import this builtin
    """
    o = _py_func(obj)

    text = ''
    if inspect.ismodule(o):
        return f'import {o.__name__}' + f' as {name}' if name else ''
    elif inspect.isbuiltin(o):
        return f'from {o.__module__} import {o.__name__}'
    elif inspect.isfunction(o):
        cvars = extract_closure_vars(o)
        tmp = ''
        for v, g in cvars.globals.items():
            tmp += extract_deps_code(g, name=v) + '\n'
            #text += extract_function_code(g).replace('@cfunc', '@jit').replace(g.__name__, v) + '\n'
        if tmp:
            text += extract_function_code(o).replace('@cfunc', '@jit').replace(o.__name__, name) + '\n'

    return text

class EventsCodeGen:
    def __init__(self,
                 events,
                 folder='__evcache__',
                 hash_length=12,
                 exclude_vars=('value',),
                 method='__call__',
                 **jitkwargs):
        self.events = events
        self.folder = folder
        self.method = method
        self.jitkwargs = jitkwargs if jitkwargs else default_jitkwargs
        self.hash_length = hash_length
        self.ev_code = []
        self.deps_code = []
        self._extract_events_code(exclude_vars)
        self.hash = make_hash(''.join(self.ev_code), self.hash_length)
        self.fname = f'ev_{self.hash}.py'
        self.fpath = os.path.join(self.folder, self.fname)
        self.existed = self._check_existed()

        if not self.existed:
            self._generate_events_code()

    def _extract_events_code(self, exclude_vars):
        for ev in self.events:
            deps, code = extract_code(ev, exclude_vars, self.method)
            self.ev_code.append(code)
            self.deps_code.append(deps)

    def _check_existed(self):
        return os.path.exists(self.fpath) and os.path.isfile(self.fpath)


    def _generate_events_code(self):
        text = 'from numba import jit, cfunc\nimport numpy as np\n\n'
        jkw = dict2str(self.jitkwargs)

        for i, evtxt in enumerate(self.ev_code):
            text += self.deps_code[i] + '\n'
            text += f'@jit({jkw})\ndef event_{i}(t, s, value):\n'
            text += evtxt + '\n'

        text += generate_call_event(len(self.events), **self.jitkwargs)
        text += '\n'

        self.code = text

    def save_and_import(self, overwrite=False):
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        if not self.existed or overwrite:
            self._generate_events_code()
            with open(self.fpath, 'wt') as f:
                f.write(self.code)
        #importlib.invalidate_caches()
        module = importlib.import_module(f'{self.folder}.{self.fname[:-3]}', '')
        return module.call_event

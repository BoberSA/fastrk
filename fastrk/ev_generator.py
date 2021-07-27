import hashlib
import importlib
import inspect
import os
import textwrap
import re

default_jitkwargs = {'nopython': True, 'nogil': True, 'fastmath': True, 'cache': True, 'boundscheck': False}


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


def unpack_cvars(obj, attr=None, code='', exclude_vars=()):
    vars = {}
    funcs = {}
    modules = {}

    if attr is None:
        cvars = inspect.getclosurevars(obj)
    else:
        cvars = inspect.getclosurevars(attr)

    for d in (cvars.nonlocals.items(), cvars.globals.items(), cvars.builtins.items()):
        for name, value in d:
            if name in exclude_vars:
                continue
            value = _py_func(value)
            if inspect.ismodule(value):
                modules[name] = value
            elif inspect.isbuiltin(value) or inspect.isfunction(value):
                funcs[name] = value
                r = re.search(rf'(\w+).{name}', code)
                if r is not None and r.group() != 'self':
                    funcs[f'{r.group()}.{name}'] = value
            elif inspect.isclass(value):
                pass
            else:
                vars[name] = value

    for name in cvars.unbound:
        if name in exclude_vars:
            continue
        try:
            value = _py_func(getattr(obj, name))
        except:
            continue
        if inspect.ismodule(value):
            modules[name] = value
        elif inspect.isbuiltin(value) or inspect.isfunction(value):
            funcs[name] = value
            r = re.search(rf'(\w+).{name}', code)
            if r is not None and r.group() != 'self':
                funcs[f'{r.group()}.{name}'] = value
        elif inspect.isclass(value):
            pass
        else:
            vars[name] = value

    return vars, funcs, modules


def extract_code(obj, exclude_vars=tuple(), method='__call__', **jitkwargs):
    obj = _py_func(obj)

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        code = inspect.getsource(obj)
        vars, funcs, modules = unpack_cvars(obj, exclude_vars=exclude_vars)
    else:
        try:
            attr = getattr(obj, method)
        except:
            return '', ''
        code = inspect.getsource(attr)
        vars, funcs, modules = unpack_cvars(obj, attr, exclude_vars=exclude_vars)

    deps = ''
    text = ''
    classnames = []
    for name, value in vars.items():
        if inspect.isclass(value):
            classnames.append(name)
            continue
        text += f'{name} = {repr(value)}\n'
    for name, m in modules.items():
        deps += extract_deps(m, name, **jitkwargs)
    for name, f in funcs.items():
        deps += extract_deps(f, name, **jitkwargs)

    code = textwrap.dedent(code.split('):\n')[1])
    source = (text + code).replace('self.','').replace('array', 'np.array')
    #for name in classnames:
    #    source = source.replace(f"{name}.", '')

    return deps, source


def extract_deps(obj, name_='', **jitkwargs):
    obj = _py_func(obj)

    if inspect.ismodule(obj):
        return f'import {obj.__name__}' + f' as {name_}\n' if name_ else '\n'
    elif inspect.isbuiltin(obj) or inspect.isfunction(obj):
        try:
            code = inspect.getsource(obj)
        except:
            try:
                module = importlib.import_module(obj.__module__)
                obj = getattr(module, obj.__name__)
                code = inspect.getsource(obj)
            except:
                return f'from {obj.__module__} import {obj.__name__}' + f' as {name_}\n' if name_ else '\n'
        vars, funcs, modules = unpack_cvars(obj)
        deps = ''
        text = ''
        #classnames = []
        for name, value in vars.items():
            if inspect.isclass(value):
                #classnames.append(name)
                continue
            text += f'{name} = {repr(value)}\n'
        for name, m in modules.items():
            deps += extract_deps(m, name)
        for name, f in funcs.items():
            deps += extract_deps(f, name)

        code += f'\n{name_} = {obj.__name__}\n' if name_ else ''
        spl = code.split('):\n')
        source = spl[0] + '):\n' + text + spl[1]

        if source.startswith('@cfunc') or source.startswith('@jit') or source.startswith('@njit'):
            source = source.replace('@cfunc', '@jit')
        else:
            source = f'@jit({dict2str(jitkwargs)})\n' + source

        return deps + '\n' + source


    raise NotImplemented(f'for objects like {obj}')

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
            deps, code = extract_code(ev, exclude_vars, self.method, **self.jitkwargs)
            self.ev_code.append(code)
            self.deps_code.append(deps)

    def _check_existed(self):
        return os.path.exists(self.fpath) and os.path.isfile(self.fpath)


    def _generate_events_code(self):
        text = 'from numba import jit, njit, cfunc\nimport numpy as np\n\n'
        jkw = dict2str(self.jitkwargs)

        for i, evtxt in enumerate(self.ev_code):
            text += self.deps_code[i] + '\n'
            text += f'@jit({jkw})\ndef event_{i}(t, s, value):\n'
            text += textwrap.indent(evtxt, '    ') + '\n'

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

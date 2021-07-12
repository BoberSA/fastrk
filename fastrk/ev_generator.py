import hashlib
import importlib
import inspect
import os
import textwrap

default_jitkwargs = {'nopython': True, 'nogil': True, 'fastmath': True, 'cache': True}

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


def extract_code(ev, exclude_vars=tuple()):
    try:
        ftext = inspect.getsource(ev.__call__)
        cvars = inspect.getclosurevars(ev.__call__)
    except:
        ftext = inspect.getsource(ev)
        cvars = inspect.getclosurevars(ev)

    text = '\n'.join([f'    {v} = {repr(ev.__getattribute__(v))}' for v in cvars.unbound if v not in exclude_vars])
    text += ''.join(textwrap.dedent(ftext).split(':\n')[1:])
    text = text.replace('self.', '').replace('array', 'np.array')

    return text


class EventsCodeGen:
    def __init__(self, events, folder='_events', hash_length=12, exclude_vars=('value',), **jitkwargs):
        self.events = events
        self.folder = folder
        self.jitkwargs = jitkwargs if jitkwargs else default_jitkwargs
        self.hash_length = hash_length
        self.ev_code = []
        self._extract_events_code(exclude_vars)
        self.hash = make_hash(''.join(self.ev_code), self.hash_length)
        self.fname = f'ev_{self.hash}.py'
        self.fpath = os.path.join(self.folder, self.fname)
        self.existed = self._check_existed()

        if not self.existed:
            self._generate_events_code()

    def _extract_events_code(self, exclude_vars):
        for ev in self.events:
            self.ev_code.append(extract_code(ev, exclude_vars))

    def _check_existed(self):
        return os.path.exists(self.fpath) and os.path.isfile(self.fpath)


    def _generate_events_code(self):
        text = 'from numba import jit, cfunc\n\n'
        jkw = dict2str(self.jitkwargs)

        for i, evtxt in enumerate(self.ev_code):
            text += f'@jit({jkw})\ndef event_{i}(t, s, value):\n'
            text += evtxt + '\n'

        text += generate_call_event(len(self.events), **self.jitkwargs)
        text += '\n'

        self.code = text

    def save_py(self, overwrite=False):
        if not self.existed or overwrite:
            with open(self.fpath, 'wt') as f:
                f.write(self.code)
            importlib.invalidate_caches()
        return self

    def import_call_event(self):
        module = importlib.import_module(f'{self.folder}.{self.fname[:-3]}', '')
        return module.call_event

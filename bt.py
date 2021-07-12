
class ButcherTable:
    '''
    Butcher Table describe any embedded Runge-Kutta method with
    matrix A and vectors b_main, b_subs, c.
    '''
    def __init__(self, name, module):
        self._asserts(module.order, module.A, module.b_main, module.b_subs, module.c)
        self.__doc__ = module.__doc__
        self.A = module.A
        self.b_main = module.b_main
        self.b_sub = module.b_subs
        self.c = module.c
        self.stages = len(self.A)
        self.order = module.order
        self.name = name

    def _asserts(self, order, A, b_main, b_subs, c):
        assert isinstance(order, tuple) or isinstance(order, list)
        assert len(order) == 2
        assert isinstance(A, list)
        assert isinstance(b_main, list)
        assert isinstance(b_subs, list)
        assert isinstance(c, list)
        assert len(A) == len(b_main)
        assert len(A) == len(b_subs)
        assert len(A[-1]) == len(c)

    def __str__(self):
        return f'ButcherTable for RK{self.name}'

import bt_6_5_8_M, bt_8_7_13_M, bt_8_9_16

BT658M = ButcherTable('658M', bt_6_5_8_M)

BT8713M = ButcherTable('8713M', bt_8_7_13_M)

BT8916 = ButcherTable('8916', bt_8_9_16)

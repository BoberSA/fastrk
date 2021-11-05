import unittest
import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings
import fastrk as frk
from model_crtbp import crtbp, crtbp_stm


class PropagationTest(unittest.TestCase):

    def setUp(self):
        self.t0 = 0.
        self.t1 = 3 * np.pi
        self.s0 = np.zeros(6)
        self.s0_stm = np.zeros(42)
        self.s0_stm[6::7] = 1.
        self.s0[[0, 2, 4]] = 9.949942666080747733e-01, 4.732924802139452415e-03, -1.973768492871211949e-02
        self.s0_stm[[0, 2, 4]] = self.s0[[0, 2, 4]]
        self.mc = np.array([3.001348389698916e-06])  # CRTBP constant for Sun-Earth system
        self.rtol, self.atol = 1e-12, 1e-12
        self.tables = ("BT8713M", "BT658M", "BT8916")

    def test_prop(self):
        warnings.filterwarnings('ignore', category=NumbaExperimentalFeatureWarning)

        for tbl in self.tables:
            module = frk.RKCodeGen(getattr(frk, tbl)).save_and_import()
            rk_prop = module.rk_prop
            arr = rk_prop(crtbp, self.s0, self.t0, self.t1, np.inf, self.rtol, self.atol, self.mc)
            arr_stm = rk_prop(crtbp_stm, self.s0_stm, self.t0, self.t1, np.inf, self.rtol, self.atol, self.mc)

class LongPropagationTest(unittest.TestCase):
    """
    L5 -> Earth transfer
    """

    def setUp(self):
        self.t0 = 0.
        self.t1 = 53.95184173720289
        self.s0 = np.zeros(6)
        self.s0_stm = np.zeros(42)
        self.s0_stm[6::7] = 1.
        self.s0[:6] =  (0.5027224192130593, -0.8721468414026837, 0.0,
                        -0.017565498647812646, -0.009604594706234613, 0.0)
        self.s0_stm[:6] = self.s0[:6]
        self.mc = np.array([3.001348389698916e-06])  # CRTBP constant for Sun-Earth system
        self.earth_x = 1 - self.mc[0]
        # self.earth_max_dist = 5.3475935828877e-05  # 8000 km
        self.earth_max_dist = 5.0133689839572194e-05  # 7500 km
        self.rtol, self.atol = 1e-12, 1e-12
        self.tables = ("BT8713M", "BT658M", "BT8916")

    def test_prop(self):
        warnings.filterwarnings('ignore', category=NumbaExperimentalFeatureWarning)

        for tbl in self.tables:
            module = frk.RKCodeGen(getattr(frk, tbl)).save_and_import()
            rk_prop = module.rk_prop
            arr = rk_prop(crtbp, self.s0, self.t0, self.t1, np.inf, self.rtol, self.atol, self.mc)
            d = ((arr[-1, 1] - self.earth_x)**2 - arr[-1, 2]**2)**0.5
            assert d < self.earth_max_dist

            arr_stm = rk_prop(crtbp_stm, self.s0_stm, self.t0, self.t1, np.inf, self.rtol, self.atol, self.mc)
            d_stm = ((arr_stm[-1, 1] - self.earth_x) ** 2 - arr_stm[-1, 2] ** 2) ** 0.5
            assert d_stm < self.earth_max_dist

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

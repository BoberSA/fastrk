import unittest
import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings
import fastrk as frk
from model_crtbp import crtbp, crtbp_stm
from events import (eventX, eventY)


class EventsTest(unittest.TestCase):

    def setUp(self):
        self.t0 = 0.
        self.t1 = 3 * np.pi
        self.s0 = np.zeros(6)
        self.s0_stm = np.zeros(42)
        self.s0_stm[6::7] = 1.
        self.s0[[0, 2, 4]] = 0.9966271059324971, 0.0050402173579027045, -0.024398902561093703
        self.s0_stm[[0, 2, 4]] = self.s0[[0, 2, 4]]
        self.mc = np.array([3.001348389698916e-06])  # CRTBP constant for Sun-Earth system
        self.L1 = 0.9900289479871328

        self.rtol, self.atol = 1e-12, 1e-12
        self.tables = ("BT8713M", "BT658M", "BT8916")

        self.events = (eventX, eventY)
        self.terminals = np.array([False, False])
        self.values = np.array([self.L1, 0.])
        self.counts = np.array([-1, -1])
        self.directions = np.array([0, 0])
        self.accurates = np.array([True, True])

    def test_events(self):
        warnings.filterwarnings('ignore', category=NumbaExperimentalFeatureWarning)

        call_event = frk.EventsCodeGen(self.events).save_and_import()

        for tbl in self.tables:
            module = frk.RKCodeGen(getattr(frk, tbl)).save_and_import()
            rk_prop_ev = module.rk_prop_ev

            # rk_prop_ev(fode, s0, t0, t, max_step, rtol, atol,
            #            _values, _terminals, _directions, _counts, _accurates,
            #            __call_event, __xtol, __rtol, __maxiter, *fargs)

            arr, ev = rk_prop_ev(crtbp, self.s0, self.t0, self.t1, np.inf, self.rtol, self.atol,
                                 self.values, self.terminals, self.directions, self.counts, self.accurates,
                                 call_event, self.atol, self.rtol, 100, self.mc)
            self.assertTrue(np.allclose(ev[ev[:, 0] == 0, 3], self.L1, rtol=self.rtol, atol=self.atol),
                            f"ev=\n{ev}")  # eventX
            self.assertTrue(np.allclose(ev[ev[:, 0] == 1, 4], 0.     , rtol=self.rtol, atol=self.atol),
                            f"ev=\n{ev}")  # eventY

            arr_stm, ev_stm = rk_prop_ev(crtbp_stm, self.s0_stm, self.t0, self.t1, np.inf, self.rtol, self.atol,
                                 self.values, self.terminals, self.directions, self.counts, self.accurates,
                                 call_event, self.atol, self.rtol, 100, self.mc)

            self.assertTrue(np.allclose(ev_stm[ev_stm[:, 0] == 0, 3], self.L1, rtol=self.rtol, atol=self.atol),
                            f"ev_stm=\n{ev_stm}")  # eventX
            self.assertTrue(np.allclose(ev_stm[ev_stm[:, 0] == 1, 4], 0.     , rtol=self.rtol, atol=self.atol),
                            f"ev_stm=\n{ev_stm}")  # eventY
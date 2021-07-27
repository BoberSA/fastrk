import fastrk as frk
import unittest
from events import (eventX, eventY)


class CodeGenTest(unittest.TestCase):

    def setUp(self):
        self.tables = ("BT8713M", "BT658M", "BT8916")

    def test_rkcodegen(self):
        for tbl in self.tables:
            module = frk.RKCodeGen(getattr(frk, tbl)).save_and_import(overwrite=True)
            self.assertIsNotNone(module, f"Error during import of rk_{tbl}.py module")
            self.assertTrue(hasattr(module, 'rk_prop'), f"Error during import of rk_prop from rk_{tbl}.py module")
            self.assertTrue(hasattr(module, 'rk_prop_ev'), f"Error during import of rk_prop_ev from rk_{tbl}.py module")

    def test_evcodegen(self):
        evg = frk.EventsCodeGen([eventX, eventY])
        call_event = evg.save_and_import(overwrite=True)
        name = getattr(evg, 'fname', 'ev_<hash>.py')

        self.assertIsNotNone(call_event, f"Error during import of {name} module")

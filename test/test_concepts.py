import unittest

from GateForge.concepts import Bus, Interface
from GateForge.core import CompileCtx, Reg, RenderCtx, Wire
from GateForge.dsl import reg, wire


class TestBase(unittest.TestCase):

    def setUp(self):
        self.compileCtx = CompileCtx("test")
        CompileCtx.Open(self.compileCtx, 0)
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Close()


class SampleBus(Bus["SampleBus"]):
    w: Wire
    r: Reg


class SampleInterface(Interface["SampleInterface"]):
    w: Wire
    r: Reg


class TestBus(TestBase):

    def test_basic(self):
        b = SampleBus.Create(w=wire().input, r=reg().output)
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "SampleBus_w")
        self.assertEqual(b.r.effectiveName, "SampleBus_r")


    def test_no_args(self):
        with self.assertRaises(Exception):
            SampleBus.Create()


    def test_wrong_arg(self):
        with self.assertRaises(Exception):
            SampleBus.Create(w=wire().input, r=reg().input, a=wire().input)


    def test_wrong_type(self):
        with self.assertRaises(Exception):
            SampleBus.Create(w=wire().input, r=wire().input)
        with self.assertRaises(Exception):
            SampleBus.Create(w=reg().input, r=reg().input)


    def test_no_dir(self):
        with self.assertRaises(Exception):
            SampleBus.Create(w=wire(), r=reg().input)


class TestInterface(TestBase):
    def test_basic(self):
        i = SampleInterface.Create(w=wire().input, r=reg().output)
        self.assertIsInstance(i.w.src, Wire)
        self.assertIsInstance(i.r.src, Reg)

        self.assertIsInstance(i.internal.w.src, Wire)
        self.assertIsInstance(i.internal.r.src, Reg)
        self.assertFalse(i.internal.w.isOutput)
        self.assertTrue(i.internal.r.isOutput)

        self.assertIsInstance(i.external.w.src, Wire)
        self.assertIsInstance(i.external.r.src, Reg)
        self.assertTrue(i.external.w.isOutput)
        self.assertFalse(i.external.r.isOutput)


if __name__ == "__main__":
    unittest.main()

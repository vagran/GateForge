import unittest

from gateforge.concepts import Bus, Interface
from gateforge.core import CompileCtx, InputNet, OutputNet, ParseException, Reg, RenderCtx, Wire
from gateforge.dsl import reg, wire


class TestBase(unittest.TestCase):

    def setUp(self):
        self.compileCtx = CompileCtx("test")
        CompileCtx.Open(self.compileCtx, 0)
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Close()


class SampleBus(Bus["SampleBus"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]


class SampleInterface(Interface["SampleInterface"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]


class TestBus(TestBase):

    def test_basic(self):
        b = SampleBus.Create(w=wire().input, r=reg().output)
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_no_args(self):
        with self.assertRaises(ParseException):
            SampleBus.Create()


    def test_wrong_arg(self):
        with self.assertRaises(ParseException):
            SampleBus.Create(w=wire().input, r=reg().input, a=wire().input)


    def test_bad_annotation(self):
        class BadBus(Bus["BadBus"]):
            w: Wire
            r: Reg

        with self.assertRaises(ParseException):
            BadBus.Create(w=wire().input, r=reg().output)


    def test_bad_dir(self):
        with self.assertRaises(ParseException):
            SampleBus.Create(w=wire().input, r=reg().input)


    def test_wrong_type(self):
        with self.assertRaises(ParseException):
            SampleBus.Create(w=wire().input, r=wire().input)
        with self.assertRaises(ParseException):
            SampleBus.Create(w=reg().input, r=reg().input)


    def test_no_dir(self):
        with self.assertRaises(ParseException):
            SampleBus.Create(w=wire(), r=reg().input)


class TestInterface(TestBase):
    def test_basic(self):
        i = SampleInterface.Create(w=wire().input, r=reg().output)
        self.assertIsInstance(i.w, InputNet)
        self.assertIsInstance(i.w.src, Wire)
        self.assertIsInstance(i.r, OutputNet)
        self.assertIsInstance(i.r.src, Reg)

        self.assertIsInstance(i.internal.w, InputNet)
        self.assertIsInstance(i.internal.w.src, Wire)
        self.assertIsInstance(i.internal.r, OutputNet)
        self.assertIsInstance(i.internal.r.src, Reg)
        self.assertFalse(i.internal.w.isOutput)
        self.assertTrue(i.internal.r.isOutput)

        self.assertIsInstance(i.external.w, OutputNet)
        self.assertIsInstance(i.external.w.src, Wire)
        self.assertIsInstance(i.external.r, InputNet)
        self.assertIsInstance(i.external.r.src, Reg)
        self.assertTrue(i.external.w.isOutput)
        self.assertFalse(i.external.r.isOutput)


    def test_bad_dir(self):
        with self.assertRaises(ParseException):
            SampleInterface.Create(w=wire().output, r=reg().output)


if __name__ == "__main__":
    unittest.main()

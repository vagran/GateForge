# mypy: disable-error-code="type-arg, valid-type"
import unittest

from gateforge.concepts import Bus, ConstructNets, Interface
from gateforge.core import CompileCtx, InputNet, OutputNet, ParseException, Port, Reg, RenderCtx, \
    Wire
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


class SampleBusConstr(Bus["SampleBus"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]

    def __init__(self):
        self.Construct(w=wire().input, r=reg().output)


class SampleInterface(Interface["SampleInterface"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]


class SampleInterfaceConstr(Interface["SampleInterface"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]

    def __init__(self):
        self.Construct(w=wire().input, r=reg().output)


class SizedBus(Bus["SizedBus"]):
    w: InputNet[Wire, (11, 8)]
    r: OutputNet[Reg, 8]
    uw: InputNet[Wire]
    ur: OutputNet[Reg]


class TestBus(TestBase):

    def test_basic(self):
        b = SampleBus.Create(w=wire().input, r=reg().output)
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_ports(self):
        b = SampleBus.Create(w=wire().input, r=reg("r").output.port)
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Port)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_constructor(self):
        b = SampleBusConstr()
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_create_default(self):
        b = SampleBus.CreateDefault()
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_create_default_override(self):
        b = SampleBus.CreateDefault(w=wire("test").input)
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "test")
        self.assertEqual(b.r.effectiveName, "r")


    def test_construct_default(self):
        b = SampleBus()
        b.ConstructDefault()
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_construct_default_override(self):
        b = SampleBus()
        b.ConstructDefault(w=wire("test").input)
        self.assertIsInstance(b.w.src, Wire)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.w.effectiveName, "test")
        self.assertEqual(b.r.effectiveName, "r")


    def test_no_args(self):
        with self.assertRaises(ParseException):
            SampleBus.Create()


    def test_wrong_arg(self):
        with self.assertRaises(ParseException):
            SampleBus.Create(w=wire().input, r=reg().input, a=wire().input)


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


    def test_sized_bus(self):
        b = SizedBus.Create(w=wire((11, 8)).input, r=reg(8).output,
                            uw=wire(2).input, ur=reg(4).output)
        self.assertIsInstance(b.w.src, Wire)
        self.assertEqual(b.w.vectorSize, 4)
        self.assertEqual(b.w.baseIndex, 8)
        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.r.vectorSize, 8)
        self.assertEqual(b.r.baseIndex, 0)
        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")


    def test_sized_bus_bad_size(self):
        with self.assertRaises(ParseException):
            SizedBus.Create(w=wire((11, 8)).input, r=reg(7).output,
                            uw=wire(2).input, ur=reg(4).output)


    def test_sized_bus_bad_base_index(self):
        with self.assertRaises(ParseException):
            SizedBus.Create(w=wire((7, 4)).input, r=reg(7).output,
                            uw=wire(2).input, ur=reg(4).output)


    def test_sized_bus_default(self):
        b = SizedBus.CreateDefault()

        self.assertIsInstance(b.w.src, Wire)
        self.assertEqual(b.w.vectorSize, 4)
        self.assertEqual(b.w.baseIndex, 8)

        self.assertIsInstance(b.r.src, Reg)
        self.assertEqual(b.r.vectorSize, 8)
        self.assertEqual(b.r.baseIndex, 0)

        self.assertIsInstance(b.uw.src, Wire)
        self.assertEqual(b.uw.vectorSize, 1)
        self.assertEqual(b.uw.baseIndex, 0)

        self.assertIsInstance(b.ur.src, Reg)
        self.assertEqual(b.ur.vectorSize, 1)
        self.assertEqual(b.ur.baseIndex, 0)

        self.assertEqual(b.w.effectiveName, "w")
        self.assertEqual(b.r.effectiveName, "r")
        self.assertEqual(b.uw.effectiveName, "uw")
        self.assertEqual(b.ur.effectiveName, "ur")


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


    def test_port(self):
        i = SampleInterface.Create(w=wire().input, r=reg("r").output.port)
        self.assertIsInstance(i.w, InputNet)
        self.assertIsInstance(i.w.src, Wire)
        self.assertIsInstance(i.r, OutputNet)
        self.assertIsInstance(i.r.src, Port)

        self.assertIsInstance(i.internal.w, InputNet)
        self.assertIsInstance(i.internal.w.src, Wire)
        self.assertIsInstance(i.internal.r, OutputNet)
        self.assertIsInstance(i.internal.r.src, Port)
        self.assertFalse(i.internal.w.isOutput)
        self.assertTrue(i.internal.r.isOutput)

        self.assertIsInstance(i.external.w, OutputNet)
        self.assertIsInstance(i.external.w.src, Wire)
        self.assertIsInstance(i.external.r, InputNet)
        self.assertIsInstance(i.external.r.src, Port)
        self.assertTrue(i.external.w.isOutput)
        self.assertFalse(i.external.r.isOutput)


    def test_bad_dir(self):
        with self.assertRaises(ParseException):
            SampleInterface.Create(w=wire().output, r=reg().output)


    def test_constructor(self):
        i = SampleInterfaceConstr()
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


class MyComponent:
    w1: Wire
    w2: Wire[32]
    w3: Wire[[31, 16]]
    w4: Wire[4, [31, 16]]
    r1: Reg[16].array(8)
    r2: Reg[16].array(8, [15, 8])
    r3: Reg


class TestConstructNets(TestBase):

    def test_basic(self):
        comp = MyComponent()
        r3 = reg()
        comp.r3 = r3
        ConstructNets(comp)
        self.assertIs(comp.r3, r3)
        self.assertIsInstance(comp.w1, Wire)
        self.assertIsNone(comp.w1.dims)
        self.assertIsInstance(comp.w2, Wire)
        self.assertEqual(str(comp.w2.dims), "[32 as [31:0]] $")
        self.assertIsInstance(comp.w3, Wire)
        self.assertEqual(str(comp.w3.dims), "[16 as [31:16]] $")
        self.assertIsInstance(comp.w4, Wire)
        self.assertEqual(str(comp.w4.dims), "[4 as [3:0]][16 as [31:16]] $")
        self.assertIsInstance(comp.r1, Reg)
        self.assertEqual(str(comp.r1.dims), "[16 as [15:0]] $[8 as [7:0]]")
        self.assertIsInstance(comp.r2, Reg)
        self.assertEqual(str(comp.r2.dims), "[16 as [15:0]] $[8 as [7:0]][8 as [15:8]]")


if __name__ == "__main__":
    unittest.main()

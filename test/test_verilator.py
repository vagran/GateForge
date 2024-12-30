import io
from pathlib import Path
import unittest

from GateForge.compiler import CompileModule
from GateForge.dsl import wire
from GateForge.verilator import VerilatorParams


def SampleModule():
    in1 = wire("in1").input.port
    in2 = wire("in2").input.port
    # clk = wire("clk").input.port
    out1 = wire("out1").output.port

    out1 <<= in1 ^ in2


class NullOutput(io.StringIO):
    def write(self, s, /):
        pass


class TestBase(unittest.TestCase):

    def setUp(self):
        verilatorParams = VerilatorParams(buildDir=str(Path(__file__).parent / "workspace"),
                                          quite=False)
        self.result = CompileModule(SampleModule, NullOutput(), verilatorParams=verilatorParams)
        self.sim = self.result.simulationModel
        self.ports = self.sim.ports


    def tearDown(self):

        pass


class TestBasic(TestBase):

    def test_basic(self):

        self.ports.in1 = 0
        self.ports.in2 = 0
        self.sim.Eval()
        self.assertEqual(self.ports.out1, 0)
        
        self.ports.in1 = 1
        self.sim.Eval()
        self.assertEqual(self.ports.out1, 1)

        self.ports.in2 = 1
        self.sim.Eval()
        self.assertEqual(self.ports.out1, 0)

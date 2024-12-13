from pathlib import Path
import unittest

from GateForge.core import CompileCtx, ParseException, RenderCtx
from GateForge.dsl import const, reg, wire


class TestBase(unittest.TestCase):

    def setUp(self):
        self.compileCtx = CompileCtx()
        CompileCtx.Open(self.compileCtx, 0)
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Close()


    def CheckResult(self, expected: str, expectedWarnings = 0):
        stmt = self.compileCtx.curBlock._statements[-1]
        # Check source is in current file
        self.assertEqual(Path(stmt.srcFrame.filename).name, "test_statements.py")
        self.assertEqual(str(stmt.Render(self.ctx)), expected)
        self.assertEqual(len(self.compileCtx._warnings), expectedWarnings)


class Test(TestBase):

    def test_continuous_assignment_const(self):
        w = wire("w")
        w <<= 2
        self.CheckResult("assign w = 'h2;")


    def test_continuous_assignment_wire(self):
        w1 = wire("w1")
        w2 = wire("w2")
        w1 <<= w2
        self.CheckResult("assign w1 = w2;")


    def test_continuous_assignment_wire_name_conflict(self):
        w1 = wire("w1")
        w2 = wire("w1")
        w1 <<= w2
        self.CheckResult("assign w1 = w1_0;")


    def test_continuous_assignment_reg_wire_slice(self):
        r = reg(8, "r")
        w = wire(16, "w")
        r <<= w[11:4]
        self.CheckResult("assign r = w[11:4];")


    def test_continuous_assignment_reg_wire_slice_less_bits(self):
        r = reg(8, "r")
        w = wire(16, "w")
        r <<= w[11:5]
        self.CheckResult("assign r = w[11:5];", 1)


    def test_continuous_assignment_reg_wire_slice_more_bits(self):
        r = reg(8, "r")
        w = wire(16, "w")
        with self.assertRaises(ParseException):
            r <<= w[11:3]


    def test_continuous_assignment_lhs_const(self):
        c= const(2)
        with self.assertRaises(ParseException):
            c <<= 42


    def test_continuous_assignment_lhs_const_slice(self):
        c= const(2)
        with self.assertRaises(ParseException):
            c[15:8] <<= 42


    # concat const assignment


if __name__ == "__main__":
    unittest.main()

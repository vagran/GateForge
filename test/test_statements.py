from pathlib import Path
import unittest

from GateForge.core import CompileCtx, ParseException, RenderCtx
from GateForge.dsl import always, const, reg, wire


class TestBase(unittest.TestCase):

    def setUp(self):
        self.compileCtx = CompileCtx("test")
        CompileCtx.Open(self.compileCtx, 0)
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Close()


    def CheckResult(self, expected: str, expectedWarnings = 0):
        stmt = self.compileCtx.curBlock._statements[-1]
        # Check source is in current file
        self.assertEqual(Path(stmt.srcFrame.filename).name, "test_statements.py")
        self.assertEqual(self.ctx.RenderNested(stmt), expected)
        self.assertEqual(len(self.compileCtx.GetWarnings()), expectedWarnings)


    def CheckEmpty(self, expectedWarnings = 0):
        self.assertEqual(len(self.compileCtx.curBlock), 0)
        self.assertEqual(len(self.compileCtx.GetWarnings()), expectedWarnings)


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
        c = const(2)
        with self.assertRaises(ParseException):
            c <<= 42


    def test_continuous_assignment_lhs_const_slice(self):
        c = const(2)
        with self.assertRaises(ParseException):
            c[15:8] <<= 42


    def test_continuous_assignment_concat_lhs(self):
        w1 = wire("w1")
        w2 = wire("w2")
        (w1 % w2).assign(3)
        self.CheckResult("assign {w1, w2} = 'h3;")


    def test_continuous_assignment_concat_non_lhs(self):
        w1 = wire("w1")
        w2 = wire("w2")
        with self.assertRaises(ParseException):
            (const(5) % w1 % w2).assign(3)


class ProceduralBlocks(TestBase):

    def test_empty_sl(self):
        w = wire(8, "w")
        with always():
            w //= 4
        self.CheckResult("""
always @* begin
    w = 'h4;
end
""".lstrip())


    def test_wire_procedural_assignment(self):
        w = wire(8, "w")
        with self.assertRaises(ParseException):
            with always():
                w <<= 4


    def test_empty_body(self):
        with always():
            pass
        self.CheckEmpty(1)


    def test_reg_procedural_assignment(self):
        r = reg(8, "r")
        with always():
            r <<= 4
        self.CheckResult("""
always @* begin
    r <= 'h4;
end
""".lstrip())


    def test_reg_single_trigger(self):
        r = reg(8, "r")
        w = wire("w")
        with always(w):
            r <<= 4
        self.CheckResult("""
always @(w) begin
    r <= 'h4;
end
""".lstrip())


    def test_reg_two_triggers(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always(w1 | w2):
            r <<= 4
        self.CheckResult("""
always @(w1, w2) begin
    r <= 'h4;
end
""".lstrip())


    def test_reg_three_triggers(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        w3 = wire("w3")
        with always(w1 | w2 | w3):
            r <<= 4
        self.CheckResult("""
always @(w1, w2, w3) begin
    r <= 'h4;
end
""".lstrip())


    def test_reg_single_edge_trigger(self):
        r = reg(8, "r")
        w = wire("w")
        with always(w.posedge):
            r <<= 4
        self.CheckResult("""
always @(posedge w) begin
    r <= 'h4;
end
""".lstrip())


    def test_reg_two_edge_triggers(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always(w1.posedge | w2.negedge):
            r <<= 4
        self.CheckResult("""
always @(posedge w1, negedge w2) begin
    r <= 'h4;
end
""".lstrip())


    def test_reg_three_edge_triggers(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        w3 = wire("w3")
        with always(w1.posedge | w2.negedge | w3.posedge):
            r <<= 4
        self.CheckResult("""
always @(posedge w1, negedge w2, posedge w3) begin
    r <= 'h4;
end
""".lstrip())


    def test_bad_sensitivity_list_const(self):
        w = wire("w")
        with self.assertRaises(ParseException):
            with always(w | 5):
                pass


    def test_bad_sensitivity_list_two_mixed(self):
        w1 = wire("w1")
        w2 = wire("w2")
        with self.assertRaises(ParseException):
            with always(w1.posedge | w2):
                w1 <<= 1


    def test_bad_sensitivity_list_three_mixed(self):
        w1 = wire("w1")
        w2 = wire("w2")
        w3 = wire("w3")
        with self.assertRaises(ParseException):
            with always(w1.posedge | w2.negedge | w3):
                pass


if __name__ == "__main__":
    unittest.main()

import io
from pathlib import Path
import unittest

from GateForge.core import CompileCtx, ParseException, RenderCtx
from GateForge.dsl import _case, _default, _else, _elseif, _if, _when, always, const, module, namespace, \
    parameter, reg, wire


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
        with io.StringIO() as output:
            stmt.Render(self.ctx.CreateNested(output))
            result = output.getvalue()
        self.assertEqual(result, expected)
        self.assertEqual(len(self.compileCtx.GetWarnings()), expectedWarnings)


    def CheckEmpty(self, expectedWarnings = 0):
        self.assertEqual(len(self.compileCtx.curBlock), 0)
        self.assertEqual(len(self.compileCtx.GetWarnings()), expectedWarnings)


class TestAssignments(TestBase):

    def test_continuous_assignment_const(self):
        w = wire("w")
        w <<= 1
        self.CheckResult("assign w = 'h1;")


    def test_continuous_assignment_const_size_exceeded(self):
        w = wire("w")
        with self.assertRaises(ParseException):
            w <<= 2


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


    def test_continuous_assignment_lhs_input(self):
        r = reg(8).input
        with self.assertRaises(ParseException):
            r <<= 42


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


    def test_slice_assignment(self):
        w = wire(8, "w")
        w[3:0] <<= 15
        self.CheckResult("assign w[3:0] = 'hf;")
        with self.assertRaises(ParseException):
            w[3:0] = 15


    def test_multiple_assignments(self):
        w = wire(8, "w")
        w[3:0] <<= 15
        w[4] <<= 0
        w[7:5] <= 3
        with self.assertRaises(ParseException):
            w[2] <<= 1
        with self.assertRaises(ParseException):
            w[4] <<= 1


    def test_multiple_assignments_slice(self):
        w = wire(8, "w")
        w[7:5][1] <<= 1
        w[7] <<= 1
        w[5] <<= 1
        with self.assertRaises(ParseException):
            w[6] <<= 1


    def test_multiple_assignments_concat(self):
        w = wire(8, "w")
        c = (w[7:5] % w[3:1])
        c <<= 0
        w[4] <<= 1
        w[0] <<= 1
        with self.assertRaises(ParseException):
            w[7] <<= 1
        with self.assertRaises(ParseException):
            w[6] <<= 1
        with self.assertRaises(ParseException):
            w[5] <<= 1
        with self.assertRaises(ParseException):
            w[3] <<= 1
        with self.assertRaises(ParseException):
            w[2] <<= 1
        with self.assertRaises(ParseException):
            w[1] <<= 1


    def test_multiple_assignments_concat_slice(self):
        w = wire(8, "w")
        w[2:0] <<= 7
        w[4] <<= 0
        w[7:5] <= 3
        # Assign bit 3
        (w[4:3] % w[1])[1] <<= 1
        with self.assertRaises(ParseException):
            w[3] <<= 1


class ProceduralBlocks(TestBase):

    def test_empty_sl(self):
        w = wire(8, "w")
        with always():
            w //= 4
        self.CheckResult("""
always @* begin
    w = 'h4;
end
""".strip())


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
""".strip())


    def test_reg_single_trigger(self):
        r = reg(8, "r")
        w = wire("w")
        with always(w):
            r <<= 4
        self.CheckResult("""
always @(w) begin
    r <= 'h4;
end
""".strip())


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
""".strip())


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
""".strip())


    def test_reg_single_edge_trigger(self):
        r = reg(8, "r")
        w = wire("w")
        with always(w.posedge):
            r <<= 4
        self.CheckResult("""
always @(posedge w) begin
    r <= 'h4;
end
""".strip())


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
""".strip())


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
""".strip())


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


    def test_if_statement(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always():
            with _if(w1 == w2):
                r <<= 4
        self.CheckResult("""
always @* begin
    if (w1 == w2) begin
        r <= 'h4;
    end
end
""".strip())


    def test_if_else_statement(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always():
            with _if(w1 == w2):
                r <<= 4
            with _else():
                r <<= 3
        self.CheckResult("""
always @* begin
    if (w1 == w2) begin
        r <= 'h4;
    end else begin
        r <= 'h3;
    end
end
""".strip())


    def test_if_else_if_statement(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always():
            with _if(w1 == w2):
                r <<= 4
            with _elseif(w1 > w2):
                r <<= 3
        self.CheckResult("""
always @* begin
    if (w1 == w2) begin
        r <= 'h4;
    end else if (w1 > w2) begin
        r <= 'h3;
    end
end
""".strip())


    def test_if_else_if_else_statement(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always():
            with _if(w1 == w2):
                r <<= 4
            with _elseif(w1 > w2):
                r <<= 3
            with _else():
                r <<= 5
        self.CheckResult("""
always @* begin
    if (w1 == w2) begin
        r <= 'h4;
    end else if (w1 > w2) begin
        r <= 'h3;
    end else begin
        r <= 'h5;
    end
end
""".strip())


    def test_if_else_if_else_nested_statement(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with always():
            with _if(w1 == w2):
                r <<= 4
            with _elseif(w1 > w2):
                with _if(w1 == 1):
                    r <<= 3
                with _else():
                    r <<= 6
            with _else():
                r <<= 5
        self.CheckResult("""
always @* begin
    if (w1 == w2) begin
        r <= 'h4;
    end else if (w1 > w2) begin
        if (w1 == 'h1) begin
            r <= 'h3;
        end else begin
            r <= 'h6;
        end
    end else begin
        r <= 'h5;
    end
end
""".strip())


    def test_if_statement_not_in_procedural_block(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")
        with self.assertRaises(ParseException):
            with _if(w1 == w2):
                r <<= 4


    def test_if_statement_else_no_match(self):
        r = reg(8, "r")
        with self.assertRaises(ParseException):
            with _else():
                r <<= 4


    def test_if_statement_else_if_no_match(self):
        r = reg(8, "r")
        with self.assertRaises(ParseException):
            with _elseif(r > 0):
                r <<= 4


    def test_if_statement_else_no_match_2(self):
        r = reg(8, "r")
        with self.assertRaises(ParseException):
            with _if(r > 0):
                r <<= 1
            r <<= 2
            with _else():
                r <<= 4


    def test_when_statement(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")

        with always():
            with _when(r):
                with _case(w1 % w2 % const(0, 6)):
                    r <<= 1
                with _case(w1 % const(0, 7)):
                    r <<= 2
                with _default():
                    r <<= 3

        self.CheckResult("""
always @* begin
    case (r)
        {w1, w2, 6'h0}: begin
            r <= 'h1;
        end
        {w1, 7'h0}: begin
            r <= 'h2;
        end
        default: begin
            r <= 'h3;
        end
    endcase
end
""".strip())


    def test_when_case_size_mismatch(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")

        with always():
            with _when(r):
                with _case(w1 % w2 % const(0, 5)):
                    r <<= 1
                with _case(w1 % const(0, 7)):
                    r <<= 2
                with _default():
                    r <<= 3

        self.CheckResult("""
always @* begin
    case (r)
        {w1, w2, 5'h0}: begin
            r <= 'h1;
        end
        {w1, 7'h0}: begin
            r <= 'h2;
        end
        default: begin
            r <= 'h3;
        end
    endcase
end
""".strip(), 1)


    def test_when_no_procedural_block(self):
        r = reg(8, "r")
        w1 = wire("w1")
        w2 = wire("w2")

        with self.assertRaises(ParseException):
            with _when(r):
                with _case(w1 % w2 % const(0, 6)):
                    r <<= 1
                with _case(w1 % const(0, 7)):
                    r <<= 2
                with _default():
                    r <<= 3


    def test_empty_when(self):
        r = reg(8, "r")
        with always():
            with _when(r):
                pass
        self.CheckResult("""
always @* begin
    case (r)
    endcase
end
""".strip(), 1)


    def test_when_unexpected_code(self):
        r = reg(8, "r")
        w1 = wire("w1")

        with self.assertRaises(ParseException):
            with always():
                with _when(r):
                    with _case(w1 % const(0, 7)):
                        r <<= 2
                    r <<= 1
                    with _default():
                        r <<= 3


    def test_when_multiple_default(self):
        r = reg(8, "r")
        w1 = wire("w1")

        with self.assertRaises(ParseException):
            with always():
                with _when(r):
                    with _case(w1 % const(0, 7)):
                        r <<= 2
                    with _default():
                        r <<= 3
                    with _default():
                        r <<= 4


class ModuleInstantiations(TestBase):

    def test_basic(self):
        w = wire("w")
        r = reg("r")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output,
                   reg([3, 1], "c").input)
        m(a=w, b=r, c=const(1) % w)
        self.CheckResult("""
MyModule MyModule_0(
    .a(w),
    .b(r),
    .c({'h1, w}));
""".strip())


    def test_basic_params(self):
        w = wire("w")
        r = reg("r")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output,
                   reg([3, 1], "c").input,
                   parameter("p1"),
                   parameter("p2"),
                   parameter("p3"))
        m(a=w, b=r, c=const(1) % w, p1="string", p3=42)
        self.CheckResult("""
MyModule #(
    .p1("string"),
    .p3(42))
    MyModule_0(
    .a(w),
    .b(r),
    .c({'h1, w}));
""".strip())


    def test_basic_namespace(self):
        w = wire("w")
        r = reg("r")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output,
                   reg([3, 1], "c").input)
        with namespace("NS"):
            m(a=w, b=r, c=const(1) % w)
        self.CheckResult("""
MyModule NS_MyModule_0(
    .a(w),
    .b(r),
    .c({'h1, w}));
""".strip())


    def test_unnamed_port(self):
        with self.assertRaises(ParseException):
            module("MyModule",
                    wire().input,
                    wire("b").output)


    def test_duplicate_port_name(self):
        with self.assertRaises(ParseException):
            module("MyModule",
                    wire("a").input,
                    wire("a").output)


    def test_wired_port_creation(self):
        a = wire("a")
        a <<= 1
        with self.assertRaises(ParseException):
            a.input.port


    def test_wired_port(self):
        a = wire("a").input
        w = wire()
        w <<= a
        with self.assertRaises(ParseException):
            module("MyModule",
                a,
                wire("b").output)


    def test_port_wired_after_declaration(self):
        a = wire("a").input
        module("MyModule",
            a,
            wire("b").output)
        w = wire()
        with self.assertRaises(ParseException):
            w <<= a


    def test_no_ports(self):
        with self.assertRaises(ParseException):
            module("MyModule")


    def test_bad_lhs(self):
        w = wire("w")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output)
        with self.assertRaises(ParseException):
            m(a=w, b=1)


    def test_bad_lhs_input(self):
        w = wire("w")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output)
        with self.assertRaises(ParseException):
            m(a=w, b=w.input)


    def test_bad_missing_input_port(self):
        w = wire("w")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").input)
        with self.assertRaises(ParseException):
            m(a=w)


    def test_bad_missing_output_port(self):
        w = wire("w")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output)
        m(a=w)
        self.CheckResult("""
MyModule MyModule_0(
    .a(w));
""".strip())


    def test_bad_undeclared_port(self):
        w = wire("w")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output)
        with self.assertRaises(ParseException):
            m(a=w, b=w, c=w)


    def test_size_mismatch(self):
        w = wire("w")
        r = reg("r")
        m = module("MyModule",
                   wire("a").input,
                   wire("b").output,
                   reg([3, 1], "c").input)
        m(a=w, b=r, c=const(1, 1) % w)
        self.CheckResult("""
MyModule MyModule_0(
    .a(w),
    .b(r),
    .c({1'h1, w}));
""".strip(), 1)


    def test_duplicate_module(self):
        module("MyModule",
               wire("a").input,
               wire("b").output)
        with self.assertRaises(ParseException):
            module("MyModule",
                wire("a").input,
                wire("b").output)


if __name__ == "__main__":
    unittest.main()

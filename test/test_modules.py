import io
from typing import Sequence
import unittest

from gateforge.compiler import CompileModule, CompileModuleToString
from gateforge.core import ParseException, RenderOptions
from gateforge.dsl import namespace, reg, wire
from test.utils import WarningTracker


prologue = """`default_nettype none

`define STRINGIFY(x) `"x`"
`define ASSERT(__condition) \\
    if (!(__condition)) begin \\
        $fatal(1, "Assertion failed: %s", `STRINGIFY(__condition)); \\
    end\n
"""

class TestBase(unittest.TestCase):

    def CheckResult(self, moduleFunc, expected: str,
                    expectedWarnings: int | str | Sequence[str] = 0,
                    moduleName = None):
        output = io.StringIO()
        result = CompileModule(moduleFunc, output,
                               renderOptions=RenderOptions(),
                               moduleName=moduleName)

        self.assertEqual(output.getvalue(), prologue + expected)

        wt = WarningTracker(self, result=result)
        wt.Check(expectedWarnings)


class Test(TestBase):

    def test_single_in_out(self):

        def TestModule():
            _in = reg("in").input.port
            _out = wire("out").output.port

            _out <<= _in

        self.CheckResult(TestModule, """
module TestModule(
    input reg in,
    output wire out);

assign out = in;
endmodule
""".lstrip())


    def test_dynamic_slice_index_wiring(self):

        def TestModule():
            _in = reg("in", 8).input.port
            idx = reg("idx", 3).input.port
            _out = wire("out", 8).output.port

            _out[idx] <<= _in[idx]

        self.CheckResult(TestModule, """
module TestModule(
    input reg[2:0] idx,
    input reg[7:0] in,
    output wire[7:0] out);

assign out[idx] = in[idx];
endmodule
""".lstrip())


    def test_namespace_ports(self):

        def TestModule():
            wIn = wire("in")
            wOut = wire("out")
            with namespace("NS"):
                w = wire("w")

            wOut <<= wIn & w

            with namespace("TestNs"):
                _in = reg("in").input.port
                with namespace("Internal"):
                    _out = wire("out").output.port

            _out <<= _in

        self.CheckResult(TestModule, """
module TestModule(
    input reg TestNs_in,
    output wire TestNs_Internal_out);
wire in;
wire out;
wire NS_w;

assign out = in & NS_w;
assign TestNs_Internal_out = TestNs_in;
endmodule
""".lstrip())


    def test_chained_proxy(self):

        def TestModule():
            _in = reg("in").input.port.input.input
            _out = wire("out").output.port.output.output

            _out <<= _in

        self.CheckResult(TestModule, """
module TestModule(
    input reg in,
    output wire out);

assign out = in;
endmodule
""".lstrip())


    def test_duplicate_port_name(self):

        def TestModule():
            _in = wire("a").input.port
            _out = reg("a").output.port

            _out <<= _in

        with self.assertRaises(ParseException):
            CompileModuleToString(TestModule)


    def test_input_port_as_output(self):

        def TestModule():
            _in = wire("a").input.port

            _in.output <<= 1

        with self.assertRaises(ParseException):
            CompileModuleToString(TestModule)


    def test_port_from_port(self):

        def TestModule():
            _out = wire("a").output.port

            w = wire("w")
            w <<= _out.input.port

        with self.assertRaises(ParseException):
            print(CompileModuleToString(TestModule))


    def test_port_wire_name_collision(self):

        def TestModule():
            _in = reg("in").input.port
            _out = wire("out").output.port
            r = reg("in")
            w = wire("out")

            _out <<= _in
            w <<= r

        self.CheckResult(TestModule, """
module TestModule(
    input reg in,
    output wire out);
reg in_1;
wire out_0;

assign out = in;
assign out_0 = in_1;
endmodule
""".lstrip())


    def test_module_name(self):

        def TestModule():
            _in = reg("in").input.port
            _out = wire("out").output.port

            _out <<= _in

        self.CheckResult(TestModule, """
module RenamedModule(
    input reg in,
    output wire out);

assign out = in;
endmodule
""".lstrip(), moduleName="RenamedModule")


    def test_different_size_comparison(self):

        def TestModule():
            w1 = wire("w1").output.port
            r2 = reg("r2").input.port
            r3 = reg("r3").input.port
            w4 = wire("w4").output.port
            w5 = wire("w5", 8).input.port
            w1 <<= r2 == r3
            w4 <<= r2 == w5

        self.CheckResult(TestModule, """
module TestModule(
    input reg r2,
    input reg r3,
    output wire w1,
    output wire w4,
    input wire[7:0] w5);

assign w1 = r2 == r3;
assign w4 = r2 == w5;
endmodule
""".lstrip(), 1)


if __name__ == "__main__":
    unittest.main()

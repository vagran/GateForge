import io
import unittest

from gateforge.compiler import CompileModule, CompileModuleToString
from gateforge.core import ParseException, RenderOptions
from gateforge.dsl import namespace, reg, wire


class TestBase(unittest.TestCase):

    def CheckResult(self, moduleFunc, expected: str, expectedWarnings = 0,
                    moduleName = None):
        output = io.StringIO()
        result = CompileModule(moduleFunc, output,
                               renderOptions=RenderOptions(prohibitUndeclaredNets=False),
                               moduleName=moduleName)
        self.assertEqual(output.getvalue(), expected)
        self.assertEqual(len(result.warnings), expectedWarnings)


class Test(TestBase):

    def test_single_in_out(self):

        def TestModule():
            _in = wire("in").input.port
            _out = reg("out").output.port

            _out <<= _in

        self.CheckResult(TestModule, """
module TestModule(
    input wire in,
    output reg out);

assign out = in;
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
                _in = wire("in").input.port
                with namespace("Internal"):
                    _out = reg("out").output.port

            _out <<= _in

        self.CheckResult(TestModule, """
module TestModule(
    input wire TestNs_in,
    output reg TestNs_Internal_out);
wire in;
wire out;
wire NS_w;

assign out = in & NS_w;
assign TestNs_Internal_out = TestNs_in;
endmodule
""".lstrip())


    def test_chained_proxy(self):

        def TestModule():
            _in = wire("in").input.port.input.input
            _out = reg("out").output.port.output.output

            _out <<= _in

        self.CheckResult(TestModule, """
module TestModule(
    input wire in,
    output reg out);

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
            _in = wire("in").input.port
            _out = reg("out").output.port
            r = reg("in")
            w = wire("out")

            _out <<= _in
            w <<= r

        self.CheckResult(TestModule, """
module TestModule(
    input wire in,
    output reg out);
reg in_1;
wire out_0;

assign out = in;
assign out_0 = in_1;
endmodule
""".lstrip())


    def test_module_name(self):

        def TestModule():
            _in = wire("in").input.port
            _out = reg("out").output.port

            _out <<= _in

        self.CheckResult(TestModule, """
module RenamedModule(
    input wire in,
    output reg out);

assign out = in;
endmodule
""".lstrip(), moduleName="RenamedModule")


    def test_different_size_comparison(self):

        def TestModule():
            w1 = wire("w1").output.port
            w2 = wire("w2").input.port
            r3 = reg("r3").input.port
            r4 = reg("r4").output.port
            w5 = wire(8, "w5").input.port
            w1 <<= w2 == r3
            r4 <<= w2 == w5

        self.CheckResult(TestModule, """
module TestModule(
    input reg r3,
    output reg r4,
    output wire w1,
    input wire w2,
    input wire[7:0] w5);

assign w1 = w2 == r3;
assign r4 = w2 == w5;
endmodule
""".lstrip(), 1)


if __name__ == "__main__":
    unittest.main()

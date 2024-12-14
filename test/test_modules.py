import io
import unittest

from GateForge.compiler import CompileModule, CompileModuleToString
from GateForge.core import ParseException
from GateForge.dsl import reg, wire


class TestBase(unittest.TestCase):

    def CheckResult(self, moduleFunc, expected: str, expectedWarnings = 0,
                    moduleName = None):
        output = io.StringIO()
        result = CompileModule(moduleFunc, output, moduleName=moduleName)
        self.assertEqual(output.getvalue(), expected)
        self.assertEqual(len(result.warnings), expectedWarnings)


class Test(TestBase):

    def test_single_in_out(self):

        def TestModule():
            _in = wire("in").input
            _out = reg("out").output

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
            _in = wire("a").input
            _out = reg("a").output

            _out <<= _in

        with self.assertRaises(ParseException):
            CompileModuleToString(TestModule)


    def test_port_wire_name_collision(self):

        def TestModule():
            _in = wire("in").input
            _out = reg("out").output
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
            _in = wire("in").input
            _out = reg("out").output

            _out <<= _in

        self.CheckResult(TestModule, """
module RenamedModule(
    input wire in,
    output reg out);

assign out = in;
endmodule
""".lstrip(), moduleName="RenamedModule")
        

if __name__ == "__main__":
    unittest.main()

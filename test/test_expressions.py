import io
import unittest

from GateForge.core import CompileCtx, Expression, ParseException, RenderCtx
from GateForge.dsl import const, reg, wire
from pathlib import Path


class TestBase(unittest.TestCase):

    def setUp(self):
        CompileCtx.Open(CompileCtx(), 0)
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Close()


    def CheckExpr(self, expr: Expression, expected: str):
        # Check source is in current file
        self.assertEqual(Path(expr.srcFrame.filename).name, "test_expressions.py")
        self.assertEqual(self.ctx.RenderNested(expr), expected)


class Const(TestBase):

    def test_basic(self):

        self.CheckExpr(const(2), "'h2")
        self.CheckExpr(const(2, 3), "3'h2")

        with self.assertRaises(ParseException):
            const(-2)

        self.assertEqual(const(2).srcFrame.name, "test_basic")

        self.CheckExpr(const("20"), "'h14")
        self.CheckExpr(const("'h20"), "'h20")
        self.CheckExpr(const("'b100"), "'h4")
        self.CheckExpr(const("'o11"), "'h9")
        self.CheckExpr(const("'d16"), "'h10")

        self.CheckExpr(const("5'd16"), "5'h10")

        with self.assertRaises(ParseException):
            const("5'b12")

        # Size trimming, warning expected
        self.CheckExpr(const("5'hffff"), "5'h1f")


class Net(TestBase):

    def test_basic(self):
        w = wire("w")
        self.CheckExpr(w, "w")
        r = reg("r")
        self.CheckExpr(r, "r")
        self.ctx.renderDecl = True
        self.CheckExpr(w, "wire w;")
        self.CheckExpr(r, "reg r;")

        self.CheckExpr(wire(2, "w"), "wire[1:0] w;")

        self.CheckExpr(wire([3, 1], "w"), "wire[3:1] w;")
        self.CheckExpr(reg((5, 0), "w"), "reg[5:0] w;")
        self.CheckExpr(reg((1, 1), "w"), "reg[1:1] w;")

        #XXX non-identifier error

        with self.assertRaises(ParseException):
            reg((1,2))
        with self.assertRaises(ParseException):
            reg(0)
        with self.assertRaises(ParseException):
            reg(-1)

        with self.assertRaises(ParseException):
            wire(",.")

        with self.assertRaises(ParseException):
            reg("16")



class Slice(TestBase):

    def test_basic(self):
        self.CheckExpr(wire(8, "w")[0], "w[0]")
        self.CheckExpr(wire(8, "w")[3], "w[3]")
        self.CheckExpr(wire(8, "w")[5:2], "w[5:2]")
        self.CheckExpr(wire((15, 8), "w")[8], "w[8]")
        self.CheckExpr(wire((15, 8), "w")[15:8], "w[15:8]")

        # Slice of slice optimization
        self.CheckExpr(wire((15, 8), "w")[11:8][2:1], "w[10:9]")
        self.CheckExpr(wire((15, 8), "w")[11:8][2:1][1], "w[10]")

        # Slice of constant optimization
        self.CheckExpr(const(0xde)[7:4], "4'hd")
        self.CheckExpr(const(0xde)[7:4][0], "1'h1")
        self.CheckExpr(const(0xde)[15:8], "8'h0")

        with self.assertRaises(ParseException):
            wire((15, 8))[0]
        with self.assertRaises(ParseException):
            wire((15, 8))[7]
        with self.assertRaises(ParseException):
            wire((15, 8))[16]
        with self.assertRaises(ParseException):
            wire((15, 8))[20:8]
        with self.assertRaises(ParseException):
            wire((15, 8))[10:8:1]
        with self.assertRaises(ParseException):
            wire((15, 8))["aaa"]


# XXX constant concat

# XXX constant posedge, negedge, input exception


if __name__ == "__main__":
    unittest.main()

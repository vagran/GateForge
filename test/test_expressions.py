import unittest

from GateForge.core import CompileCtx, Expression, ParseException, RenderCtx
from GateForge.dsl import const, reg, wire
from pathlib import Path


class TestBase(unittest.TestCase):

    def setUp(self):
        self.compileCtx = CompileCtx("test")
        CompileCtx.Open(self.compileCtx, 0)
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Close()


    def CheckExpr(self, expr: Expression, expected: str, expectedWarnings = 0):
        # Check source is in current file
        self.assertEqual(Path(expr.srcFrame.filename).name, "test_expressions.py")
        self.assertEqual(self.ctx.RenderNested(expr), expected)
        self.assertEqual(len(self.compileCtx.GetWarnings()), expectedWarnings)


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
        self.CheckExpr(const("5'hffff"), "5'h1f", 1)


class Net(TestBase):

    def test_basic(self):
        w = wire("w")
        self.CheckExpr(w, "w")
        r = reg("r")
        self.CheckExpr(r, "r")
        self.ctx.renderDecl = True
        self.CheckExpr(w, "wire w")
        self.CheckExpr(r, "reg r")

        self.CheckExpr(wire(2, "w"), "wire[1:0] w")

        self.CheckExpr(wire([3, 1], "w"), "wire[3:1] w")
        self.CheckExpr(reg((5, 0), "w"), "reg[5:0] w")
        self.CheckExpr(reg((1, 1), "w"), "reg[1:1] w")

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

        with self.assertRaises(ParseException):
            reg("reg")



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


class Concat(TestBase):

    def test_basic(self):
        w1 = wire("w1")
        w2 = wire(8, "w2")
        w3 = wire("w3")
        w4 = wire("w4")
        self.CheckExpr(w1 % w2, "{w1, w2}")
        self.CheckExpr(w1 % w2 % w3, "{w1, w2, w3}")
        self.CheckExpr(w1 % w2 % w3 % w4, "{w1, w2, w3, w4}")
        self.CheckExpr(w1 % w2[7:4] % w3 % w4, "{w1, w2[7:4], w3, w4}")

        self.CheckExpr(const(5) % w2 % w3, "{'h5, w2, w3}")
        with self.assertRaises(ParseException):
            self.CheckExpr(w1 % 5 % w3, "")
        with self.assertRaises(ParseException):
            self.CheckExpr(w1 % w2 % 5, "")
        self.CheckExpr(w1 % const(5, 3) % w3, "{w1, 3'h5, w3}")



class Arithmetic(TestBase):

    def test_basic(self):
        w1 = wire("w1")
        w2 = wire(8, "w2")
        w3 = wire("w3")
        w4 = wire("w4")

        self.CheckExpr(w1 | w2, "w1 | w2")
        self.CheckExpr(w1 | w2 | w3, "w1 | w2 | w3")
        self.CheckExpr(w1 | w2 | w3 | w4, "w1 | w2 | w3 | w4")
        self.CheckExpr(const(5) | w2 | w3, "'h5 | w2 | w3")
        self.CheckExpr(w1 | 5 | w3, "w1 | 'h5 | w3")
        self.CheckExpr(w1 | w2 | 5, "w1 | w2 | 'h5")

        self.CheckExpr(w1 & w2, "w1 & w2")
        self.CheckExpr(w1 & w2 & w3, "w1 & w2 & w3")
        self.CheckExpr(w1 & w2 & w3 & w4, "w1 & w2 & w3 & w4")

        self.CheckExpr(w1 ^ w2, "w1 ^ w2")
        self.CheckExpr(w1 ^ w2 ^ w3, "w1 ^ w2 ^ w3")
        self.CheckExpr(w1 ^ w2 ^ w3 ^ w4, "w1 ^ w2 ^ w3 ^ w4")

        self.CheckExpr(w1 | w2 & w3 | w4, "w1 | (w2 & w3) | w4")
        self.CheckExpr((w1 | w2) & (w3 | w4), "(w1 | w2) & (w3 | w4)")

        self.CheckExpr(~w1, "~w1")
        self.CheckExpr(~w1 | ~w2, "~w1 | ~w2")
        self.CheckExpr(~w1 | ~w2 | ~w3, "~w1 | ~w2 | ~w3")
        self.CheckExpr(~(w1 | w2), "~(w1 | w2)")
        self.CheckExpr(~(w1 | w2) | ~w3, "~(w1 | w2) | ~w3")
        self.CheckExpr(~(w1 | ~w2) & (~w3 | w4), "~(w1 | ~w2) & (~w3 | w4)")
        self.CheckExpr(~(w1 | ~w2)[7:4] & (~w3 | w4), "~(w1 | ~w2)[7:4] & (~w3 | w4)")

        self.CheckExpr(w1.xnor(w2), "w1 ~^ w2")
        self.CheckExpr(w1.xnor(w2.reduce_and), "w1 ~^ &w2")
        self.CheckExpr(w1 | (w2 | w3).reduce_nand | w4, "w1 | ~&(w2 | w3) | w4")

        self.CheckExpr(w1.reduce_or | w2.reduce_nor | w3.reduce_xor | w4.reduce_xnor,
                       "|w1 | ~|w2 | ^w3 | ~^w4", 3)

        with self.assertRaises(ParseException):
            w1.reduce_and.reduce_nand


class Comparison(TestBase):
    def test_basic(self):
        w1 = wire("w1")
        w2 = wire(8, "w2")
        w3 = wire("w3")
        w4 = wire("w4")

        self.CheckExpr(w1 == w2, "w1 == w2")
        self.CheckExpr(w1 == w2 | w3, "w1 == (w2 | w3)")
        self.CheckExpr(w1 | w2 == w3 | w4, "(w1 | w2) == (w3 | w4)")
        self.CheckExpr(w1 | (w2 == w3) | w4, "w1 | (w2 == w3) | w4")
        # Keep in mind comparison operators chaining in Python, so parentheses are mandatory
        self.CheckExpr((w1 < w2) == (w3 > w4), "(w1 < w2) == (w3 > w4)")
        self.CheckExpr((w1 <= w2) != (w3 >= w4), "(w1 <= w2) != (w3 >= w4)")

        self.CheckExpr(w1 == 5, "w1 == 'h5")
        self.CheckExpr(const(5) == w1, "'h5 == w1")


class Replication(TestBase):
    def test_basic(self):
        w1 = wire("w1")
        self.CheckExpr(w1.replicate(5), "{5{w1}}")
        self.CheckExpr(const(5, 3).replicate(4), "{4{3'h5}}")
        # Unbound size replication
        with self.assertRaises(ParseException):
            (const(5) % w1).replicate(3)


if __name__ == "__main__":
    unittest.main()

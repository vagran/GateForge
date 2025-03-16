import io
from typing import Sequence
import unittest

from gateforge.core import CompileCtx, Expression, ParseException, RenderCtx
from gateforge.dsl import call, concat, cond, const, namespace, reg, wire
from pathlib import Path

from test.utils import WarningTracker


class TestBase(unittest.TestCase):

    def setUp(self):
        self.compileCtx = CompileCtx("test")
        CompileCtx.Open(self.compileCtx, 0)
        self.ctx = RenderCtx()
        self.wt = WarningTracker(self, self.compileCtx)


    def tearDown(self):
        CompileCtx.Close()


    def CheckExpr(self, expr: Expression, expected: str, expectedWarnings:int | str | Sequence[str] = 0):
        # Check source is in current file
        self.assertEqual(Path(expr.srcFrame.filename).name, "test_expressions.py")
        with io.StringIO() as output:
            expr.Render(self.ctx.CreateNested(output))
            result = output.getvalue()
        self.assertEqual(result, expected)
        self.wt.Check(expectedWarnings)


class Const(TestBase):

    def test_basic(self):

        self.CheckExpr(const(2), "'h2")
        self.CheckExpr(const(2, 3), "3'h2")

        self.CheckExpr(const(-2), "-'h2")
        self.CheckExpr(const(-2, 3), "-3'h2")

        self.assertEqual(const(2).srcFrame.name, "test_basic")

        self.CheckExpr(const("20"), "'h14")
        self.CheckExpr(const("'h20"), "'h20")
        self.CheckExpr(const("'b100"), "'h4")
        self.CheckExpr(const("'o11"), "'h9")
        self.CheckExpr(const("'d16"), "'h10")
        self.CheckExpr(const(False), "1'h0")
        self.CheckExpr(const(True), "1'h1")
        self.CheckExpr(const("-'h20"), "-'h20")
        self.CheckExpr(const("-7'h20"), "-7'h20")

        self.CheckExpr(const("5'd16"), "5'h10")

        with self.assertRaises(ParseException):
            const("5'b12")

        # Insufficient size
        with self.assertRaises(ParseException):
            const("5'hffff")

        with self.assertRaises(ParseException):
            const(-1, 1)

        with self.assertRaises(ParseException):
            const("-1'h1")

        self.CheckExpr(const(-1, 2), "-2'h1")

        self.CheckExpr(const("'hz1"), "'bzzzz0001")
        self.CheckExpr(const("'h?1"), "'bzzzz0001")
        self.CheckExpr(const("16'h_x_z1"), "16'bxxxxxxxxzzzz0001")
        self.CheckExpr(const("'oz1"), "'bzzz001")
        self.CheckExpr(const("'bz1"), "'bz1")
        with self.assertRaises(ParseException):
            const("'dz1")


class Net(TestBase):

    def test_basic(self):
        w = wire("w")
        self.CheckExpr(w, "w")
        r = reg("r")
        self.CheckExpr(r, "r")
        self.ctx.renderDecl = True
        self.CheckExpr(w, "wire w")
        self.CheckExpr(r, "reg r")

        self.CheckExpr(wire("w", 2), "wire[1:0] w")

        self.CheckExpr(wire("w", [3, 1]), "wire[3:1] w")
        self.CheckExpr(reg("w", (5, 0)), "reg[5:0] w")
        self.CheckExpr(reg("w", (1, 1)), "reg[1:1] w")

        self.CheckExpr(wire("w", 2, 3), "wire[1:0][2:0] w")
        self.CheckExpr(wire("w", 2, (8, 6)), "wire[1:0][8:6] w")
        self.CheckExpr(wire("w", [3, 1], (8, 6)), "wire[3:1][8:6] w")
        self.CheckExpr(wire("w", [1, 3], (8, 6)), "wire[1:3][8:6] w")
        self.CheckExpr(wire("w", [1, 3], (8, 6)).array(4), "wire[1:3][8:6] w[3:0]")
        self.CheckExpr(wire("w", [1, 3], (8, 6)).array(4, 6), "wire[1:3][8:6] w[3:0][5:0]")
        self.CheckExpr(wire("w", [1, 3], (8, 6)).array((3, 1), [7, 3]), "wire[1:3][8:6] w[3:1][7:3]")
        self.CheckExpr(wire("w", [1, 3], (8, 6)).array(5, [3, 7]), "wire[1:3][8:6] w[4:0][3:7]")

        self.assertTrue(wire((31, 2)).dims == [(31, 2)])
        self.assertFalse(wire((31, 2)).dims == [(31, 3)])
        self.assertTrue(wire((31, 2), 8).dims == [(31, 2), 8])
        self.assertFalse(wire((31, 2), 8).dims == [(31, 2), 9])

        with namespace("TestNs"):
            w = wire("w", [3, 1])
        self.CheckExpr(w, "wire[3:1] TestNs_w")

        with namespace("TestNs"):
            w = wire("w", [3, 1]).input
        self.CheckExpr(w, "wire[3:1] TestNs_w")

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
        self.CheckExpr(wire("w", 8)[0], "w[0]")
        self.CheckExpr(wire("w", 8)[3], "w[3]")
        self.CheckExpr(wire("w", 8).input[3], "w[3]")
        self.CheckExpr(wire("w", 8).output[3], "w[3]")
        self.CheckExpr(wire("w", 8)[5:2], "w[5:2]")
        self.CheckExpr(wire("w", (15, 8))[8], "w[8]")
        self.CheckExpr(wire("w", (15, 8))[15:8], "w[15:8]")

        # Slice of slice optimization (removed for now)
        # self.CheckExpr(wire("w", (15, 8))[11:8][2:1], "w[10:9]")
        # self.CheckExpr(wire("w", (15, 8))[11:8][2:1][1], "w[10]")
        self.CheckExpr(wire("w", (15, 8))[11:8][2:1], "w[11:8][2:1]")
        self.CheckExpr(wire("w", (15, 8))[11:8][2:1][1], "w[11:8][2:1][1]")

        # Slice of constant optimization
        self.CheckExpr(const(0xde)[7:4], "4'hd")
        self.CheckExpr(const(0xde)[7:4][0], "1'h1")
        self.CheckExpr(const(0xde)[15:8], "8'h0")

        # Extended slices support
        self.CheckExpr(const(0xde)[7:], "8'hde")
        self.CheckExpr(const(0xde)[:0], "8'hde")
        self.CheckExpr(const(0xde)[:], "8'hde")
        self.CheckExpr(const(0xde)[:4], "4'hd")
        self.CheckExpr(const(0xde)[3:], "4'he")
        self.CheckExpr(wire("w", 8)[5:], "w[5:0]")
        self.CheckExpr(wire("w", 8)[:2], "w[7:2]")
        self.CheckExpr(wire("w", 8)[:], "w[7:0]")

        with self.assertRaises(ParseException):
            const(0xde)[7:4][4]

        # Reverse endianness in slice
        with self.assertRaises(ParseException):
            wire((15, 8))[10:12]
        with self.assertRaises(ParseException):
            wire((8, 15))[12:10]
        self.CheckExpr(wire("w", (8, 15))[10:12], "w[10:12]")

        # Off-by-one both endianness
        self.CheckExpr(wire("w", (8, 15))[8], "w[8]")
        self.CheckExpr(wire("w", (8, 15))[15], "w[15]")
        with self.assertRaises(ParseException):
            wire((8, 15))[16]
        with self.assertRaises(ParseException):
            wire((8, 15))[7]

        self.CheckExpr(wire("w", (15, 8))[8], "w[8]")
        self.CheckExpr(wire("w", (15, 8))[15], "w[15]")
        with self.assertRaises(ParseException):
            wire((15, 8))[16]
        with self.assertRaises(ParseException):
            wire((15, 8))[7]

        self.CheckExpr(wire("w", (15, 8))[wire("w2")], "w[w2]")

        self.assertEqual(1, wire().vectorSize)
        self.assertEqual(5, wire(5).vectorSize)
        self.assertEqual(8, wire((11, 4)).vectorSize)
        self.assertEqual(32, wire((11, 4), 4).vectorSize)
        self.assertEqual(48, wire((11, 4), (15, 10)).vectorSize)
        self.assertEqual(48, wire((11, 4), (15, 10)).array(4).vectorSize)

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
        with self.assertRaises(ParseException):
            wire()[0]


class Concat(TestBase):

    def test_basic(self):
        w1 = wire("w1")
        w2 = wire("w2", 8)
        w3 = wire("w3")
        w4 = wire("w4")
        w5 = wire("w5", 2, 3)
        self.CheckExpr(w1 % w2, "{w1, w2}")
        self.CheckExpr(w1 % w2 % w3, "{w1, w2, w3}")
        self.CheckExpr(w1 % w2 % w3 % w4, "{w1, w2, w3, w4}")
        self.CheckExpr(w1 % w2[7:4] % w3 % w4, "{w1, w2[7:4], w3, w4}")
        self.CheckExpr(concat(w1, w2[7:4], w3, w4), "{w1, w2[7:4], w3, w4}")

        self.CheckExpr(const(5) % w2 % w3, "{'h5, w2, w3}")
        with self.assertRaises(ParseException):
            self.CheckExpr(w1 % 5 % w3, "")
        with self.assertRaises(ParseException):
            self.CheckExpr(w1 % w2 % 5, "")
        self.CheckExpr(w1 % const(5, 3) % w3, "{w1, 3'h5, w3}")

        self.assertEqual((w1 % w2).vectorSize, 9)
        self.assertEqual((w1 % w2 % w3).vectorSize, 10)
        self.assertEqual((w1 % w2 % w3 % w4).vectorSize, 11)
        self.assertEqual(len(w1 % w2 % w3 % w4), 11)
        self.assertEqual((w1 % w2 % w5).vectorSize, 15)

        self.assertEqual((const(5) % w1).vectorSize, 4)

        with self.assertRaises(ParseException):
            w1 % 5

        with self.assertRaises(ParseException):
            w1 % wire(5).array(10)

        with self.assertRaises(ParseException):
            (w1 % 5)[1][1]


class Shift(TestBase):

    def test_basic(self):
        w1 = wire("w1")
        w2 = wire("w2", 8)
        w3 = wire("w3")
        w4 = wire("w4")

        self.CheckExpr(w2.sll(2), "w2 << 'h2")
        self.CheckExpr(w2.srl(2), "w2 >> 'h2")
        self.CheckExpr(w2.sra(2), "w2 >>> 'h2")


class Arithmetic(TestBase):

    def test_basic(self):
        w1 = wire("w1")
        w2 = wire("w2", 8)
        w3 = wire("w3")
        w4 = wire("w4")

        self.assertEqual(len(w1), 1)
        self.assertEqual(len(w2), 8)

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
        self.CheckExpr(~(w1 | ~w2)[7:4] & (~w3 | w4), "~{w1 | ~w2}[7:4] & (~w3 | w4)")

        self.CheckExpr(w1.xnor(w2), "w1 ~^ w2")
        self.CheckExpr(w1.xnor(w2.reduce_and), "w1 ~^ &w2")
        self.CheckExpr(w1 | (w2 | w3).reduce_nand | w4, "w1 | ~&(w2 | w3) | w4")

        self.CheckExpr(w1 + w2, "w1 + w2")
        self.CheckExpr(w1 - w2, "w1 - w2")

        self.CheckExpr(w1.reduce_or | w2.reduce_nor | w3.reduce_xor | w4.reduce_xnor,
                       "|w1 | ~|w2 | ^w3 | ~^w4", 3)

        with self.assertRaises(ParseException):
            w1.reduce_and.reduce_nand


class Comparison(TestBase):
    def test_basic(self):
        w1 = wire("w1")
        w2 = wire("w2", 8)
        w3 = wire("w3")
        w4 = wire("w4")

        self.CheckExpr(w1 == w2, "w1 == w2")
        self.CheckExpr(w1 == w3 | w4, "w1 == (w3 | w4)")
        self.CheckExpr(w1 | w2 == w3 | w4, "(w1 | w2) == (w3 | w4)")
        self.CheckExpr(w1 | (w2 == w3) | w4, "w1 | (w2 == w3) | w4")
        # Keep in mind comparison operators chaining in Python, so parentheses are mandatory
        self.CheckExpr((w1 < w2) == (w3 > w4), "(w1 < w2) == (w3 > w4)")
        self.CheckExpr((w1 <= w2) != (w3 >= w4), "(w1 <= w2) != (w3 >= w4)")

        self.CheckExpr(w1 == 5, "w1 == 'h5")
        self.CheckExpr(const(5) == w1, "'h5 == w1")

        self.CheckExpr(w1 == True, "w1 == 1'h1")

        with self.assertRaises(ParseException):
            if w1 == w2:
                pass


class Replication(TestBase):
    def test_basic(self):
        w1 = wire("w1")
        self.CheckExpr(w1.replicate(5), "{5{w1}}")
        self.assertEqual(5, w1.replicate(5).vectorSize)
        self.CheckExpr(const(5, 3).replicate(4), "{4{3'h5}}")
        # Unbound size replication
        with self.assertRaises(ParseException):
            (const(5) % w1).replicate(3)

        self.CheckExpr(wire("w", (5, 2)).replicate(5), "{5{w}}")
        self.assertEqual(20, wire("w", (5, 2)).replicate(5).vectorSize)
        with self.assertRaises(ParseException):
            w1.array(5).replicate(3)



class FunctionCall(TestBase):
    def test_basic(self):
        w1 = wire("w1")
        w2 = wire("w2")
        self.CheckExpr(w1.signed, "$signed(w1)")
        self.CheckExpr(call("someFunc", w1, w2, 2), "someFunc(w1, w2, 'h2)")


class Conditional(TestBase):
    def test_basic(self):
        w1 = wire("w1")
        w2 = wire("w2", 8)
        w3 = wire("w3")
        w4 = wire("w4")
        self.CheckExpr((w1 == w2[0]).cond(w3, w4), "(w1 == w2[0]) ? w3 : w4")
        self.CheckExpr(cond(w1 == w2[0], w3, w4), "(w1 == w2[0]) ? w3 : w4")
        self.CheckExpr(((w1 == w2[0]).cond(w3, w4)).cond(w1, w2.cond(w3, w4)),
                       "((w1 == w2[0]) ? w3 : w4) ? w1 : (w2 ? w3 : w4)")

        with self.assertRaises(ParseException):
            cond(w1, wire(2).array(3), wire(3).array(3))

        self.CheckExpr(w1.cond(wire("a", 2).array(3), wire("b", 2).array(3)),
                       "w1 ? a : b")

        self.assertEqual(w1.cond(wire(5, 2), wire(6, 3)).vectorSize, 18)

        with self.assertRaises(ParseException):
            cond("a", w1, w2)


if __name__ == "__main__":
    unittest.main()

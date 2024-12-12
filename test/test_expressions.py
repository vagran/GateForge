import unittest

from GateForge.core import CompileCtx, Expression, ModuleCtx, ParseException, RenderCtx
from GateForge.dsl import const, reg, wire


class TestBase(unittest.TestCase):

    def setUp(self):
        ctx = CompileCtx()
        CompileCtx.Open(ctx)
        ctx.OpenModule(ModuleCtx())
        self.ctx = RenderCtx()


    def tearDown(self):
        CompileCtx.Current().CloseModule()
        CompileCtx.Close()


    def CheckExpr(self, expr: Expression, expected: str):
        self.assertEqual(str(expr.Render(self.ctx)), expected)


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


class Net(TestBase):

    def test_basic(self):
        w = wire()
        self.CheckExpr(w, "w_0")
        r = reg()
        self.CheckExpr(r, "r_1")
        self.ctx.renderDecl = True
        self.CheckExpr(w, "wire w_0;")
        self.CheckExpr(r, "reg r_1;")

        self.CheckExpr(wire(2, "w"), "wire[1:0] w;")
        # Duplicated name
        with self.assertRaises(ParseException):
            wire("w")

        self.CheckExpr(wire([3, 1], "a"), "wire[3:1] a;")
        self.CheckExpr(reg((5, 0), "b"), "reg[5:0] b;")
        self.CheckExpr(reg((1, 1), "c"), "reg[1:1] c;")

        with self.assertRaises(ParseException):
            reg((1,2))
        with self.assertRaises(ParseException):
            reg(0)
        with self.assertRaises(ParseException):
            reg(-1)


# constant slice

# constant concat

if __name__ == "__main__":
    unittest.main()

import unittest

from GateForge.core import CompileCtx, Expression, ParseException, RenderCtx
from GateForge.dsl import const


class TestBase(unittest.TestCase):

    def setUp(self):
        self.ctx = RenderCtx()
        CompileCtx.Open(CompileCtx())

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()

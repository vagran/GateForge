from typing import List, Optional, Tuple
import traceback
import re
import math


class ParseException(Exception):
    pass


class RenderCtx:
    pass


class CompileCtx:

    lastFrame: Optional[traceback.FrameSummary] = None

    @staticmethod
    def Current() -> "CompileCtx":
        global _curCompileCtx
        if _curCompileCtx is None:
            raise Exception("Synthesizable functions are not allowed to be called outside the compiler")
        return _curCompileCtx


    @staticmethod
    def Open(ctx: "CompileCtx"):
        global _curCompileCtx
        if _curCompileCtx is not None:
            raise Exception("Compilation context override")
        _curCompileCtx = ctx


    @staticmethod
    def Close():
        global _curCompileCtx
        if _curCompileCtx is None:
            raise Exception("Compilation context not open")
        _curCompileCtx = None


    def Warning(self, msg: str, frame: Optional[traceback.FrameSummary] = None):
        #XXX
        if frame is None:
            frame = self.lastFrame
        if frame is not None:
            loc = f"{frame.filename}:{frame.lineno}"
            print(f"WARN [{loc}] {msg}")
        else:
            print(f"WARN {msg}")


_curCompileCtx: Optional[CompileCtx] = None


class RenderResult:
    _strings: List[str]


    def __init__(self, line: Optional[str]):
        self._strings = []
        if line is not None:
            self._strings.append(line)


    def Append(self, chunk: "RenderResult"):
        self._strings.extend(chunk._strings)


    def __str__(self):
        return "\n".join(self._strings)


class SyntaxNode:
    # Stack frame of the Python source code for this node
    srcFrame: Optional[traceback.FrameSummary] = None

    def __init__(self, frameDepth=0):
        # Raise exception if not context
        ctx = CompileCtx.Current()
        if frameDepth is not None:
            self.srcFrame = traceback.extract_stack()[-frameDepth - 2]
            ctx.lastFrame = self.srcFrame

    def Render(self, ctx: RenderCtx) -> RenderResult:
        raise NotImplementedError()


class Expression(SyntaxNode):
    size: Optional[int] = None
    isLhs: bool = False


class Const(Expression):
    # Minimal number of bits required to present the value without trimming
    valueSize: int
    # Constant value
    value: int

    _valuePat = re.compile(r"(?:(\d+)?'([bdoh]))?([\da-f_]+)", re.RegexFlag.IGNORECASE)


    def __init__(self, value: str | int, size: Optional[int] = None, *, frameDepth=0):
        super().__init__(frameDepth + 1)

        if isinstance(value, str):
            if size is not None:
                raise ParseException("Size should not be specified for string value")
            self.value, self.size = Const._ParseStringValue(value)

        else:
            self.value = value
            self.size = size

        if self.value < 0:
            raise ParseException("Negative values not allowed")

        self.valueSize = Const.GetMinValueBits(self.value)

        if self.size is not None and self.valueSize > self.size:
            CompileCtx.Current().Warning(
                f"Constant explicit size less than value required size: {self.size} < {self.valueSize}")


    def Render(self, ctx: RenderCtx) -> RenderResult:
        return RenderResult(f"{'' if self.size is None else self.size}'h{self.value:x}")


    @staticmethod
    def GetMinValueBits(x: int) -> int:
        if x == 0:
            return 1  # One bit is needed to represent zero
        elif x < 0:
            # For negative numbers assume one sign bit
            return math.floor(math.log2(abs(x))) + 2
        else:
            # For non-negative numbers, just calculate the bits needed
            return math.floor(math.log2(x)) + 1

    @staticmethod
    def _ParseStringValue(valueStr: str) -> Tuple[int, Optional[int]]:
        """
        Parse Verilog constant literal.
        :return: numeric value and optional size.
        """
        m = Const._valuePat.fullmatch(valueStr)
        if m is None:
            raise ParseException(f"Bad constant format: `{valueStr}`")
        groups = m.groups()

        size = None

        if groups[0] is not None:
            try:
                size = int(groups[0])
            except:
                raise ParseException(f"Bad size value in constant: `{groups[0]}")
            if size <= 0:
                raise ParseException(f"Bad size value in constant: `{size}`")

        base = 10
        if groups[1] is not None:
            b = groups[1].lower()
            if b == "b":
                base = 2
            elif b == "o":
                base = 8
            elif b == "d":
                base = 10
            elif b == "h":
                base = 16
            else:
                raise Exception("Unhandled base char")

        try:
            value = int(groups[2], base)
        except:
            raise ParseException(f"Unable to parse value `{groups[2]}` with base {base}")

        return value, size



class ConcatExpr(Expression):

    pass


class Net(Expression):

    pass

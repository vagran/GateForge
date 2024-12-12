import threading
from typing import List, Optional, Tuple
import traceback
import re
import math
from pathlib import Path


class ParseException(Exception):
    pass


class ModuleCtx:
    _curNetIdx: int
    _netNames: dict[str, traceback.FrameSummary]
    _ports: dict["Expression", "Port"]

    compileCtx: "CompileCtx"


    def __init__(self):
        self._curNetIdx = 0
        self._netNames = dict()
        self._ports = dict()


    def GenerateNetName(self, isReg: bool) -> str:
        while True:
            idx = self._curNetIdx
            self._curNetIdx += 1
            name = f"{'r' if isReg else 'w'}_{idx}"
            if name in self._netNames:
                continue
            return name


    def RegisterNetName(self, name: str, frame: traceback.FrameSummary):
        f = self._netNames.get(name, None)
        if f is not None:
            raise ParseException(f"Net with name `{name}` already declared at {f.filename}:{f.lineno}")
        self._netNames[name] = frame


    def CheckPort(self, src: "Expression", frameDepth: int) -> "Expression":
        """
        Check if the expression is originated from another module, create (or use existing) port if
        so.
        """
        if isinstance(src, Const):
            # Constants used directly even if passed externally.
            return src
        if src.moduleCtx is self:
            # Originated from this module.
            return src
        if src.moduleCtx.compileCtx is not self.compileCtx:
            raise Exception("Encountered expression from different compilation context")
        port = self._ports.get(src, None)
        if port is not None:
            return port
        port = Port(src, frameDepth + 1)
        self._ports[src] = port
        return port


class CompileCtx:

    lastFrame: Optional[traceback.FrameSummary] = None

    _threadLocal = threading.local()
    _moduleStack: List[ModuleCtx]


    def __init__(self):
        self._moduleStack = []


    @staticmethod
    def _GetCurrent() -> Optional["CompileCtx"]:
        return getattr(CompileCtx._threadLocal, "_current", None)


    @staticmethod
    def _SetCurrent(value: Optional["CompileCtx"]):
        CompileCtx._threadLocal._current = value


    @staticmethod
    def Current() -> "CompileCtx":
        cur = CompileCtx._GetCurrent()
        if cur is None:
            raise Exception("Synthesizable functions are not allowed to be called outside the compiler")
        return cur


    @staticmethod
    def Open(ctx: "CompileCtx"):
        if CompileCtx._GetCurrent() is not None:
            raise Exception("Compilation context override")
        CompileCtx._SetCurrent(ctx)


    @staticmethod
    def Close():
        if CompileCtx._GetCurrent() is None:
            raise Exception("Compilation context not open")
        CompileCtx._SetCurrent(None)


    @staticmethod
    def EnsureContext():
        CompileCtx.Current().moduleCtx


    @property
    def moduleCtx(self) -> ModuleCtx:
        if len(self._moduleStack) == 0:
            raise Exception("No module context")
        return self._moduleStack[-1]


    def OpenModule(self, ctx: ModuleCtx):
        self._moduleStack.append(ctx)
        ctx.compileCtx = self


    def CloseModule(self):
        if len(self._moduleStack) == 0:
            raise Exception("Module stack underflow")
        self._moduleStack.pop()


    def Warning(self, msg: str, frame: Optional[traceback.FrameSummary] = None):
        #XXX
        if frame is None:
            frame = self.lastFrame
        if frame is not None:
            loc = f"{frame.filename}:{frame.lineno}"
            print(f"WARN [{loc}] {msg}")
        else:
            print(f"WARN {msg}")


class RenderCtx:
    # Render declaration instead of expression when True
    renderDecl: bool = False

    #XXX


class RenderResult:
    _strings: List[str]


    def __init__(self, line: Optional[str]):
        self._strings = []
        if line is not None:
            self._strings.append(line)


    def Append(self, chunk: "RenderResult"):
        self._strings.extend(chunk._strings)


    def Line(self) -> str:
        """Assume single line"""
        if len(self._strings) != 1:
            raise Exception("Expected single line render result")
        return self._strings[0]


    def __str__(self):
        return "\n".join(self._strings)


class SyntaxNode:
    moduleCtx: ModuleCtx
    # Stack frame of the Python source code for this node
    srcFrame: Optional[traceback.FrameSummary] = None
    # String value to use in diagnostic messages
    strValue: Optional[str] = None


    def __init__(self, frameDepth: int):
        # Raise exception if no context
        ctx = CompileCtx.Current()
        self.moduleCtx = ctx.moduleCtx
        # Ensure module
        ctx.moduleCtx
        self.srcFrame = self.GetFrame(frameDepth + 1)
        ctx.lastFrame = self.srcFrame


    def __str__(self) -> str:
        if self.strValue is None:
            s = type(self).__name__
        else:
            s = self.strValue
        if self.srcFrame is not None:
            s += f"[{Path(self.srcFrame.filename).name}:{self.srcFrame.lineno}]"
        return s


    def GetFrame(self, frameDepth: int):
        return traceback.extract_stack()[-frameDepth - 2]


    def Render(self, ctx: RenderCtx) -> RenderResult:
        raise NotImplementedError()


class Expression(SyntaxNode):
    size: Optional[int] = None
    isLhs: bool = False


    def _CheckSlice(self, s: int | slice) -> Tuple[int, int]:
        if isinstance(s, int):
            if s < 0:
                raise ParseException(f"Negative slice index: {s}")
            return (s, 1)
        elif isinstance(s, slice):
            if s.step is not None:
                raise ParseException("Slice cannot have step specified")
            if not isinstance(s.start, int):
                raise ParseException(f"Slice MSB is not int: {s.start}")
            if not isinstance(s.stop, int):
                raise ParseException(f"Slice LSB is not int: {s.start}")
            if s.start < s.stop:
                raise ParseException(f"Slice MSB should be not less than LSB, has {s.start} < {s.stop}")
            return (s.stop, s.start - s.stop + 1)
        else:
            raise ParseException(f"Expected int or slice, has `{s}: {type(s).__name__}`")


    def __getitem__(self, s):
        index, size = self._CheckSlice(s)
        return SliceExpr(self, index, size, 1)


class Const(Expression):
    # Minimal number of bits required to present the value without trimming
    valueSize: int
    # Constant value
    value: int

    _valuePat = re.compile(r"(?:(\d+)?'([bdoh]))?([\da-f_]+)", re.RegexFlag.IGNORECASE)


    def __init__(self, value: str | int, size: Optional[int] = None, *, frameDepth: int):
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
            self.value = self.value & ((1 << self.size) - 1)

        self.strValue = f"Const({self.value})"


    def __getitem__(self, s):
        # Slicing is optimized to produce constant
        index, size = self._CheckSlice(s)
        mask = (1 << size) - 1
        return Const((self.value >> index) & mask, size, frameDepth=1)


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


class Net(Expression):
    isLhs = True
    baseIndex: int = 0
    isReg: bool
    name: str


    def __init__(self, *, size: int, baseIndex: Optional[int], isReg: bool, name: Optional[str],
                 frameDepth: int):
        super().__init__(frameDepth + 1)
        if size <= 0:
            raise ParseException(f"Net size should be positive, have {size}")
        self.size = size
        if baseIndex is not None:
            if baseIndex < 0:
                raise ParseException(f"Net base index cannot be negative, have {baseIndex}")
            self.baseIndex = baseIndex
        self.isReg = isReg

        modCtx = CompileCtx.Current().moduleCtx
        if name is None:
            name = modCtx.GenerateNetName(isReg)
        modCtx.RegisterNetName(name, self.GetFrame(frameDepth + 1))
        self.name = name


    def __getitem__(self, s):
        # Need to adjust index according to baseIndex
        index, size = self._CheckSlice(s)
        if index < self.baseIndex:
            raise ParseException(f"Index is less than LSB index: {index} < {self.baseIndex}")
        return SliceExpr(self, index - self.baseIndex, size, 1)


    def Render(self, ctx: RenderCtx) -> RenderResult:
        if ctx.renderDecl:
            s = "reg" if self.isReg else "wire"
            assert self.size is not None
            if self.baseIndex != 0 or self.size > 1:
                s += f"[{self.baseIndex + self.size - 1}:{self.baseIndex}]"
            s += f" {self.name};"
            return RenderResult(s)
        return RenderResult(self.name)


class Wire(Net):
    isReg = False


class Reg(Net):
    isReg = True


# Port instance is created when external expression is encountered.
class Port(Net):
    src: Expression
    # Set when used as output in LHS expression
    isOutput = False

    def __init__(self, src: Expression, frameDepth):
        if isinstance(src, Net):
            baseIndex = src.baseIndex
        else:
            baseIndex = 0
        assert src.size is not None
        super().__init__(size=src.size, baseIndex=baseIndex, isReg=True,
                         name=getattr(src, "name", None), frameDepth=frameDepth + 1)
        self.src = src
        self.isLhs = src.isLhs
        self.size = src.size

    #XXX


class ConcatExpr(Expression):
    #XXX
    pass


class SliceExpr(Expression):
    src: Expression
    # Index is always zero-based, nets' base index is applied before if needed.
    index: int


    def __init__(self, src: Expression, index: int, size: int, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.src = CompileCtx.Current().moduleCtx.CheckPort(src, frameDepth + 1)
        self.isLhs = self.src.isLhs
        if self.src.size is not None:
            self._CheckRange(index, size, self.src.size)
        self.size = size
        self.index = index


    def _CheckRange(self, index: int, size: int, srcSize: int):
        if index + size > srcSize:
            raise ParseException(
                "Slice exceeds source expression size: "
                f"[{index + size - 1}:{index}] out of {srcSize} bits source")


    def __getitem__(self, s):
        # Slicing is optimized to use inner slice source directly
        index, size = self._CheckSlice(s)
        self._CheckRange(index, size, self.size)
        return SliceExpr(self.src, self.index + index, size, 1)


    def Render(self, ctx: RenderCtx) -> RenderResult:
        index = self.index
        assert self.size is not None
        size = self.size
        if isinstance(self.src, Net):
            index += self.src.baseIndex
        if size == 1:
            s = str(index)
        else:
            s = f"{index + size - 1}:{index}"
        return RenderResult(f"{self.src.Render(ctx).Line()}[{s}]")


class ArithmeticExpr(Expression):
    #XXX
    pass


class ConditionalExpr(Expression):
    #XXX
    pass


class Statement(SyntaxNode):
    #XXX
    pass


class Block(SyntaxNode):
    #XXX
    pass


class AssignmentStatement(Statement):
    #XXX
    pass


class IfStatement(Statement):
    #XXX
    pass


class CaseStatement(Statement):
    #XXX
    pass


class ProceduralBlock(Statement):
    #XXX
    pass

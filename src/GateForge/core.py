from dataclasses import dataclass
from io import TextIOBase
import io
import threading
from typing import Iterator, List, Optional, Tuple
import traceback
import re
import math
from pathlib import Path


class ParseException(Exception):
    pass


@dataclass
class WarningMsg:
    msg: str
    frame: Optional[traceback.FrameSummary]


    def __str__(self):
        if self.frame is not None:
            loc = f"{self.frame.filename}:{self.frame.lineno}"
            return f"WARN [{loc}] {self.msg}"
        return f"WARN {self.msg}"


class CompileCtx:

    lastFrame: Optional[traceback.FrameSummary] = None

    _threadLocal = threading.local()
    _curNetIdx: int
    _netNames: dict[str, "Net"]
    _ports: dict["Expression", "Port"]
    _blockStack: List["Block"]
    _warnings: List[WarningMsg]


    def __init__(self):
        self._curNetIdx = 0
        self._netNames = dict()
        self._ports = dict()
        self._blockStack = list()
        self._warnings = list()


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
    def Open(ctx: "CompileCtx", frameDepth: int):
        if CompileCtx._GetCurrent() is not None:
            raise Exception("Compilation context override")
        CompileCtx._SetCurrent(ctx)
        ctx._Open(frameDepth + 1)


    @staticmethod
    def Close():
        if CompileCtx._GetCurrent() is None:
            raise Exception("Compilation context not open")
        CompileCtx._SetCurrent(None)


    def Warning(self, msg: str, frame: Optional[traceback.FrameSummary] = None):
        #XXX
        if frame is None:
            frame = self.lastFrame
        wm = WarningMsg(msg, frame)
        self._warnings.append(wm)
        print(wm)


    def GenerateNetName(self, isReg: bool, initialName: Optional[str] = None) -> str:
        if initialName is not None:
            if initialName not in self._netNames:
                return initialName
            namePrefix = initialName
        elif isReg:
            namePrefix = "r"
        else:
            namePrefix = "w"

        while True:
            idx = self._curNetIdx
            self._curNetIdx += 1
            name = f"{namePrefix}_{idx}"
            if name in self._netNames:
                continue
            return name


    def RegisterNet(self, net: "Net"):
        f = self._netNames.get(net.name, None)
        if f is not None:
            raise ParseException(f"Net with name `{net.name}` already declared at {net.fullLocation}")
        self._netNames[net.name] = net


    def PushStatement(self, stmt: "Statement"):
        self.curBlock.PushStatement(stmt)


    @property
    def curBlock(self) -> "Block":
        if len(self._blockStack) == 0:
            raise Exception("Block stack underflow")
        return self._blockStack[-1]


    def _Open(self, frameDepth: int):
        self._blockStack.append(Block(frameDepth + 1))


@dataclass
class RenderOptions:
    sourceMap: bool = False


class RenderCtx:
    options: RenderOptions = RenderOptions()
    # Render declaration instead of expression when True
    renderDecl: bool = False

    output: TextIOBase


    def CreateNested(self, output: TextIOBase):
        ctx = RenderCtx()
        ctx.options = self.options
        ctx.renderDecl = self.renderDecl
        ctx.output = output
        return ctx


    def RenderNested(self, node: "SyntaxNode") -> str:
        with io.StringIO() as output:
            ctx = self.CreateNested(output)
            node.Render(ctx)
            return output.getvalue()


    def Write(self, s: str):
        self.output.write(s)


class SyntaxNode:
    # Stack frame of the Python source code for this node
    srcFrame: traceback.FrameSummary
    # String value to use in diagnostic messages
    strValue: Optional[str] = None


    def __init__(self, frameDepth: int):
        # Raise exception if no context
        ctx = CompileCtx.Current()
        self.srcFrame = self.GetFrame(frameDepth + 1)
        ctx.lastFrame = self.srcFrame

    @property
    def location(self) -> str:
        return f"{Path(self.srcFrame.filename).name}:{self.srcFrame.lineno}"


    @property
    def fullLocation(self) -> str:
        return f"{self.srcFrame.filename}:{self.srcFrame.lineno}"


    def __str__(self) -> str:
        if self.strValue is None:
            s = type(self).__name__
        else:
            s = self.strValue
        if self.srcFrame is not None:
            s += f"[{self.location}]"
        return s


    def GetFrame(self, frameDepth: int):
        return traceback.extract_stack()[-frameDepth - 2]


    def Render(self, ctx: RenderCtx):
        raise NotImplementedError()


class Expression(SyntaxNode):
    size: Optional[int] = None
    isLhs: bool = False
    # Expression is wired when used in some statement. Unwired expressions are dangling and do not
    # affect the produced output, however, may be used, for example, to define external module ports.
    isWired: bool = False
    wiringFrame: traceback.FrameSummary


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


    def _GetChildren(self) -> Iterator["Expression"]: # type: ignore
        "Should iterate child expression nodes"
        yield from ()


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        """
        Wire the expression.
        :return: True if wired, false if already was wired earlier.
        """
        for child in self._GetChildren():
            child._Wire(isLhs, frameDepth + 1)
        if isLhs and not self.isLhs:
            #XXX provide definition frames
            raise ParseException("Attempting to wire RHS expression as LHS, assignment target cannot be written to")
        if self.isWired:
            return False
        self.isWired = True
        self.wiringFrame = self.GetFrame(frameDepth + 1)
        return True


    def __ilshift__(self, rhs: "Expression | int"):
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=1)


    def __ifloordiv__(self, rhs: "Expression | int"):
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=1)


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


    def Render(self, ctx: RenderCtx):
        ctx.Write(f"{'' if self.size is None else self.size}'h{self.value:x}")


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
    # Name specified when net is created
    initialName: Optional[str] = None
    # Actual name is resolved when wired
    name: str

    _identPat = re.compile(r"[a-z_][a-z_$\d]*", re.RegexFlag.IGNORECASE)


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

        if name is not None:
            self._CheckIdentifier(name)
            self.initialName = name


    def __getitem__(self, s):
        # Need to adjust index according to baseIndex
        index, size = self._CheckSlice(s)
        if index < self.baseIndex:
            raise ParseException(f"Index is less than LSB index: {index} < {self.baseIndex}")
        return SliceExpr(self, index - self.baseIndex, size, 1)


    def Render(self, ctx: RenderCtx):
        name = self.name if self.isWired else self.initialName
        if name is None:
            raise Exception("Cannot render unwired unnamed net")
        if ctx.renderDecl:
            s = "reg" if self.isReg else "wire"
            assert self.size is not None
            if self.baseIndex != 0 or self.size > 1:
                s += f"[{self.baseIndex + self.size - 1}:{self.baseIndex}]"
            s += f" {name};"
            ctx.Write(s)
        else:
            ctx.Write(name)


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth):
            return False
        ctx = CompileCtx.Current()
        self.name = ctx.GenerateNetName(self.isReg, self.initialName)
        ctx.RegisterNet(self)
        return True


    @property
    def input(self):
        #XXX check frameDepth for property
        return Port(self, False, 1)


    @property
    def output(self):
        #XXX check frameDepth for property
        return Port(self, True, 1)


    def _CheckIdentifier(self, s: str):
        if self._identPat.fullmatch(s) is None:
            raise ParseException(f"Not a valid identifier: {s}")


class Wire(Net):
    isReg = False


class Reg(Net):
    isReg = True


class Port(Net):
    src: Net
    isOutput: bool

    def __init__(self, src: Net, isOutput: bool, frameDepth: int):
        if src.size is None:
            raise ParseException("Port cannot be created from unsized net")
        if src.initialName is None:
            raise ParseException("Port cannot be created from unnamed net")
        super().__init__(size=src.size, baseIndex=src.baseIndex, isReg=isOutput,
                         name=src.initialName, frameDepth=frameDepth + 1)
        self.src = src
        self.isLhs = isOutput

    #XXX

    #XXX wire exact name


class ConcatExpr(Expression):
    #XXX
    pass


class SliceExpr(Expression):
    src: Expression
    # Index is always zero-based, nets' base index is applied before if needed.
    index: int


    def __init__(self, src: Expression, index: int, size: int, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.src = src
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


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.src


    def Render(self, ctx: RenderCtx):
        index = self.index
        assert self.size is not None
        size = self.size
        if isinstance(self.src, Net):
            index += self.src.baseIndex
        if size == 1:
            s = str(index)
        else:
            s = f"{index + size - 1}:{index}"
        ctx.Write(f"{ctx.RenderNested(self.src)}[{s}]")


class ArithmeticExpr(Expression):
    #XXX
    pass


class ConditionalExpr(Expression):
    #XXX
    pass


class Statement(SyntaxNode):

    def __init__(self, frameDepth: int):
        super().__init__(frameDepth + 1)
        CompileCtx.Current().PushStatement(self)

    #XXX


class Block(SyntaxNode):
    _statements: List[Statement]


    def __init__(self, frameDepth: int):
        super().__init__(frameDepth + 1)
        self._statements = list()


    def PushStatement(self, stmt: Statement):
        self._statements.append(stmt)

    #XXX


class AssignmentStatement(Statement):
    lhs: Expression
    rhs: Expression
    isBlocking: bool

    def __init__(self, lhs: Expression, rhs: Expression | int, *, isBlocking: bool, frameDepth: int):
        super().__init__(frameDepth + 1)
        if isinstance(rhs, int):
            rhs = Const(rhs, frameDepth=frameDepth + 1)
        self.lhs = lhs
        self.rhs= rhs
        self.isBlocking = isBlocking
        lhs._Wire(True, frameDepth + 1)
        rhs._Wire(False, frameDepth + 1)

        assert lhs.size is not None
        if rhs.size is not None:
            if rhs.size > lhs.size:
                raise ParseException(f"Assignment size exceeded: {lhs.size} <<= {rhs.size}")
            elif rhs.size < lhs.size:
                CompileCtx.Current().Warning(f"Assignment of insufficient size: {lhs.size} <<= {rhs.size}",
                                             self.srcFrame)


    def Render(self, ctx: RenderCtx):
        #XXX check if in procedural block
        ctx.Write(f"assign {ctx.RenderNested(self.lhs)} = {ctx.RenderNested(self.rhs)};")


    #XXX


class IfStatement(Statement):
    #XXX
    pass


class CaseStatement(Statement):
    #XXX
    pass


class ProceduralBlock(Statement):
    #XXX
    pass

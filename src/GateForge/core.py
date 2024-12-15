from dataclasses import dataclass
from io import TextIOBase
import io
import threading
from typing import Iterable, Iterator, List, Optional, Tuple
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


@dataclass
class RenderOptions:
    indent: str = "    "
    sourceMap: bool = False


_identPat = re.compile(r"[a-z_][a-z_$\d]*", re.RegexFlag.IGNORECASE)
_keywords = [
    "always",
    "always_comb",
    "always_ff",
    "always_latch",
    "assign",
    "begin",
    "case",
    "else",
    "end",
    "endcase",
    "endfunction",
    "endmodule",
    "endprimitive",
    "endtable",
    "endtask",
    "enum",
    "for",
    "forever",
    "function",
    "if",
    "initial",
    "input",
    "integer",
    "localparam",
    "logic",
    "module",
    "negedge",
    "output",
    "parameter",
    "posedge",
    "primitive",
    "real",
    "reg",
    "repeat",
    "table",
    "task",
    "time",
    "timescale",
    "typedef",
    "while",
    "wire"
]

def _CheckIdentifier(s: str):
    if _identPat.fullmatch(s) is None:
        raise ParseException(f"Not a valid identifier: {s}")
    if s in _keywords:
        raise ParseException(f"Cannot use Verilog keyword as identifier: {s}")


class CompileCtx:

    lastFrame: Optional[traceback.FrameSummary] = None
    isProceduralBlock: bool = False

    _threadLocal = threading.local()
    _curNetIdx: int
    _nets: dict[str, "Net"]
    _ports: dict[str, "Port"]
    _blockStack: List["Block"]
    _warnings: List[WarningMsg]


    def __init__(self, moduleName: str):
        self._curNetIdx = 0
        self._nets = dict()
        self._ports = dict()
        self._blockStack = list()
        self._warnings = list()
        self.moduleName = moduleName
        _CheckIdentifier(moduleName)


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
            if initialName not in self._nets:
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
            if name in self._nets:
                continue
            return name


    def RegisterNet(self, net: "Net"):
        existing = self._nets.get(net.name, None)
        if existing is not None:
            raise ParseException(
                f"Net with name `{net.name}` already declared at {existing.fullLocation}, "
                f"redeclaration at {net.fullLocation}")
        self._nets[net.name] = net
        if isinstance(net, Port):
            self._ports[net.name] = net


    def PushStatement(self, stmt: "Statement"):
        self.curBlock.PushStatement(stmt)


    @property
    def curBlock(self) -> "Block":
        if len(self._blockStack) == 0:
            raise Exception("Block stack underflow")
        return self._blockStack[-1]


    def PushBlock(self, block: "Block"):
        self._blockStack.append(block)


    def PopBlock(self) -> "Block":
        if len(self._blockStack) < 2:
            raise Exception("Unexpected pop of root block")
        return self._blockStack.pop()


    @property
    def indent(self) -> int:
        return len(self._blockStack) - 1


    def _Open(self, frameDepth: int):
        self._blockStack.append(Block(frameDepth=frameDepth + 1))


    def Render(self, output: TextIOBase, renderOptions: RenderOptions):
        if len(self._blockStack) != 1:
            raise Exception(f"Unexpected block stack size: {len(self._blockStack)}")

        ctx = RenderCtx()
        ctx.options = renderOptions
        ctx.output = output

        ctx.renderDecl = True
        self._RenderModuleDeclaration(ctx)
        self._RenderNetsDeclarations(ctx)
        ctx.Write("\n")

        ctx.renderDecl = False
        self._blockStack[0].Render(ctx)

        ctx.Write("endmodule\n")


    def GetWarnings(self) -> Iterable[WarningMsg]:
        return self._warnings


    def _RenderModuleDeclaration(self, ctx: "RenderCtx"):
        ctx.Write(f"module {self.moduleName}(")
        isFirst = True
        for port in sorted(self._ports.values(), key=lambda p: p.name):
            if not isFirst:
                ctx.Write(",\n")
                ctx.Write(ctx.options.indent)
            else:
                ctx.Write("\n")
                ctx.Write(ctx.options.indent)
                isFirst = False
            port.Render(ctx)
        ctx.Write(");\n")


    def _RenderNetsDeclarations(self, ctx: "RenderCtx"):
        for net in sorted(self._nets.values(), key=lambda n: n.name):
            if isinstance(net, Port):
                continue
            net.Render(ctx)
            ctx.Write(";\n")


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


    def WriteIndent(self, indent: int):
        self.Write(self.options.indent * indent)


class SyntaxNode:
    # Stack frame of the Python source code for this node
    srcFrame: traceback.FrameSummary
    # String value to use in diagnostic messages
    strValue: Optional[str] = None
    indent: int


    def __init__(self, frameDepth: int):
        # Raise exception if no context
        ctx = CompileCtx.Current()
        self.srcFrame = self.GetFrame(frameDepth + 1)
        ctx.lastFrame = self.srcFrame
        self.indent = ctx.indent


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


    def _GetLeafNodes(self) -> Iterator["Expression"]:
        hasChildren = False
        for child in self._GetChildren():
            hasChildren = True
            yield from child._GetLeafNodes()
        if not hasChildren:
            yield self


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        """
        Wire the expression.
        :return: True if wired, false if already was wired earlier.
        """
        if isLhs and not self.isLhs:
            raise ParseException(
                "Attempting to wire RHS expression as LHS, assignment target cannot be written to\n" +
                self._DescribeNonLhs())
        if self.isWired:
            return False
        for child in self._GetChildren():
            child._Wire(isLhs, frameDepth + 1)
        self.isWired = True
        self.wiringFrame = self.GetFrame(frameDepth + 1)
        return True


    def _DescribeNonLhs(self, indent: int =  0) -> str:
        """Get structure details for non-LHS expression used in LHS context"""
        assert not self.isLhs
        s = f"{indent * '    '}Non-LHS {self}"
        for e in self._GetChildren():
            if e.isLhs:
                continue
            s += f"\n{e._DescribeNonLhs(indent + 1)}"
        return s


    def __ilshift__(self, rhs: "Expression | int"):
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=1)


    def __ifloordiv__(self, rhs: "Expression | int"):
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=1)


    def assign(self, rhs: "Expression | int"):
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=1)


    def bassign(self, rhs: "Expression | int"):
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=1)


    def __mod__(self, rhs: "Expression | int"):
        return ConcatExpr((self, rhs), 1)


    def __or__(self, rhs: "Expression | int | SensitivityList"):
        if isinstance(rhs, SensitivityList):
            return rhs._Combine(self, 1)
        return ArithmeticExpr("|", self, rhs, 1)


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
        self.strValue = f"{'Reg' if isReg else 'Wire'}"
        if name is not None:
            self.strValue += f"(`{name}`)"

        if name is not None:
            _CheckIdentifier(name)
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
            s += f" {name}"
            ctx.Write(s)
        else:
            ctx.Write(name)


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth + 1):
            return False
        ctx = CompileCtx.Current()
        # Ports have fixed names, so it is initialized in constructor
        if not hasattr(self, "name"):
            self.name = ctx.GenerateNetName(self.isReg, self.initialName)
        ctx.RegisterNet(self)
        return True


    @property
    def input(self) -> "Port":
        return Port(self, False, 1)


    @property
    def output(self) -> "Port":
        return Port(self, True, 1)


    @property
    def posedge(self) -> "EdgeTrigger":
        return EdgeTrigger(self, True, 1)


    @property
    def negedge(self) -> "EdgeTrigger":
        return EdgeTrigger(self, False, 1)



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
        self.isOutput = isOutput
        self.name = src.initialName
        self.strValue = f"{'Output' if isOutput else 'Input'}(`{self.name}`)"


    # Treating source Net as absorbed so it is not enumerated as child.


    def Render(self, ctx: RenderCtx):
        if ctx.renderDecl:
            ctx.Write("output" if self.isOutput else "input")
            ctx.Write(" ")
            super().Render(ctx)
        else:
            super().Render(ctx)


class ConcatExpr(Expression):
    src: List[Expression]

    def __init__(self, src: Iterable[Expression | int], frameDepth: int):
        super().__init__(frameDepth + 1)
        self.src = list(ConcatExpr._FlattenConcat(src, frameDepth + 1))
        self._CalculateSize()


    @staticmethod
    def _FlattenConcat(src: Iterable[Expression | int], frameDepth: int) -> Iterator[Expression]:
        """
        Flatten concatenation of concatenations as much as possible
        """
        for e in src:
            if isinstance(e, ConcatExpr):
                yield from ConcatExpr._FlattenConcat(e.src, frameDepth + 1)
            elif isinstance(e, int):
                yield Const(e, frameDepth=frameDepth + 1)
            else:
                yield e


    def _CalculateSize(self):
        size = 0
        isFirst = True
        isLhs = True
        for e in self.src:
            if isFirst:
                isFirst = False
                if e.size is None:
                    size = None
            else:
                if e.size is None:
                    raise ParseException(
                        "Concatenation can have expression with unbound size on left-most position"
                         f" only, unbound expression: {e}")
                if size is not None:
                    size += e.size
            if not e.isLhs:
                isLhs = False
        self.size = size
        self.isLhs = isLhs


    def _GetChildren(self) -> Iterator["Expression"]:
        yield from self.src


    def Render(self, ctx: RenderCtx):
        ctx.Write("{")
        isFirst = True
        for e in self.src:
            if isFirst:
                isFirst = False
            else:
                ctx.Write(", ")
            e.Render(ctx)
        ctx.Write("}")


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
    op: str
    lhs: Expression
    rhs: Expression


    def __init__(self, op: str, lhs: Expression, rhs: Expression | int, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Op({op})"
        self.op = op
        self.lhs = lhs
        if isinstance(rhs, int):
            self.rhs = Const(rhs, frameDepth=frameDepth + 1)
        else:
            self.rhs = rhs


    def _ToSensitivityList(self, frameDepth: int) -> "SensitivityList":
        if self.op != "|":
            raise ParseException(f"Only `|` operation allowed for sensitivity list, has `{self.op}`")

        sl = SensitivityList(frameDepth + 1)

        if isinstance(self.lhs, Net):
            sl.PushSignal(self.lhs)
        elif isinstance(self.lhs, ArithmeticExpr):
            sl.signals.extend(self.lhs._ToSensitivityList(frameDepth + 1).signals)
        else:
            raise ParseException(f"Bad item for sensitivity list: {self.lhs}")

        if isinstance(self.rhs, Net):
            sl.PushSignal(self.rhs)
        elif isinstance(self.rhs, ArithmeticExpr):
            sl.signals.extend(self.rhs._ToSensitivityList(frameDepth + 1).signals)
        else:
            raise ParseException(f"Bad item for sensitivity list: {self.rhs}")

        return sl

    #XXX


class ConditionalExpr(Expression):
    #XXX
    pass


class Statement(SyntaxNode):

    def __init__(self, frameDepth: int, deferPush: bool = False):
        super().__init__(frameDepth + 1)
        if not deferPush:
            CompileCtx.Current().PushStatement(self)

    #XXX


class Block(SyntaxNode):
    _statements: List[Statement]


    def __init__(self, frameDepth: int):
        super().__init__(frameDepth + 1)
        self._statements = list()


    def __len__(self):
        return len(self._statements)


    def PushStatement(self, stmt: Statement):
        self._statements.append(stmt)


    def Render(self, ctx: RenderCtx):
        for stmt in self._statements:
            ctx.WriteIndent(stmt.indent)
            stmt.Render(ctx)
            ctx.Write("\n")


class AssignmentStatement(Statement):
    lhs: Expression
    rhs: Expression
    isBlocking: bool
    isProceduralBlock: bool

    def __init__(self, lhs: Expression, rhs: Expression | int, *, isBlocking: bool, frameDepth: int):
        super().__init__(frameDepth + 1)
        if isinstance(rhs, int):
            rhs = Const(rhs, frameDepth=frameDepth + 1)
        self.lhs = lhs
        self.rhs= rhs
        self.isBlocking = isBlocking
        self.isProceduralBlock = CompileCtx.Current().isProceduralBlock
        lhs._Wire(True, frameDepth + 1)
        rhs._Wire(False, frameDepth + 1)

        if self.isProceduralBlock and not self.isBlocking:
            for e in self.lhs._GetLeafNodes():
                if isinstance(e, Net) and not e.isReg:
                    raise ParseException(f"Procedural assignment to wire {e}")

        assert lhs.size is not None
        if rhs.size is not None:
            if rhs.size > lhs.size:
                raise ParseException(f"Assignment size exceeded: {lhs.size} <<= {rhs.size}")
            elif rhs.size < lhs.size:
                CompileCtx.Current().Warning(f"Assignment of insufficient size: {lhs.size} <<= {rhs.size}",
                                             self.srcFrame)


    def Render(self, ctx: RenderCtx):
        if self.isProceduralBlock:
            op = "=" if self.isBlocking else "<="
            ctx.Write(f"{ctx.RenderNested(self.lhs)} {op} {ctx.RenderNested(self.rhs)};")
        else:
            ctx.Write(f"assign {ctx.RenderNested(self.lhs)} = {ctx.RenderNested(self.rhs)};")


class IfStatement(Statement):
    #XXX
    pass


class CaseStatement(Statement):
    #XXX
    pass


class EdgeTrigger(SyntaxNode):
    net: Net
    isPositive: bool


    def __init__(self, net: Net, isPositive: bool, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.net = net
        self.isPositive = isPositive


    def Render(self, ctx: RenderCtx):
        ctx.Write("posedge" if self.isPositive else "negedge")
        ctx.Write(" ")
        self.net.Render(ctx)


    def _Wire(self, frameDepth: int) -> bool:
        return self.net._Wire(False, frameDepth + 1)


    def __or__(self, rhs: "EdgeTrigger") -> "SensitivityList":
        sl = SensitivityList(1)
        sl.PushSignal(self)
        sl.PushSignal(rhs)
        return sl


class SensitivityList(SyntaxNode):
    signals: List[EdgeTrigger | Net]


    def __init__(self, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.signals = list()


    def PushSignal(self, signal: Net | EdgeTrigger):
        if not isinstance(signal, Net) and not isinstance(signal, EdgeTrigger):
            raise ParseException(f"Unexpected item in a sensitivity list: {signal}")
        self.signals.append(signal)


    def _Wire(self, frameDepth: int):
        isEdge = None
        for sig in self.signals:
            if isinstance(sig, Net):
                if isEdge is not None and isEdge:
                    raise ParseException("Mixed edge and activity triggers")
                isEdge = False
                sig._Wire(False, frameDepth + 1)
            else:
                if isEdge is not None and not isEdge:
                    raise ParseException("Mixed edge and activity triggers")
                isEdge = True
                sig._Wire(frameDepth + 1)


    def Render(self, ctx: RenderCtx):
        isFirst = True
        for sig in self.signals:
            if isFirst:
                isFirst = False
            else:
                ctx.Write(", ")
            sig.Render(ctx)


    def _Combine(self, rhs: "Expression | SensitivityList | EdgeTrigger", frameDepth: int):
        sl = SensitivityList(frameDepth + 1)
        sl.signals.extend(self.signals)
        if isinstance(rhs, SensitivityList):
            sl.signals.extend(rhs.signals)
        elif isinstance(rhs, ArithmeticExpr):
            sl.signals.extend(rhs._ToSensitivityList(frameDepth + 1).signals)
        elif isinstance(rhs, EdgeTrigger):
            sl.signals.append(rhs)
        elif isinstance(rhs, Net):
            raise ParseException(f"Cannot mix edge and activity triggers: {rhs}")
        else:
            raise ParseException(f"Bad item for sensitivity list: {rhs}")
        return sl


    def __or__(self, rhs: "Expression | SensitivityList | EdgeTrigger"):
        return self._Combine(rhs, 1)


class ProceduralBlock(Statement):
    sensitivityList: Optional[SensitivityList]
    body: Block


    def __init__(self, sensitivityList: Optional[SensitivityList], frameDepth: int):
        super().__init__(frameDepth + 1, deferPush=True)
        self.sensitivityList = sensitivityList


    def __enter__(self):
        self.body = Block(1)
        ctx = CompileCtx.Current()
        ctx.PushBlock(self.body)
        ctx.isProceduralBlock = True


    def __exit__(self, excType, excValue, tb):
        ctx = CompileCtx.Current()
        if ctx.PopBlock() is not self.body:
            raise Exception("Unexpected current block")
        ctx.isProceduralBlock = False
        if len(self.body) == 0:
            ctx.Warning(f"Empty procedural block", self.srcFrame)
        else:
            if self.sensitivityList is not None:
                self.sensitivityList._Wire(1)
            ctx.PushStatement(self)


    def Render(self, ctx: RenderCtx):
        ctx.Write("always @")
        if self.sensitivityList is None:
            ctx.Write("*")
        else:
            ctx.Write("(")
            self.sensitivityList.Render(ctx)
            ctx.Write(")")
        ctx.Write(" begin\n")
        self.body.Render(ctx)
        ctx.Write("end\n")

    #XXX

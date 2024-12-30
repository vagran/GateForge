from dataclasses import dataclass
from enum import Enum
from io import TextIOBase
import threading
from typing import Any, Iterable, Iterator, List, Optional, Tuple
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
    moduleName: str
    lastFrame: Optional[traceback.FrameSummary] = None
    isProceduralBlock: bool = False
    isInitialBlock = False

    _threadLocal = threading.local()
    _curNetIdx: int
    _curModuleIdx: int
    _nets: dict[str, "Net"]
    _ports: dict[str, "Port"]
    _modules: dict[str, "Module"]
    _blockStack: List["Block"]
    _warnings: List[WarningMsg]
    _namespace: List[str]


    def __init__(self, moduleName: str):
        self._curNetIdx = 0
        self._curModuleIdx = 0
        self._nets = dict()
        self._ports = dict()
        self._modules = dict()
        self._blockStack = list()
        self._warnings = list()
        self._namespace = list()
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


    def GenerateNetName(self, isReg: bool, initialName: Optional[str] = None,
                        namespacePrefix: Optional[str] = None) -> str:

        if namespacePrefix is None:
            namespacePrefix = ""
        if initialName is not None:
            namePrefix = initialName
            if namespacePrefix + namePrefix not in self._nets:
                return namePrefix
        elif isReg:
            namePrefix = "r"
        else:
            namePrefix = "w"

        while True:
            idx = self._curNetIdx
            self._curNetIdx += 1
            name = f"{namePrefix}_{idx}"
            fullName = namespacePrefix + name
            if fullName in self._nets or fullName in self._modules:
                continue
            return name


    def GenerateModuleInstanceName(self, moduleName: str,
                                   namespacePrefix: Optional[str] = None) -> str:
        while True:
            idx = self._curModuleIdx
            self._curModuleIdx += 1
            name = f"{moduleName}_{idx}"
            if namespacePrefix is None:
                fullName = name
            else:
                fullName = namespacePrefix + name
            if fullName in self._nets or fullName in self._modules:
                continue
            return name


    def RegisterNet(self, net: "Net"):
        name = net.fullName
        existing = self._nets.get(name, None)
        if existing is not None:
            raise ParseException(
                f"Net with name `{name}` already declared at {existing.fullLocation}, "
                f"redeclaration at {net.fullLocation}")
        self._nets[name] = net
        if isinstance(net, Port):
            self._ports[name] = net


    def RegisterModule(self, module: "Module"):
        existing = self._modules.get(module.name, None)
        if existing is not None:
            raise ParseException(
                f"Module with name `{module.name}` already declared at {existing.fullLocation}")
        self._modules[module.name] = module


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


    def PushNamespace(self, name: str):
        _CheckIdentifier(name)
        self._namespace.append(name)


    def PopNamespace(self) -> str:
        if len(self._namespace) == 0:
            raise Exception("Namespace stack underflow")
        return self._namespace.pop()


    @property
    def namespacePrefix(self) -> str:
        return "".join(map(lambda n: n + "_", self._namespace))


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


    def GetPortNames(self) -> Iterable[str]:
        return self._ports.keys()


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
    namespacePrefix: str


    def __init__(self, frameDepth: int):
        # Raise exception if no context
        ctx = CompileCtx.Current()
        self.srcFrame = self.GetFrame(frameDepth + 1)
        ctx.lastFrame = self.srcFrame
        self.indent = ctx.indent
        self.namespacePrefix = ctx.namespacePrefix


    @staticmethod
    def GetLocation(frame: traceback.FrameSummary) -> str:
        return f"{Path(frame.filename).name}:{frame.lineno}"


    @staticmethod
    def GetFullLocation(frame: traceback.FrameSummary) -> str:
        return f"{frame.filename}:{frame.lineno}"


    @property
    def location(self) -> str:
        return SyntaxNode.GetLocation(self.srcFrame)


    @property
    def fullLocation(self) -> str:
        return SyntaxNode.GetFullLocation(self.srcFrame)


    def __str__(self) -> str:
        if self.strValue is None:
            s = type(self).__name__
        else:
            s = self.strValue
        if self.srcFrame is not None:
            s += f"[{self.location}]"
        return s


    def GetFrame(self, frameDepth: int) -> traceback.FrameSummary:
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
    nonWireableReason: Optional[str] = None
    needParentheses: bool = False


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


    def __getitem__(self, s) -> "SliceExpr":
        index, size = self._CheckSlice(s)
        return SliceExpr(self, index, size, 1)


    def __setitem__(self, idx, value):
        if not isinstance(value, SliceExpr):
            raise ParseException("Slice item can be only set by `<<=` or `//=` operator.")


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
        if self.nonWireableReason is not None:
            raise ParseException(f"Expression wiring prohibited: {self} - {self.nonWireableReason}")
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


    def _Assign(self, bitIndex: Optional[int], frameDepth: int):
        """Called to mark assignment to the expression.

        :param bitIndex: Bit index which is driven. None if whole expression affected.
        """
        assert self.isWired
        assert self.isLhs
        # Should be overridden


    def _DescribeNonLhs(self, indent: int =  0) -> str:
        """Get structure details for non-LHS expression used in LHS context"""
        assert not self.isLhs
        s = f"{indent * '    '}Non-LHS {self}"
        for e in self._GetChildren():
            if e.isLhs:
                continue
            s += f"\n{e._DescribeNonLhs(indent + 1)}"
        return s


    @staticmethod
    def _CheckType(e: "Expression") -> "Expression":
        if not isinstance(e, Expression):
            raise ParseException(f"Expected instance of `Expression`, has `{type(e).__name__}`")
        return e


    @staticmethod
    def _FromRaw(e: "RawExpression", frameDepth: int) -> "Expression":
        if isinstance(e, int):
            return Const(e, None, frameDepth=frameDepth + 1)
        if isinstance(e, bool):
            return Const(e, None, frameDepth=frameDepth + 1)
        return Expression._CheckType(e)


    def RenderNested(self, ctx: RenderCtx):
        if self.needParentheses:
            ctx.Write("(")
        self.Render(ctx)
        if self.needParentheses:
            ctx.Write(")")


    def __ilshift__(self, rhs: "RawExpression") -> "Expression":
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=1)
        return self


    def __ifloordiv__(self, rhs: "RawExpression") -> "Expression":
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=1)
        return self


    def assign(self, rhs: "RawExpression"):
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=1)


    def bassign(self, rhs: "RawExpression"):
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=1)


    def __mod__(self, rhs: "RawExpression") -> "ConcatExpr":
        return ConcatExpr((self, rhs), 1)


    def __or__(self, rhs: "RawExpression | SensitivityList") -> "ArithmeticExpr | SensitivityList":
        if isinstance(rhs, SensitivityList):
            return rhs._Combine(self, 1)
        return ArithmeticExpr("|", (self, rhs), 1)


    def __and__(self, rhs: "RawExpression") -> "ArithmeticExpr":
        return ArithmeticExpr("&", (self, rhs), 1)


    def __xor__(self, rhs: "RawExpression") -> "ArithmeticExpr":
        return ArithmeticExpr("^", (self, rhs), 1)


    def xnor(self, rhs: "RawExpression") -> "ArithmeticExpr":
        return ArithmeticExpr("~^", (self, rhs), 1)


    def __invert__(self) -> "UnaryOperator":
        return UnaryOperator("~", self, 1)


    def __add__(self, rhs: "RawExpression") -> "ArithmeticExpr":
        return ArithmeticExpr("+", (self, rhs), 1)


    def __sub__(self, rhs: "RawExpression") -> "ArithmeticExpr":
        return ArithmeticExpr("-", (self, rhs), 1)


    @property
    def reduce_and(self) -> "ReductionOperator":
        return ReductionOperator("&", self, 1)


    @property
    def reduce_nand(self) -> "ReductionOperator":
        return ReductionOperator("~&", self, 1)


    @property
    def reduce_or(self) -> "ReductionOperator":
        return ReductionOperator("|", self, 1)


    @property
    def reduce_nor(self) -> "ReductionOperator":
        return ReductionOperator("~|", self, 1)


    @property
    def reduce_xor(self) -> "ReductionOperator":
        return ReductionOperator("^", self, 1)


    @property
    def reduce_xnor(self) -> "ReductionOperator":
        return ReductionOperator("~^", self, 1)


    def __lt__(self, rhs: "RawExpression") -> "ComparisonExpr":
        return ComparisonExpr("<", self, rhs, 1)


    def __gt__(self, rhs: "RawExpression") -> "ComparisonExpr":
        return ComparisonExpr(">", self, rhs, 1)


    def __le__(self, rhs: "RawExpression") -> "ComparisonExpr":
        return ComparisonExpr("<=", self, rhs, 1)


    def __ge__(self, rhs: "RawExpression") -> "ComparisonExpr":
        return ComparisonExpr(">=", self, rhs, 1)


    def __eq__(self, rhs: "RawExpression") -> "ComparisonExpr": # type: ignore
        return ComparisonExpr("==", self, rhs, 1)


    def __ne__(self, rhs: "RawExpression") -> "ComparisonExpr": # type: ignore
        return ComparisonExpr("!=", self, rhs, 1)


    def replicate(self, count: int) -> "ReplicationOperator":
        return ReplicationOperator(self, count, 1)


    def cond(self, ifCase: "RawExpression", elseCase: "RawExpression") -> "ConditionalExpr":
        return ConditionalExpr(self, ifCase, elseCase, 1)


RawExpression = Expression | int | bool


class Const(Expression):
    # Minimal number of bits required to present the value without trimming
    valueSize: int
    # Constant value
    value: int

    _valuePat = re.compile(r"(?:(\d+)?'([bdoh]))?([\da-f_]+)", re.RegexFlag.IGNORECASE)


    def __init__(self, value: str | int | bool, size: Optional[int] = None, *, frameDepth: int):
        super().__init__(frameDepth + 1)

        if isinstance(value, str):
            if size is not None:
                raise ParseException("Size should not be specified for string value")
            self.value, self.size = Const._ParseStringValue(value)
        elif isinstance(value, bool):
            self.value = 1 if value else 0
            self.size = 1
        else:
            self.value = value
            self.size = size

        if self.value < 0:
            raise ParseException("Negative values not allowed")

        self.valueSize = Const.GetMinValueBits(self.value)

        if self.size is not None and self.valueSize > self.size:
            trimmed = self.value & ((1 << self.size) - 1)
            CompileCtx.Current().Warning(
                f"Constant explicit size less than value required size: {self.size} < {self.valueSize}, "
                f"value trimmed {self.value} => {trimmed}")
            self.value = trimmed

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
    size: int
    isLhs = True
    baseIndex: int = 0
    isReg: bool
    # Name specified when net is created
    initialName: Optional[str] = None
    # Actual name is resolved when wired
    name: str
    # Bits assignments map
    assignments: List[Optional[traceback.FrameSummary]]


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
            _CheckIdentifier(name)
            self.initialName = name
            self.strValue += f"(`{name}`)"

        self.assignments = [None for i in range(size)]


    def __getitem__(self, s):
        # Need to adjust index according to baseIndex
        index, size = self._CheckSlice(s)
        if index < self.baseIndex:
            raise ParseException(f"Index is less than LSB index: {index} < {self.baseIndex}")
        return SliceExpr(self, index - self.baseIndex, size, 1)


    @property
    def effectiveName(self):
        return self.name if self.isWired else self.initialName


    @property
    def fullName(self):
        if not hasattr(self, "name"):
            raise Exception("Name not yet resolved")
        return self.namespacePrefix + self.name


    def SetName(self, name: str):
        if hasattr(self, "name"):
            raise Exception("Cannot set net name after it has been resolved")
        self.initialName = name


    def Render(self, ctx: RenderCtx):
        name = self.effectiveName
        if name is None:
            raise Exception("Cannot render unwired unnamed net")
        if ctx.renderDecl:
            # Port render method prepends this declaration
            if ctx.options.sourceMap and type(self) is not Port:
                ctx.Write(f"// {self.fullLocation}\n")
                ctx.WriteIndent(self.indent)
            s = "reg" if self.isReg else "wire"
            assert self.size is not None
            if self.baseIndex != 0 or self.size > 1:
                s += f"[{self.baseIndex + self.size - 1}:{self.baseIndex}]"
            s += f" {self.namespacePrefix}{name}"
            ctx.Write(s)
        else:
            ctx.Write(self.namespacePrefix)
            ctx.Write(name)


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth + 1):
            return False
        ctx = CompileCtx.Current()
        # Ports have fixed names, so it is initialized in constructor
        if not hasattr(self, "name"):
            self.name = ctx.GenerateNetName(self.isReg, self.initialName, self.namespacePrefix)
        ctx.RegisterNet(self)
        return True


    def _Assign(self, bitIndex: Optional[int], frameDepth: int):

        def _AssignBit(bitIndex: int, frame: traceback.FrameSummary):
            prevFrame = self.assignments[bitIndex]
            # Just track last assignment for register for now, for some future use.
            if not self.isReg and prevFrame is not None:
                raise ParseException(
                    f"Wire re-assignment, `{self}[{bitIndex}]` was previously assigned at " +
                    SyntaxNode.GetFullLocation(prevFrame))
            self.assignments[bitIndex] = frame

        frame = self.GetFrame(frameDepth + 1)
        if bitIndex is None:
            for i in range(self.size):
                _AssignBit(i, frame)
        else:
            _AssignBit(bitIndex, frame)


    @property
    def input(self) -> "NetProxy":
        return NetProxy(self, False, 1)


    @property
    def output(self) -> "NetProxy":
        return NetProxy(self, True, 1)


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


class NetProxy(Net):
    src: Net
    isOutput: bool


    def __init__(self, src: Net, isOutput: bool, frameDepth: int):
        if not isinstance(src, Net):
            raise ParseException(f"Net type expected, has `{type(src).__name__}`")
        self.isOutput = isOutput
        super().__init__(size=src.size, baseIndex=src.baseIndex, isReg=src.isReg,
                         name=src.initialName, frameDepth=frameDepth + 1)
        self.src = src
        self.isLhs = isOutput


    @property
    def name(self):
        return self.src.name


    @property
    def strValue(self):
        name = self.effectiveName
        result = "Output" if self.isOutput else "Input"
        result += "Reg" if self.isReg else "Wire"
        if name is None:
            return result
        return f"{result}(`{name}`)"

    @strValue.setter
    def strValue(self, value):
        # Ignore setting in base constructor
        pass


    def SetName(self, name: str):
        self.src.SetName(name)
        self.initialName = name


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.src


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        # Skip Net._Wire() implementation since `name` handling should be done in proxied object
        # only.
        return Expression._Wire(self, isLhs, frameDepth + 1)


    @property
    def input(self) -> "NetProxy":
        if self.isOutput:
            return NetProxy(self.src, False, 1)
        return self


    @property
    def output(self) -> "NetProxy":
        if self.isOutput:
            return self
        raise ParseException(f"Cannot use input net as output: {self}")


    @property
    def port(self) -> "Port":
        return Port(self, 1)


class Port(Net):
    src: NetProxy
    isOutput: bool


    def __init__(self, src: NetProxy, frameDepth: int):
        if not isinstance(src, NetProxy):
            raise ParseException(f"NetProxy type expected, has `{type(src).__name__}`")
        if isinstance(src.src, Port):
            raise ParseException(f"Cannot take port from port: {src.src}")
        if src.size is None:
            raise ParseException("Port cannot be created from unsized net")
        if src.initialName is None:
            raise ParseException("Port cannot be created from unnamed net")
        if src.src.isWired:
            raise ParseException("Port cannot be created from wired net, wired at " +
                                 SyntaxNode.GetFullLocation(src.src.wiringFrame))
        super().__init__(size=src.size, baseIndex=src.baseIndex, isReg=src.isReg,
                         name=src.initialName, frameDepth=frameDepth + 1)
        self.src = src
        self.isLhs = src.isOutput
        self.isOutput = src.isOutput
        # Resolved name should be identical to the specified one. Conflict, if any, will be detected
        # on wiring.
        self.name = src.initialName
        # Source is absorbed and should never be wired.
        src.nonWireableReason = f"Used as port at {SyntaxNode.GetFullLocation(self.srcFrame)}"
        src.src.nonWireableReason = src.nonWireableReason
        self.strValue = f"{'Output' if self.isOutput else 'Input'}(`{self.name}`)"


    # Treating source Net as absorbed so it is not enumerated as child.


    def Render(self, ctx: RenderCtx):
        if ctx.renderDecl:
            if ctx.options.sourceMap:
                ctx.Write(f"// {self.fullLocation}\n")
                ctx.WriteIndent(self.indent)
            ctx.Write("output" if self.src.isOutput else "input")
            ctx.Write(" ")
            super().Render(ctx)
        else:
            super().Render(ctx)


    @property
    def input(self) -> "NetProxy":
        return NetProxy(self, False, 1)


    @property
    def output(self) -> "NetProxy":
        if not self.isOutput:
            raise ParseException(f"Cannot use input port as output: {self}")
        return NetProxy(self, True, 1)


class ConcatExpr(Expression):
    # Left to right
    args: List[Expression]
    # Minimal number of bits required to present the value without trimming. Useful when having
    # left-most const value with unbound size.
    valueSize: int

    def __init__(self, args: Iterable[RawExpression], frameDepth: int):
        super().__init__(frameDepth + 1)
        self.args = list(ConcatExpr._FlattenConcat(args, frameDepth + 1))
        self._CalculateSize()


    @staticmethod
    def _FlattenConcat(src: Iterable[RawExpression], frameDepth: int) -> Iterator[Expression]:
        """
        Flatten concatenation of concatenations as much as possible
        """
        for e in src:
            if isinstance(e, ConcatExpr):
                yield from ConcatExpr._FlattenConcat(e.args, frameDepth + 1)
            else:
                yield Expression._FromRaw(e, frameDepth + 1)


    def _CalculateSize(self):
        size = 0
        valueSize = 0
        isFirst = True
        isLhs = True
        for e in self.args:
            if isFirst:
                isFirst = False
                if e.size is None:
                    size = None
                    if hasattr(e, "valueSize"):
                        valueSize = e.valueSize
                    else:
                        valueSize = None
                    continue
            else:
                if e.size is None:
                    raise ParseException(
                        "Concatenation can have expression with unbound size on left-most position"
                         f" only, unbound expression: {e}")
            if size is not None:
                size += e.size
            if valueSize is not None:
                valueSize += e.size
            if not e.isLhs:
                isLhs = False
        self.size = size
        self.isLhs = isLhs
        if valueSize is not None:
            self.valueSize = valueSize


    def _GetChildren(self) -> Iterator["Expression"]:
        yield from self.args


    def _Assign(self, bitIndex: Optional[int], frameDepth: int):
        if bitIndex is None:
            for arg in self.args:
                arg._Assign(None, frameDepth + 1)
        else:
            index = 0
            for arg in reversed(self.args):
                if arg.size is None or bitIndex < index + arg.size:
                    assert bitIndex >= index
                    arg._Assign(bitIndex - index, frameDepth + 1)
                    return
                index += arg.size
            raise Exception(f"Assignment index out of range: {bitIndex}")


    def Render(self, ctx: RenderCtx):
        ctx.Write("{")
        isFirst = True
        for e in self.args:
            if isFirst:
                isFirst = False
            else:
                ctx.Write(", ")
            e.Render(ctx)
        ctx.Write("}")


class SliceExpr(Expression):
    size: int
    arg: Expression
    # Index is always zero-based, nets' base index is applied before if needed.
    index: int


    def __init__(self, arg: Expression, index: int, size: int, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.arg = Expression._CheckType(arg)
        self.isLhs = self.arg.isLhs
        if self.arg.size is not None:
            self._CheckRange(index, size, self.arg.size)
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
        return SliceExpr(self.arg, self.index + index, size, 1)


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.arg


    def _Assign(self, bitIndex: Optional[int], frameDepth: int):
        if bitIndex is None:
            for i in range(self.index, self.index + self.size):
                self.arg._Assign(i, frameDepth + 1)
        else:
            self.arg._Assign(self.index + bitIndex, frameDepth + 1)


    def Render(self, ctx: RenderCtx):
        index = self.index
        assert self.size is not None
        size = self.size
        if isinstance(self.arg, Net):
            index += self.arg.baseIndex
        if size == 1:
            s = str(index)
        else:
            s = f"{index + size - 1}:{index}"
        self.arg.RenderNested(ctx)
        ctx.Write(f"[{s}]")


class ArithmeticExpr(Expression):
    op: str
    args: List[Expression]
    needParentheses = True


    def __init__(self, op: str, args: Iterable[RawExpression], frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Op({op})"
        self.op = op
        self.args = list(self._FlattenArithmeticExpr(args, frameDepth + 1))
        self.size = self._CalculateSize()


    def _FlattenArithmeticExpr(self, src: Iterable[RawExpression], frameDepth: int) -> Iterator[Expression]:
        """
        Flatten combine nested expressions as much as possible
        """
        for e in src:
            if isinstance(e, ArithmeticExpr):
                if e.op == self.op:
                    yield from self._FlattenArithmeticExpr(e.args, frameDepth + 1)
                else:
                    yield e
            else:
                yield Expression._FromRaw(e, frameDepth + 1)


    def _CalculateSize(self):
        size = None
        for e in self.args:
            if size is not None and (size is None or e.size > size):
                size = e.size
        self.size = size


    def _ToSensitivityList(self, frameDepth: int) -> "SensitivityList":
        if self.op != "|":
            raise ParseException(f"Only `|` operation allowed for sensitivity list, has `{self.op}`")

        sl = SensitivityList(frameDepth + 1)
        for e in self.args:
            if isinstance(e, Net):
                sl.PushSignal(e)
            else:
                raise ParseException(f"Bad item for sensitivity list: {e}")

        return sl


    def _GetChildren(self) -> Iterator["Expression"]:
        yield from self.args


    def Render(self, ctx: RenderCtx):
        isFirst = True
        for e in self.args:
            if isFirst:
                isFirst = False
            else:
                ctx.Write(f" {self.op} ")
            e.RenderNested(ctx)


class ComparisonExpr(Expression):
    op: str
    rhs: Expression
    lhs: Expression
    needParentheses = True
    size = 1


    def __init__(self, op: str, lhs: Expression, rhs: RawExpression, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Cmp({op})"
        self.op = op
        self.lhs = Expression._CheckType(lhs)
        self.rhs = Expression._FromRaw(rhs, frameDepth + 1)


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth + 1):
            return False

        lhsSize = self.lhs.size
        if lhsSize is None and hasattr(self.lhs, "valueSize"):
            lhsSize = self.lhs.valueSize

        rhsSize = self.rhs.size
        if rhsSize is None and hasattr(self.rhs, "valueSize"):
            rhsSize = self.rhs.valueSize

        if lhsSize is not None and rhsSize is not None and lhsSize != rhsSize:
            CompileCtx.Current().Warning(
                "Comparing operands of different size: "
                f"{lhsSize}'{self.lhs} <=> {rhsSize}'{self.rhs}")

        return True


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.lhs
        yield self.rhs


    def Render(self, ctx: RenderCtx):
        self.lhs.RenderNested(ctx)
        ctx.Write(f" {self.op} ")
        self.rhs.RenderNested(ctx)


class UnaryOperator(Expression):
    op: str
    arg: Expression


    def __init__(self, op: str, arg: RawExpression, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Unary({op})"
        self.op = op
        self.arg = Expression._FromRaw(arg, frameDepth + 1)
        self.size = self.arg.size


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.arg


    def Render(self, ctx: RenderCtx):
        ctx.Write(self.op)
        self.arg.RenderNested(ctx)


class ReductionOperator(UnaryOperator):
    def __init__(self, op: str, arg: RawExpression, frameDepth: int):
        if isinstance(arg, ReductionOperator):
            raise ParseException("Reduction operator applied on another reduction, probably a bug")
        super().__init__(op, arg, frameDepth + 1)
        self.strValue = f"Reduce({op})"
        self.size = 1
        if self.arg.size == 1:
            CompileCtx.Current().Warning(f"Reduction operator applied to 1 bit argument {arg}",
                                         self.srcFrame)


class ReplicationOperator(Expression):
    arg: Expression
    count: int

    def __init__(self, arg: Expression, count: int, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Replicate({count})"
        self.arg = Expression._CheckType(arg)
        self.count = count
        if self.arg.size is None:
            raise ParseException(f"Replication operand should have size bound: {arg}")
        self.size = self.arg.size * count


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.arg


    def Render(self, ctx: RenderCtx):
        ctx.Write("{")
        ctx.Write(str(self.count))
        ctx.Write("{")
        self.arg.Render(ctx)
        ctx.Write("}}")


class ConditionalExpr(Expression):
    condition: Expression
    ifCase: Expression
    elseCase: Expression
    needParentheses = True


    def __init__(self, condition: Expression, ifCase: RawExpression, elseCase: RawExpression,
                 frameDepth: int):
        super().__init__(frameDepth + 1)
        self.condition = Expression._CheckType(condition)
        self.ifCase = Expression._FromRaw(ifCase, frameDepth + 1)
        self.elseCase = Expression._FromRaw(elseCase, frameDepth + 1)
        if self.ifCase.size is not None and self.elseCase.size is not None:
            self.size = self.ifCase.size if self.ifCase.size >= self.elseCase.size else self.elseCase.size


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.condition
        yield self.ifCase
        yield self.elseCase


    def Render(self, ctx: RenderCtx):
        self.condition.RenderNested(ctx)
        ctx.Write(" ? ")
        self.ifCase.RenderNested(ctx)
        ctx.Write(" : ")
        self.elseCase.RenderNested(ctx)


class StatementScope(Enum):
    NON_PROCEDURAL = 1
    PROCEDURAL = 2
    ANY = NON_PROCEDURAL | PROCEDURAL


class Statement(SyntaxNode):
    allowedScope: StatementScope = StatementScope.ANY


    def __init__(self, frameDepth: int, deferPush: bool = False):
        super().__init__(frameDepth + 1)
        if not deferPush:
            ctx = CompileCtx.Current()
            if (self.allowedScope.value & StatementScope.NON_PROCEDURAL.value) == 0 and \
                not ctx.isProceduralBlock:
                raise ParseException("Statement not allowed in procedural block")
            if (self.allowedScope.value & StatementScope.PROCEDURAL.value) == 0 and \
                ctx.isProceduralBlock:
                raise ParseException("Statement not allowed outside a procedural block")
            ctx.PushStatement(self)


class Block(SyntaxNode):
    _statements: List[Statement]


    def __init__(self, frameDepth: int):
        super().__init__(frameDepth + 1)
        self._statements = list()


    def __len__(self):
        return len(self._statements)


    def PushStatement(self, stmt: Statement):
        self._statements.append(stmt)


    @property
    def lastStatement(self) -> Statement:
        if len(self._statements) == 0:
            raise Exception("Expected non-empty statements list")
        return self._statements[-1]


    def Render(self, ctx: RenderCtx):
        for stmt in self._statements:
            if ctx.options.sourceMap:
                ctx.WriteIndent(stmt.indent)
                ctx.Write(f"// {stmt.fullLocation}\n")
            ctx.WriteIndent(stmt.indent)
            stmt.Render(ctx)
            ctx.Write("\n")


class AssignmentStatement(Statement):
    lhs: Expression
    rhs: Expression
    isBlocking: bool
    isProceduralBlock: bool
    isInitialBlock: bool


    def __init__(self, lhs: Expression, rhs: RawExpression, *, isBlocking: bool, frameDepth: int):
        super().__init__(frameDepth + 1)
        rhs = Expression._FromRaw(rhs, frameDepth + 1)
        self.lhs = Expression._CheckType(lhs)
        self.rhs = rhs
        self.isBlocking = isBlocking
        ctx = CompileCtx.Current()
        self.isProceduralBlock = ctx.isProceduralBlock
        self.isInitialBlock= ctx.isInitialBlock
        lhs._Wire(True, frameDepth + 1)
        rhs._Wire(False, frameDepth + 1)
        lhs._Assign(None, frameDepth + 1)

        if self.isProceduralBlock and not self.isBlocking:
            for e in self.lhs._GetLeafNodes():
                if isinstance(e, Net) and not e.isReg:
                    raise ParseException(f"Procedural assignment to wire {e}")

        assert lhs.size is not None
        if rhs.size is not None:
            if rhs.size > lhs.size:
                raise ParseException(f"Assignment size exceeded: {lhs.size} bits <<= {rhs.size} bits")
            elif rhs.size < lhs.size:
                ctx.Warning(f"Assignment of insufficient size: {lhs.size} bits <<= {rhs.size} bits",
                            self.srcFrame)
        if hasattr(rhs, "valueSize"):
            if rhs.valueSize > lhs.size:
                raise ParseException(f"Constant minimal size exceeds assignment target size: {lhs.size} bits <<= {rhs.valueSize} bits")


    def Render(self, ctx: RenderCtx):
        if self.isProceduralBlock:
            op = "=" if self.isBlocking or self.isInitialBlock else "<="
            self.lhs.Render(ctx)
            ctx.Write(f" {op} ")
            self.rhs.Render(ctx)
        else:
            ctx.Write("assign ")
            self.lhs.Render(ctx)
            ctx.Write(" = ")
            self.rhs.Render(ctx)
        ctx.Write(";")


class IfContext:
    stmt: "IfStatement"
    # None for else clause
    condition: Optional[Expression] = None
    body: Block


    def __init__(self, stmt: "IfStatement", condition: Optional[Expression]):
        self.stmt = stmt
        if condition is not None:
            self.condition = Expression._CheckType(condition)


    def __enter__(self):
        if self.condition is not None:
            self.condition._Wire(False, 1)
        self.body = Block(1)
        CompileCtx.Current().PushBlock(self.body)


    def __exit__(self, excType, excValue, tb):
        if CompileCtx.Current().PopBlock() is not self.body:
            raise Exception("Unexpected block in stack")
        if self.condition is not None:
            self.stmt.conditions.append(self.condition)
            self.stmt.blocks.append(self.body)
        else:
            if self.stmt.elseBlock is not None:
                raise ParseException("More than one `_else` clause specified")
            self.stmt.elseBlock = self.body


class IfStatement(Statement):
    allowedScope = StatementScope.PROCEDURAL
    conditions: List[Expression]
    blocks: List[Block]
    elseBlock: Optional[Block] = None


    def __init__(self, frameDepth):
        super().__init__(frameDepth + 1)
        compileCtx = CompileCtx.Current()
        if not compileCtx.isProceduralBlock:
            raise ParseException("`if` statement can only be used in a procedural block")
        self.conditions = list()
        self.blocks = list()


    def _GetContext(self, condition: Optional[Expression]) -> IfContext:
        return IfContext(self, condition)


    def Render(self, ctx: RenderCtx):
        assert len(self.conditions) > 0
        assert len(self.conditions) == len(self.blocks)
        isFirst = True
        for condition, block in zip(self.conditions, self.blocks):
            if isFirst:
                isFirst = False
                ctx.Write("if")
            else:
                ctx.WriteIndent(self.indent)
                ctx.Write("end else if")
            ctx.Write(" (")
            condition.Render(ctx)
            ctx.Write(") begin\n")
            block.Render(ctx)

        if self.elseBlock is not None:
            ctx.WriteIndent(self.indent)
            ctx.Write("end else begin\n")
            self.elseBlock.Render(ctx)

        ctx.WriteIndent(self.indent)
        ctx.Write("end")


class CaseContext:
    stmt: "WhenStatement"
    # None for default case
    condition: Optional[Expression] = None
    body: Block


    def __init__(self, stmt: "WhenStatement", condition: Optional[Expression]):
        self.stmt = stmt
        if condition is not None:
            self.condition = Expression._CheckType(condition)


    def __enter__(self):
        if self.condition is not None:
            self.condition._Wire(False, 1)
        self.body = Block(1)
        CompileCtx.Current().PushBlock(self.body)


    def __exit__(self, excType, excValue, tb):
        if CompileCtx.Current().PopBlock() is not self.body:
            raise Exception("Unexpected block in stack")
        if self.condition is not None:
            self.stmt.conditions.append(self.condition)
            self.stmt.blocks.append(self.body)
        else:
            if self.stmt.defaultBlock is not None:
                raise ParseException("More than one `_default` case specified for `_when` statement")
            self.stmt.defaultBlock = self.body


class WhenStatement(Statement):
    allowedScope = StatementScope.PROCEDURAL
    switch: Expression
    conditions: List[Expression]
    blocks: List[Block]
    defaultBlock: Optional[Block] = None
    # Catches any statements inside `when` body. There should be nothing in normal case.
    dummyBlock: Block


    def __init__(self, switch: Expression, frameDepth):
        super().__init__(frameDepth + 1)
        compileCtx = CompileCtx.Current()
        if not compileCtx.isProceduralBlock:
            raise ParseException("`when` statement can only be used in a procedural block")
        self.switch = Expression._CheckType(switch)
        self.conditions = list()
        self.blocks = list()


    def _GetContext(self, condition: Optional[Expression]) -> CaseContext:
        return CaseContext(self, condition)


    def __enter__(self):
        self.switch._Wire(False, 1)
        self.dummyBlock = Block(1)
        CompileCtx.Current().PushBlock(self.dummyBlock)


    def __exit__(self, excType, excValue, tb):
        ctx = CompileCtx.Current()
        if ctx.PopBlock() is not self.dummyBlock:
            raise Exception("Unexpected current block")
        if len(self.dummyBlock) > 0:
            raise ParseException(
                "No synthesizable code other than `_case` and `_default` blocks allowed in "
                f"`_when` statement, has {self.dummyBlock._statements[0]}")
        if len(self.conditions) == 0:
            ctx.Warning("No cases in `_when` statement")
        self._CheckSize()


    def Render(self, ctx: RenderCtx):
        assert len(self.conditions) == len(self.blocks)
        ctx.Write("case (")
        self.switch.Render(ctx)
        ctx.Write(")\n")

        for condition, block in zip(self.conditions, self.blocks):
            ctx.WriteIndent(self.indent + 1)
            condition.Render(ctx)
            ctx.Write(": begin\n")
            block.Render(ctx)
            ctx.WriteIndent(self.indent + 1)
            ctx.Write("end\n")

        if self.defaultBlock is not None:
            ctx.WriteIndent(self.indent + 1)
            ctx.Write("default: begin\n")
            self.defaultBlock.Render(ctx)
            ctx.WriteIndent(self.indent + 1)
            ctx.Write("end\n")

        ctx.WriteIndent(self.indent)
        ctx.Write("endcase")


    def _CheckSize(self):
        size = self.switch.size
        if size is None:
            return
        ctx = CompileCtx.Current()
        for e in self.conditions:
            if e.size is not None and e.size != size:
                ctx.Warning(f"'`_when` expression size mismatch: {size} != {e.size} ({e})")


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
    allowedScope = StatementScope.NON_PROCEDURAL
    sensitivityList: Optional[SensitivityList]
    body: Block


    def __init__(self, sensitivityList: Optional[SensitivityList], frameDepth: int):
        super().__init__(frameDepth + 1, deferPush=True)
        self.sensitivityList = sensitivityList


    def __enter__(self):
        self.body = Block(1)
        ctx = CompileCtx.Current()
        if ctx.isProceduralBlock:
            raise ParseException("Nested procedural block")
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
        ctx.Write("end")


class InitialBlock(Statement):
    allowedScope = StatementScope.NON_PROCEDURAL
    body: Block


    def __init__(self, frameDepth: int):
        super().__init__(frameDepth + 1, deferPush=True)


    def __enter__(self):
        self.body = Block(1)
        ctx = CompileCtx.Current()
        if ctx.isProceduralBlock:
            raise ParseException("Nested procedural block")
        ctx.PushBlock(self.body)
        ctx.isProceduralBlock = True
        ctx.isInitialBlock = True


    def __exit__(self, excType, excValue, tb):
        ctx = CompileCtx.Current()
        if ctx.PopBlock() is not self.body:
            raise Exception("Unexpected current block")
        ctx.isProceduralBlock = False
        ctx.isInitialBlock = False
        if len(self.body) == 0:
            ctx.Warning(f"Empty initial block", self.srcFrame)
        else:
            ctx.PushStatement(self)


    def Render(self, ctx: RenderCtx):
        ctx.Write("initial begin\n")
        self.body.Render(ctx)
        ctx.Write("end")


class ModuleParameter(SyntaxNode):
    name: str


    def __init__(self, name: str, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.name = name


class Module(SyntaxNode):
    name: str
    ports: dict[str, NetProxy]
    params: dict[str, ModuleParameter]


    def __init__(self, name: str, ports: dict[str, NetProxy], params: dict[str, ModuleParameter],
                 frameDepth: int):
        super().__init__(frameDepth + 1)
        self.name = name
        self.strValue = f"Module(`{name})`"
        self.ports = ports
        self.params = params
        if len(ports) == 0:
            raise ParseException("No ports specified for a module declaration")
        for port in ports.values():
            if port.src.isWired:
                raise ParseException(
                    "Module declaration ports should not be used in any synthesizable code, "
                    f"port {port} has been wired at {SyntaxNode.GetFullLocation(port.src.wiringFrame)}")
            port.nonWireableReason = f"Used as module declaration port at {SyntaxNode.GetFullLocation(self.srcFrame)}"
            port.src.nonWireableReason = port.nonWireableReason
        CompileCtx.Current().RegisterModule(self)


    def __call__(self, **bindings: Any) -> "ModuleInstance":
        return ModuleInstance(self, bindings, 1)


class ModuleInstance(Statement):
    allowedScope = StatementScope.NON_PROCEDURAL
    module: Module
    portBindings: dict[str, Expression]
    paramBindings: dict[str, Any]
    name: str


    def __init__(self, module: Module, bindings: dict[str, Any], frameDepth):
        super().__init__(frameDepth + 1)
        self.module = module
        self.portBindings = dict()
        self.paramBindings = dict()
        for name, e in bindings.items():
            if name in module.params:
                self.paramBindings[name] = e
                continue

            e = Expression._FromRaw(e, frameDepth + 1)
            if name not in module.ports:
                raise ParseException(f"No such port in module: `{name}`")
            port = module.ports[name]
            e._Wire(port.isOutput, frameDepth + 1)
            if e.size is not None and port.size != e.size:
                CompileCtx.Current().Warning(
                    f"Port `{port}` binding size mismatch: {port.size} != {e.size} ({e})")
            self.portBindings[name] = e

        for name, port in module.ports.items():
            if not port.isOutput and name not in bindings:
                raise ParseException(f"Module input port not bound: `{name}`")

        self.name = CompileCtx.Current().GenerateModuleInstanceName(module.name, self.namespacePrefix)


    def Render(self, ctx: RenderCtx):
        ctx.Write(f"{self.module.name} ")

        if len(self.paramBindings) > 0:
            ctx.Write("#(\n")
            for index, (name, e) in enumerate(self.paramBindings.items()):
                isLast = index == len(self.paramBindings) - 1
                ctx.WriteIndent(self.indent + 1)
                if isinstance(e, str):
                    e = f"\"{e}\""
                ctx.Write(f".{name}({e})")
                if isLast:
                    ctx.Write(")\n")
                else:
                    ctx.Write(",\n")
            ctx.WriteIndent(self.indent + 1)

        ctx.Write(f"{self.namespacePrefix}{self.name}(\n")

        for index, (name, e) in enumerate(self.portBindings.items()):
            isLast = index == len(self.portBindings) - 1
            ctx.WriteIndent(self.indent + 1)
            ctx.Write(f".{name}(")
            e.Render(ctx)
            if isLast:
                ctx.Write("));")
            else:
                ctx.Write("),\n")


class Namespace(SyntaxNode):
    name: str

    def __init__(self, name: str, frameDepth):
        super().__init__(frameDepth + 1)
        self.name = name


    def __enter__(self):
        CompileCtx.Current().PushNamespace(self.name)


    def __exit__(self, excType, excValue, tb):
        if CompileCtx.Current().PopNamespace() != self.name:
            raise Exception("Unexpected current namespace")

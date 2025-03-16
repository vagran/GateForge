from dataclasses import dataclass
from enum import Enum, auto
from io import TextIOBase
import threading
from typing import Any, Generator, Generic, Iterable, Iterator, List, Optional, Self, Sequence, \
    Tuple, Type, TypeVar
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
    # Add "`default_nettype none" in prologue
    prohibitUndeclaredNets: bool = True
    # use `always_ff` for sequential `always` blocks (edge triggered) and `always_comb` for
    # combinational ones (with (possibly empty) sensitivity list).
    svProceduralBlocks: bool = False


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
    proceduralBlock: Optional["ProceduralBlock"] = None
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


    @property
    def isProceduralBlock(self) -> bool:
        return self.proceduralBlock is not None or self.isInitialBlock


    def _Open(self, frameDepth: int):
        self._blockStack.append(Block(frameDepth=frameDepth + 1))


    def Finish(self):
        pass


    def Render(self, output: TextIOBase, renderOptions: RenderOptions):
        if len(self._blockStack) != 1:
            raise Exception(f"Unexpected block stack size: {len(self._blockStack)}")

        ctx = RenderCtx(renderOptions)
        ctx.output = output

        if renderOptions.prohibitUndeclaredNets:
            ctx.Write("`default_nettype none\n")

        ctx.Write("""
`define STRINGIFY(x) `"x`"
`define ASSERT(__condition) \\
    if (!(__condition)) begin \\
        $fatal(1, "Assertion failed: %s", `STRINGIFY(__condition)); \\
    end\n
""")

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
    options: RenderOptions
    # Render declaration instead of expression when True
    renderDecl: bool = False

    output: TextIOBase


    def __init__(self, options: Optional[RenderOptions] = None):
        self.options = options if options is not None else RenderOptions()


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
    indent: int = 0
    namespacePrefix: str = ""
    # Can be instantiated outside compiling context
    isVoidCtxAllowed = False
    # Instantiated outside compiling context, never wired.
    isVoidCtx = False


    def __init__(self, frameDepth: int):
        ctx = CompileCtx._GetCurrent()
        if ctx is None:
            if not self.isVoidCtxAllowed:
                raise Exception(
                    "Synthesizable functions are not allowed to be called outside the compiler")
            self.isVoidCtx = True
        self.srcFrame = self.GetFrame(frameDepth + 1)
        if ctx is not None:
            ctx.lastFrame = self.srcFrame
            self.indent = ctx.indent
            self.namespacePrefix = ctx.namespacePrefix


    @staticmethod
    def GetLocation(frame: traceback.FrameSummary) -> str:
        return f"{Path(frame.filename).name}:{frame.lineno}"


    @staticmethod
    def GetFullLocation(frame: traceback.FrameSummary) -> str:
        return f"{frame.filename}:{frame.lineno}"


    @staticmethod
    def GetSourceMapEntry(frame: traceback.FrameSummary) -> str:
        return f"file://{frame.filename}#{frame.lineno}"


    @property
    def location(self) -> str:
        return SyntaxNode.GetLocation(self.srcFrame)


    @property
    def fullLocation(self) -> str:
        return SyntaxNode.GetFullLocation(self.srcFrame)


    @property
    def sourceMapEntry(self) -> str:
        return SyntaxNode.GetSourceMapEntry(self.srcFrame)


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


# Immutable net dimensions descriptor
class Dimensions:
    # Packed dimensions, left to right, each element is (baseIndex, size). Size may be negative to
    # indicate little endianness.
    packed: Optional[Tuple[Tuple[int, int],...]] = None
    # Unpacked dimensions, left to right, each element is (baseIndex, size)
    unpacked: Optional[Tuple[Tuple[int, int],...]] = None

    # Size of bits vector (product of packed dimensions sizes)
    vectorSize: int = 1


    def __init__(self, packedDims: Optional[Tuple[Tuple[int, int],...]],
                 unpackedDims: Optional[Tuple[Tuple[int, int],...]]):
        if packedDims is not None:
            Dimensions._ValidateDims(packedDims)
            self.packed = packedDims
            self.vectorSize = Dimensions._CalculateSize(packedDims)
        if unpackedDims is not None:
            Dimensions._ValidateDims(unpackedDims)
            self.unpacked = unpackedDims


    @property
    def isArray(self) -> bool:
        return self.unpacked is not None


    @property
    def baseIndex(self) -> int:
        """Outermost dimension base index.
        """
        return self._GetOutermostDimension()[0]


    @staticmethod
    def Parse(packedDims: Optional[Sequence[int | Sequence[int]] | int],
              unpackedDims: Optional[Sequence[int | Sequence[int]] | int]) -> "Dimensions":
        return Dimensions(Dimensions._Parse(packedDims), Dimensions._Parse(unpackedDims))


    @staticmethod
    def ParseSlice(index: "int | bool | slice | Expression",
                   ctxDim: Optional["Dimensions"] = None) -> "int | Tuple[int, int] | Expression":
        """Parse slice argument.

        :return: Either slicing expression or (baseIndex, size) tuple.
        """

        dim = ctxDim._GetOutermostDimension() if ctxDim is not None else None

        def CheckConst(index: Any) -> Optional[int]:
            if isinstance(index, int):
                return index
            elif isinstance(index, bool):
                return 1 if index else 0
            elif isinstance(index, Const):
                return index.value
            return None

        const = CheckConst(index)
        if const is not None:
            return const

        elif isinstance(index, slice):

            if index.step is not None:
                raise ParseException(f"Slice cannot have step specified: {index}")

            msb: int | None
            if index.start is None and dim is not None:
                msb = Dimensions.Msb(dim)
            else:
                msb = CheckConst(index.start)
            if msb is None:
                raise ParseException(f"Slice MSB is not constant number: {index.start}")

            lsb: int | None
            if index.stop is None and dim is not None:
                lsb = dim[0]
            else:
                lsb = CheckConst(index.stop)
            if lsb is None:
                raise ParseException(f"Slice LSB is not constant number: {index.stop}")

            if msb >= lsb:
                return (lsb, msb - lsb + 1)
            else:
                return (lsb, msb - lsb - 1)

        elif isinstance(index, Expression):
            return index

        raise ParseException(f"Bad index type: {type(index).__name__}")


    @staticmethod
    def Vector(size: int) -> "Dimensions":
        return Dimensions(((0, size),), None)


    def Slice(self, index: "int | bool | slice | Expression") -> Optional["Dimensions"]:
        """Perform slicing of the dimensioned expression. Validate if possible (if constant
        expression supplied).
        :return: Reduced dimensions.
        """

        slice = Dimensions.ParseSlice(index, self)

        if isinstance(slice, tuple):
            dim = self._GetOutermostDimension()
            if (slice[1] < 0) != (dim[1] < 0):
                raise ParseException(
                    f"Slice endianness does not match array dimension endianness: {index}")
            self._CheckIndex(slice[0])
            if slice[1] < 0:
                self._CheckIndex(slice[0] + slice[1] + 1)
            else:
                self._CheckIndex(slice[0] + slice[1] - 1)
            # At least Verilator make slice of little-endian array be big-endian. So make same logic
            # for now.
            return self._Reduced((0, abs(slice[1])))

        if isinstance(slice, int):
            self._CheckIndex(slice)

        return self._Reduced()


    def Array(self, dims: Tuple[Tuple[int, int],...]) -> "Dimensions":
        """Add unpacked dimensions.
        :param dims: Additional (rightmost) unpacked dimensions, left to right.
        """
        return Dimensions(self.packed, self.unpacked + dims if self.unpacked is not None else dims)


    @staticmethod
    def MakeArray(src: Optional["Dimensions"], dims: Sequence[int | Sequence[int]] | int) -> "Dimensions":
        _dims = Dimensions._Parse(dims)
        if _dims is None:
            raise ParseException("Empty dimension is not allowed for array declaration")
        if src is None:
            return Dimensions(None, _dims)
        return src.Array(_dims)


    def RenderDeclaration(self, ctx: RenderCtx, name: str):
        if self.packed is not None:
            for dim in self.packed:
                ctx.Write(self.StrDimension(dim))
        ctx.Write(" ")
        ctx.Write(name)
        if self.unpacked is not None:
            for dim in self.unpacked:
                ctx.Write(self.StrDimension(dim))


    def Match(self, other: "Dimensions", unpackedOnly: bool = False) -> bool:
        """Match two dimensions. Unpacked part should match exactly. Packed vector size should be
        equal.
        """
        if self.unpacked != other.unpacked:
            return False
        if unpackedOnly:
            return True
        return self.vectorSize == other.vectorSize


    @staticmethod
    def MatchAny(d1: Optional["Dimensions"], d2: Optional["Dimensions"],
                 unpackedOnly: bool = False) -> bool:
        if d1 is not None and d2 is not None:
            return d1.Match(d2, unpackedOnly)
        if d1 is not None and d1.isArray:
            return False
        if d2 is not None and d2.isArray:
            return False
        if unpackedOnly:
            return True
        if d1 is not None and d1.vectorSize != 1:
            return False
        if d2 is not None and d2.vectorSize != 1:
            return False
        return True


    def GetVectorBitIndex(self, index: int) -> Tuple[int,...]:
        """_summary_

        :param index: Zero-based linear index of vector bit.
        :return: Multidimensional index of vector bit.
        """
        assert self.packed is not None
        result: List[int] = list()
        for dim in reversed(self.packed):
            size = abs(dim[1])
            idx = index % size
            index //= size
            result.append(Dimensions.GetIndex(dim, idx))
        if index != 0:
            raise Exception(f"Index out of range: {index}")
        return tuple(reversed(result))


    def __len__(self) -> int:
        dim = self._GetOutermostDimension()
        return abs(dim[1])


    def __eq__(self, other: "Dimensions | int | Sequence[int]") -> bool: # type: ignore
        if isinstance(other, Dimensions):
            return self.unpacked == other.unpacked and self.packed == other.packed
        _other = Dimensions.Parse(other, None)
        return self.unpacked == _other.unpacked and self.packed == _other.packed


    def Str(self, name: Optional[str] = None) -> str:
        s = ""
        if self.packed is not None:
            for dim in self.packed:
                s += f"[{abs(dim[1])} as {Dimensions.StrDimension(dim)}]"
        if len(s) > 0:
            s += " "
        s += "$" if name is None else name
        if self.unpacked is not None:
            for dim in self.unpacked:
                s += f"[{abs(dim[1])} as {Dimensions.StrDimension(dim)}]"
        return s


    def __str__(self) -> str:
        return self.Str()


    @staticmethod
    def StrAny(dim: Optional["Dimensions"], name: Optional[str] = None) -> str:
        if dim is None:
            return ""
        return dim.Str(name)


    @staticmethod
    def Msb(dim: Tuple[int, int]) -> int:
        baseIndex, size = dim
        if size > 0:
            return baseIndex + size - 1
        return baseIndex + size + 1


    @staticmethod
    def StrDimension(dim: Tuple[int, int], brackets: bool = True) -> str:
        baseIndex = dim[0]
        msb = Dimensions.Msb(dim)
        if brackets:
            return f"[{msb}:{baseIndex}]"
        return f"{msb}:{baseIndex}"


    @staticmethod
    def EnumerateIndices(dim: Tuple[int, int] | int) -> Generator[int]:
        if isinstance(dim, int):
            yield dim
            return
        baseIndex, size = dim
        if size < 0:
            for i in range(baseIndex, baseIndex + size, -1):
                yield i
        else:
            for i in range(baseIndex, baseIndex + size):
                yield i


    @staticmethod
    def GetIndex(dim: Tuple[int, int], index: int) -> int:
        """
        :param dim: Dimension range.
        :param index: Zero-based index in the range.
        :returns: Index from the range.
        """
        if index >= abs(dim[1]):
            raise Exception(f"Index out of range: {index} for {Dimensions.StrDimension(dim)}")
        if dim[1] < 0:
            return dim[0] - index
        return dim[0] + index


    @staticmethod
    def StrIndex(index: Tuple[int,...]) -> str:
        s = ""
        for i in index:
            s += f"[{i}]"
        return s


    def _CheckIndex(self, index: int):
        dim = self._GetOutermostDimension()
        baseIndex, size = dim
        if size > 0:
            if index >= baseIndex and index < baseIndex + size:
                return
        else:
            if index <= baseIndex and index > baseIndex + size:
                return
        raise ParseException(f"Index out of range: {index} of {Dimensions.StrDimension(dim)}")


    def _Reduced(self, newDim: Optional[Tuple[int, int]] = None) -> Optional["Dimensions"]:
        """
        :param newDim: Replace the outermost dimension instead of stripping if specified.
        :return: Object with one reduced dimension (the outermost).
        """
        if self.unpacked is not None:
            if newDim is not None:
                return Dimensions(self.packed, (newDim, *self.unpacked[1:]))
            if len(self.unpacked) == 1:
                return Dimensions(self.packed, None)
            return Dimensions(self.packed, self.unpacked[1:])
        assert self.packed is not None
        if newDim is not None:
            return Dimensions((newDim, *self.packed[1:]), None)
        if len(self.packed) == 1:
            return None
        return Dimensions(self.packed[1:], None)


    def _GetOutermostDimension(self) -> Tuple[int, int]:
        if self.unpacked is not None:
            return self.unpacked[0]
        assert self.packed is not None
        return self.packed[0]


    @staticmethod
    def _Parse(dims: Optional[Sequence[int | Sequence[int]] | int]) -> Optional[Tuple[Tuple[int, int],...]]:
        if isinstance(dims, int):
            if dims <= 0:
                raise ParseException(f"Dimension size should be positive: {dims}")
            return ((0, dims),)

        if dims is None or len(dims) == 0:
            return None
        result: List[Tuple[int, int]] = []
        for dim in dims:
            if isinstance(dim, int):
                if dim <= 0:
                    raise ParseException(f"Dimension size should be positive: {dim}")
                result.append((0, dim))
                continue
            if len(dim) != 2:
                raise ParseException(f"Expected two elements in dimension item: {dim}")
            leftIndex, rightIndex = dim
            if not isinstance(leftIndex, int):
                raise ParseException(f"Expected integer index: {leftIndex}")
            if not isinstance(rightIndex, int):
                raise ParseException(f"Expected integer index: {rightIndex}")
            if leftIndex >= rightIndex:
                result.append((rightIndex, leftIndex - rightIndex + 1))
            else:
                # Little-endian
                result.append((rightIndex, leftIndex - rightIndex - 1))

        return tuple(result)


    @staticmethod
    def _ValidateDims(dims: Tuple[Tuple[int, int],...]):
        for dim in dims:
            if len(dim) != 2:
                raise ParseException("Expected two elements in dimension item")
            _, size = dim
            if size == 0:
                raise ParseException("Array zero size")


    @staticmethod
    def _CalculateSize(dims: Tuple[Tuple[int, int],...]):
        size = 1
        for _, dSize in dims:
            size *= abs(dSize)
        return size


# Holds information from annotated net members
class TypeTag:
    cls: Type
    dims: Optional[Dimensions] = None


    def __init__(self, cls: Type, dims: Optional[Dimensions]):
        self.cls = cls
        if dims is not None:
            self.dims = dims


    @staticmethod
    def Create(cls: Type, dims: Sequence[int | Sequence[int]] | int) -> "TypeTag":
        return TypeTag(cls, Dimensions.Parse(dims, None))


    def array(self, *size: Sequence[int] | int):
        return TypeTag(self.cls, Dimensions.MakeArray(self.dims, size))


    @staticmethod
    def CheckAnnotation(annotation: Any, allowedTypes: Sequence[Type]) -> "Optional[TypeTag]":
        if isinstance(annotation, TypeTag):
            return annotation
        if annotation in allowedTypes:
            return TypeTag(annotation, None)
        return None


class Expression(SyntaxNode):
    dims: Optional[Dimensions] = None
    # Single-dimensional vector can be of unbound size, e.g. unbound size constant or its
    # concatenation
    isUnboundSize: bool = False
    isLhs: bool = False
    # Expression is wired when used in some statement. Unwired expressions are dangling and do not
    # affect the produced output, however, may be used, for example, to define external module ports.
    isWired: bool = False
    wiringFrame: traceback.FrameSummary
    nonWireableReason: Optional[str] = None
    needParentheses: bool = False


    @property
    def vectorSize(self) -> int:
        if self.dims is None:
            return 1
        return self.dims.vectorSize


    @property
    def baseIndex(self) -> int:
        if self.dims is None:
            return 0
        return self.dims.baseIndex


    @property
    def isArray(self) -> bool:
        if self.dims is None:
            return False
        return self.dims.isArray


    # For allowing syntax `myNet: Expression[32]`, or `Net[[11, 8]]`
    def __class_getitem__(cls, size: Sequence[int | Sequence[int]] | int) -> TypeTag:
        if not isinstance(size, tuple):
            size = (size, ) # type: ignore
        return TypeTag.Create(cls, size)


    def __getitem__(self, index: "int | bool | slice | Expression") -> "SliceExpr":
        if self.dims is None:
            raise ParseException(f"Attempting to slice dimensionless expression: {self}")
        return SliceExpr(self, index, 1)


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
        if self.isVoidCtx:
            return True
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


    def _Assign(self, bitIndex: Optional[Tuple[int, ...]], frameDepth: int):
        """Called to mark assignment to the expression.

        :param bitIndex: Bit index which is driven, tuple of indices of each dimension, left to
            right. None if whole expression affected.
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


    def RenderNested(self, ctx: RenderCtx, useCurly: bool = False):
        if self.needParentheses:
            ctx.Write("{" if useCurly else "(")
        self.Render(ctx)
        if self.needParentheses:
            ctx.Write("}" if useCurly else ")")


    def __bool__(self):
        raise ParseException(
            "Use of synthesizable expression in boolean context. "
            "Use bitwise operators and dedicated statements. "
            "Use parenthesis when mixing bitwise and comparison operators.")


    def __len__(self) -> int:
        if self.dims is None:
            return 1
        return len(self.dims)


    def __ilshift__(self, rhs: "RawExpression") -> Self:
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=1)
        return self


    def __ifloordiv__(self, rhs: "RawExpression") -> Self:
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=1)
        return self


    def assign(self, rhs: "RawExpression", *, frameDepth=0):
        AssignmentStatement(self, rhs, isBlocking=False, frameDepth=frameDepth + 1)


    def bassign(self, rhs: "RawExpression", *, frameDepth=0):
        AssignmentStatement(self, rhs, isBlocking=True, frameDepth=frameDepth + 1)


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


    def __iadd__(self, rhs: "RawExpression") -> "Expression":
        AssignmentStatement(self, ArithmeticExpr("+", (self, rhs), 1),
                            isBlocking=False, frameDepth=1)
        return self


    def __isub__(self, rhs: "RawExpression") -> "Expression":
        AssignmentStatement(self, ArithmeticExpr("-", (self, rhs), 1),
                            isBlocking=False, frameDepth=1)
        return self


    def __iand__(self, rhs: "RawExpression") -> "Expression":
        AssignmentStatement(self, ArithmeticExpr("&", (self, rhs), 1),
                            isBlocking=False, frameDepth=1)
        return self


    def __ior__(self, rhs: "RawExpression") -> "Expression": # type: ignore
        AssignmentStatement(self, ArithmeticExpr("|", (self, rhs), 1),
                            isBlocking=False, frameDepth=1)
        return self


    def __ixor__(self, rhs: "RawExpression") -> "Expression":
        AssignmentStatement(self, ArithmeticExpr("^", (self, rhs), 1),
                            isBlocking=False, frameDepth=1)
        return self


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


    def sll(self, rhs: "RawExpression") -> "ShiftExpr":
        return ShiftExpr("<<", self, rhs, 1)


    def srl(self, rhs: "RawExpression") -> "ShiftExpr":
        return ShiftExpr(">>", self, rhs, 1)


    def sra(self, rhs: "RawExpression") -> "ShiftExpr":
        return ShiftExpr(">>>", self, rhs, 1)


    @property
    def signed(self) -> "FunctionCallExpr":
        return FunctionCallExpr("$signed", [self], self.dims, 1)


RawExpression = Expression | int | bool


class Const(Expression):
    isVoidCtxAllowed = True
    dims: Dimensions
    # Minimal number of bits required to represent the value without trimming
    valueSize: int
    # Constant value
    value: int
    # 'z' bits in a value
    zMask: int = 0
    # 'x' bits in a value
    xMask: int = 0

    _valuePat = re.compile(r"(?:(-)?(\d+)?'([bdoh]))?([\da-f_zx\?]+)", re.RegexFlag.IGNORECASE)


    def __init__(self, value: str | int | bool, size: Optional[int] = None, *, frameDepth: int):
        super().__init__(frameDepth + 1)

        zMask = 0
        xMask = 0
        if isinstance(value, str):
            if size is not None:
                raise ParseException("Size should not be specified for string value")
            self.value, zMask, xMask, size = Const._ParseStringValue(value)
        elif isinstance(value, bool):
            self.value = 1 if value else 0
            size = 1
        else:
            self.value = value

        self.valueSize = Const.GetMinValueBits(self.value)

        if zMask != 0:
            self.zMask = zMask
            maskSize = Const._GetMsbIndex(self.zMask) + 1
            if maskSize > self.valueSize:
                self.valueSize = maskSize

        if xMask != 0:
            self.xMask = xMask
            maskSize = Const._GetMsbIndex(self.xMask) + 1
            if maskSize > self.valueSize:
                self.valueSize = maskSize

        if size is None:
            size = self.valueSize
            self.isUnboundSize = True
        self.dims = Dimensions.Vector(size)

        if not self.isUnboundSize and self.valueSize > len(self.dims):
            raise ParseException(
                f"Constant explicit size less than value required size: "
                f"{len(self.dims)} < {self.valueSize}")

        self.strValue = f"Const({self.value})"


    def __getitem__(self, s):
        # Slicing is optimized to produce constant
        slice = Dimensions.ParseSlice(s, self.dims)
        if isinstance(slice, Expression):
            return SliceExpr(self, s, 1)
        if isinstance(slice, int):
            index = slice
            size = 1
        else:
            index, size = slice
        if index < 0:
            raise ParseException(f"Negative index for constant slice: {index}")
        if not self.isUnboundSize:
            # Validate range
            self.dims.Slice(s)
        mask = (1 << size) - 1
        return Const((self.value >> index) & mask, size, frameDepth=1)


    def Render(self, ctx: RenderCtx):
        if self.zMask == 0 and self.xMask == 0:
            ctx.Write(f"{'-' if self.value < 0 else ''}{'' if self.isUnboundSize else len(self.dims)}'h{abs(self.value):x}")
        else:
            # Render binary form
            ctx.Write(f"{'-' if self.value < 0 else ''}{'' if self.isUnboundSize else len(self.dims)}'b")
            for i in range(self.valueSize - 1, -1, -1):
                mask = 1 << i
                if self.zMask & mask != 0:
                    ctx.Write("z")
                elif self.xMask & mask != 0:
                    ctx.Write("x")
                elif self.value & mask != 0:
                    ctx.Write("1")
                else:
                    ctx.Write("0")


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
    def _GetMsbIndex(n):
        if n == 0:
            return None
        index = 0
        while n > 1:
            n >>= 1
            index += 1
        return index


    @staticmethod
    def _ParseStringValue(valueStr: str) -> Tuple[int, int, int, Optional[int]]:
        """
        Parse Verilog constant literal.
        :return: numeric value, z-mask, x-mask and optional size.
        """
        m = Const._valuePat.fullmatch(valueStr)
        if m is None:
            raise ParseException(f"Bad constant format: `{valueStr}`")
        groups = m.groups()

        size = None

        if groups[1] is not None:
            try:
                size = int(groups[1])
            except:
                raise ParseException(f"Bad size value in constant: `{groups[1]}")
            if size <= 0:
                raise ParseException(f"Bad size value in constant: `{size}`")

        base = 10
        if groups[2] is not None:
            b = groups[2].lower()
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

        zMask = 0
        xMask = 0

        def GetDigit(c):
            if c >= "0" and c <= "9":
                return ord(c) - ord("0")
            if c >= "a" and c <= "f":
                return ord(c) - ord("a") + 10
            raise ParseException(f"Bad digit in numeric literal: `{c}` in {groups[3]}")


        numDigits = 0
        value = 0
        digitBits = int(math.log2(base))
        leftmostDigit = None
        for digit in groups[3].lower():
            if digit == "_":
                continue
            if leftmostDigit is None:
                leftmostDigit = digit
            if digit != "z" and digit != "x" and digit != "?":
                digitValue = GetDigit(digit)
                if digitValue >= base:
                    raise ParseException(f"Bad digit in numeric literal: `{digit}` in {groups[3]} with base {base}")
            if base == 10:
                if digit == "z" or digit == "x" or digit == "?":
                    raise ParseException(f"'{digit}' not allowed in decimal numeric literal: {groups[3]}")
                value = value * 10 + digitValue
            else:
                zMask <<= digitBits
                xMask <<= digitBits
                value <<= digitBits
                if digit == "z" or digit == "?":
                    zMask |= (1 << digitBits) - 1
                elif digit == "x":
                    xMask |= (1 << digitBits) - 1
                else:
                    value |= digitValue
            numDigits += 1

        if numDigits == 0:
            raise ParseException(f"Bad numeric literal, no digits: {groups[3]}")

        if groups[0] is not None:
            value = -value

        if size is not None and base != 10 and (leftmostDigit == "z" or leftmostDigit == "x"):
            # Leftmost bits should be filled with `z` or `x` if explicitly specified size is greater
            # than the specified value and leftmost digit is `z` or `x`
            valueSize = numDigits * digitBits
            if valueSize < size:
                for i in range(valueSize, size):
                    mask = 1 << i
                    if leftmostDigit == "z":
                        zMask |= mask
                    else:
                        xMask |= mask

        return value, zMask, xMask, size


class AssignmentTracker:
    """Radix trie with multi-dimensional index prefixes.
    """

    class Node:
        prefix: Tuple[int, ...] = tuple()
        children: List["AssignmentTracker.Node"]
        value: Optional[traceback.FrameSummary] = None


        def _Match(self, index: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
            """Match the specified index to this node prefix.add()
            :return: Tailing part of the index if matched, `None` if not matched.
            """
            if len(index) <= len(self.prefix):
                if index != self.prefix[0:len(index)]:
                    return None
                return tuple()
            if index[0:len(self.prefix)] != self.prefix:
                return None
            return index[len(self.prefix):]


        def GetFirstValue(self):
            if self.value is not None:
                return self.value
            return self.children[0].GetFirstValue()


        def Get(self, index: Tuple[int, ...]) -> Optional[traceback.FrameSummary]:
            if self.value is not None:
                return self.value
            for child in self.children:
                newIndex = child._Match(index)
                if newIndex is None:
                    continue
                if len(newIndex) == 0:
                    return child.GetFirstValue()
                return child.Get(newIndex)

            return None


        def Set(self, index: Tuple[int, ...], value: traceback.FrameSummary):
            tail = self._Match(index)
            if tail is None:
                raise Exception("No prefix match")
            splitPos = len(index) - len(tail)
            if splitPos == len(self.prefix):
                for child in self.children:
                    if child._Match(tail) is not None:
                        child.Set(tail, value)
                        return
                node = AssignmentTracker.Node()
                node.prefix = tail
                node.value = value
                node.children = list()
                self.children.append(node)
            else:
                node = AssignmentTracker.Node()
                node.prefix = self.prefix[splitPos:]
                node.children = self.children
                self.prefix = self.prefix[0:splitPos]
                self.children = [node]


    root: Node


    def __init__(self):
        self.root = AssignmentTracker.Node()
        self.root.children = list()


    def Get(self, index: Tuple[int, ...]) -> Optional[traceback.FrameSummary]:
        return self.root.Get(index)


    def Set(self, index: Tuple[int, ...], value: traceback.FrameSummary):
        self.root.Set(index, value)


class Net(Expression):
    isLhs = True
    isReg: bool
    # Name specified when net is created
    initialName: Optional[str] = None
    # Actual name is resolved when wired
    name: str
    # Bits assignments map, for assignments in procedural blocks
    procAssignments: AssignmentTracker
    # For assignments outside of procedural blocks
    nonProcAssignments: AssignmentTracker


    def __init__(self, *, dims: Optional[Dimensions], isReg: bool, name: Optional[str],
                 frameDepth: int):
        super().__init__(frameDepth + 1)
        if dims is not None:
            self.dims = dims
        self.isReg = isReg
        self.strValue = f"{'Reg' if isReg else 'Wire'}"

        if name is not None:
            _CheckIdentifier(name)
            self.initialName = name
            self.strValue += f"(`{name}`)"

        self.procAssignments = AssignmentTracker()
        self.nonProcAssignments = AssignmentTracker()


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
                ctx.Write(f"// {self.sourceMapEntry}\n")
                ctx.WriteIndent(self.indent)
            ctx.Write("reg" if self.isReg else "wire")
            if self.dims is not None:
                self.dims.RenderDeclaration(ctx, f"{self.namespacePrefix}{name}")
            else:
                ctx.Write(f" {self.namespacePrefix}{name}")
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


    def _Assign(self, bitIndex: Optional[Tuple[int,...]], frameDepth: int):
        ctx = CompileCtx.Current()

        def _AssignBit(bitIndex: Tuple[int,...], frame: traceback.FrameSummary):
            # We cannot properly check for conflict in procedural assignments without analyzing
            # whole the flow. So just check for conflict with non-procedural continuous assignment.
            prevProcFrame = self.procAssignments.Get(bitIndex)
            prevNonProcFrame = self.nonProcAssignments.Get(bitIndex)
            if prevNonProcFrame is not None:
                prevFrame = prevNonProcFrame
            elif not ctx.isProceduralBlock and prevProcFrame is not None:
                prevFrame = prevProcFrame
            else:
                prevFrame = None
            if prevFrame is not None:
                raise ParseException(
                    f"Net re-assignment, `{self}{Dimensions.StrIndex(bitIndex)}` was previously assigned at " +
                    SyntaxNode.GetFullLocation(prevFrame))
            if ctx.isProceduralBlock:
                self.procAssignments.Set(bitIndex, frame)
            else:
                self.nonProcAssignments.Set(bitIndex, frame)

        frame = self.GetFrame(frameDepth + 1)
        if bitIndex is None:
            _AssignBit((), frame)
        else:
            _AssignBit(bitIndex, frame)


    def array(self, *dims: int | Sequence[int]) -> "Net":
        cls: Type[Reg | Wire]
        if isinstance(self, Reg):
            cls = Reg
        elif isinstance(self, Wire):
            cls = Wire
        else:
            raise ParseException("Can be called only for Reg or Wire instance")
        if self.isWired:
            raise ParseException(
                f"Cannot declare array on wired expression, wired at {SyntaxNode.GetFullLocation(self.wiringFrame)}")
        return cls(dims=Dimensions.MakeArray(self.dims, dims),
                   isReg=self.isReg, name=self.initialName, frameDepth=1)


    @property
    def input(self) -> "InputNet":
        return InputNet(self, 1)


    @property
    def output(self) -> "OutputNet":
        return OutputNet(self, 1)


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
        super().__init__(dims=src.dims, isReg=src.isReg,
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
    def input(self) -> "InputNet":
        if self.isOutput:
            return InputNet(self.src, 1)
        return self # type: ignore


    @property
    def output(self) -> "OutputNet":
        if self.isOutput:
            return self # type: ignore
        raise ParseException(f"Cannot use input net as output: {self}")


    @property
    def port(self) -> "Port":
        return Port(self, 1)

    # Allow syntax like `wire("myPort").output.port <<= someExpression`
    @port.setter
    def port(self, value: "Port"):
        if not isinstance(value, Port) or value.src is not self:
            raise ParseException("Only `<<=` or `//=` statement can be used to assign port value")


class InputOutputTypeTag(TypeTag):
    isOutput: bool

    @staticmethod
    def Create(cls, netTypeAndSize: Any) -> "InputOutputTypeTag":
        if isinstance(netTypeAndSize, tuple):
            tag = InputOutputTypeTag(netTypeAndSize[0], Dimensions.Parse(netTypeAndSize[1:], None))
        else:
            tag = InputOutputTypeTag(netTypeAndSize, None)
        tag.isOutput = cls.isOutput
        return tag


TNet = TypeVar("TNet", Wire, Reg)


class OutputNet(Generic[TNet], NetProxy):
    isOutput = True


    def __init__(self, src, frameDepth):
        super().__init__(src, True, frameDepth + 1)


    # For allowing syntax `myNet: OutputNet[Wire]`, or `OutputNet[Wire, 8]`
    def __class_getitem__(cls, netTypeAndSize: Any) -> TypeTag:
        return InputOutputTypeTag.Create(cls, netTypeAndSize)


class InputNet(Generic[TNet], NetProxy):
    isOutput = False


    def __init__(self, src, frameDepth):
        super().__init__(src, False, frameDepth + 1)


    # For allowing syntax `myNet: InputNet[Wire]`, or `InputNet[Wire, 8]`
    def __class_getitem__(cls, netTypeAndSize: Any) -> TypeTag:
        return InputOutputTypeTag.Create(cls, netTypeAndSize)


class Port(Net):
    src: NetProxy
    isOutput: bool


    def __init__(self, src: NetProxy, frameDepth: int):
        if not isinstance(src, NetProxy):
            raise ParseException(f"NetProxy type expected, has `{type(src).__name__}`")
        if isinstance(src.src, Port):
            raise ParseException(f"Cannot create port from port: {src.src}")
        if src.initialName is None:
            raise ParseException("Port cannot be created from unnamed net")
        if src.src.isWired:
            raise ParseException("Port cannot be created from wired net, wired at " +
                                 SyntaxNode.GetFullLocation(src.src.wiringFrame))
        super().__init__(dims=src.dims, isReg=src.isReg, name=src.initialName,
                         frameDepth=frameDepth + 1)
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
                ctx.Write(f"// {self.sourceMapEntry}\n")
                ctx.WriteIndent(self.indent)
            ctx.Write("output" if self.src.isOutput else "input")
            ctx.Write(" ")
            super().Render(ctx)
        else:
            super().Render(ctx)


    @property
    def input(self) -> "InputNet":
        return InputNet(self, 1)


    @property
    def output(self) -> "OutputNet":
        if not self.isOutput:
            raise ParseException(f"Cannot use input port as output: {self}")
        return OutputNet(self, 1)


class ConcatExpr(Expression):
    dims: Dimensions
    # Left to right
    args: List[Expression]


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
        isFirst = True
        isLhs = True

        for e in self.args:
            if e.isArray:
                raise ParseException(f"Unpacked array is not supported in concatenation: {e}")
            if isFirst:
                isFirst = False
                if e.isUnboundSize:
                    self.isUnboundSize = True
            else:
                if e.isUnboundSize:
                    raise ParseException(
                        "Concatenation can have expression with unbound size on left-most position"
                         f" only, unbound expression: {e}")
            size += e.vectorSize
            if not e.isLhs:
                isLhs = False

        self.dims = Dimensions.Vector(size)
        self.isLhs = isLhs


    def _GetChildren(self) -> Iterator["Expression"]:
        yield from self.args


    def _Assign(self, bitIndex: Optional[Tuple[int,...]], frameDepth: int):
        if bitIndex is None:
            for arg in self.args:
                arg._Assign(None, frameDepth + 1)
            return

        if len(bitIndex) != 1:
            raise ParseException("Expected single dimension for concatenation assignment")

        bitIdx = bitIndex[0]
        argBaseIdx = 0
        for arg in reversed(self.args):
            if bitIdx < argBaseIdx + arg.vectorSize:
                assert bitIdx >= argBaseIdx
                if arg.dims is None:
                    # Assign the only bit
                    arg._Assign((0,), frameDepth + 1)
                else:
                    arg._Assign(arg.dims.GetVectorBitIndex(bitIdx - argBaseIdx), frameDepth + 1)
                return
            argBaseIdx += arg.vectorSize
        raise Exception(f"Assignment index out of range: {bitIndex} >= {self.vectorSize}")


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
    arg: Expression
    # Either single constant index, constant range or single dynamic index
    index: int | Tuple[int, int] | "Expression"


    def __init__(self, arg: Expression, index: "int | bool | slice | Expression", frameDepth: int):
        super().__init__(frameDepth + 1)
        self.arg = Expression._CheckType(arg)
        if self.arg.dims is None:
            raise Exception("Cannot slice dimensionless expression")
        self.index = Dimensions.ParseSlice(index, self.arg.dims)
        self.dims = self.arg.dims.Slice(index)
        self.isLhs = self.arg.isLhs


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.arg
        # Do not return index expression here since we do not want to apply default wiring and
        # assignment tracking logic to it.


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        """
        Index expression should always be wired as RHS. So make this override.
        """
        if not super()._Wire(isLhs, frameDepth):
            return False
        if isinstance(self.index, Expression):
            self.index._Wire(False, frameDepth + 1)
        return True


    def _Assign(self, bitIndex: Optional[Tuple[int,...]], frameDepth: int):
        if isinstance(self.index, Expression):
            # Dynamic slicing, cannot track
            return

        if isinstance(self.index, int):
            if bitIndex is None:
                self.arg._Assign((self.index,), frameDepth + 1)
            else:
                self.arg._Assign((self.index, *bitIndex), frameDepth + 1)
            return

        # Slice range
        if bitIndex is None:
            for i in Dimensions.EnumerateIndices(self.index):
                self.arg._Assign((i,), frameDepth + 1)
            return

        assert len(bitIndex) > 0

        idx = Dimensions.GetIndex(self.index, bitIndex[0])
        self.arg._Assign((idx, *bitIndex[1:]), frameDepth + 1)


    def Render(self, ctx: RenderCtx):
        # Arithmetic expression cannot be sliced in Verilog, enclose in curly braces
        self.arg.RenderNested(ctx, True)
        ctx.Write("[")
        if isinstance(self.index, int):
            ctx.Write(str(self.index))
        elif isinstance(self.index, tuple):
            ctx.Write(Dimensions.StrDimension(self.index, False))
        else:
            self.index.Render(ctx)
        ctx.Write("]")


class ArithmeticExpr(Expression):
    op: str
    args: List[Expression]
    needParentheses = True


    def __init__(self, op: str, args: Iterable[RawExpression], frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Op({op})"
        self.op = op
        self.args = list(self._FlattenArithmeticExpr(args, frameDepth + 1))
        self._CalculateSize()


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
        size = 0
        for e in self.args:
            if e.isArray:
                raise ParseException(f"Unpacked array is not supported in arithmetic expression: {e}")
            if e.vectorSize > size:
                size = e.vectorSize
        self.dims = Dimensions.Vector(size)


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth + 1):
            return False
        ctx = CompileCtx.Current()
        for e in self.args:
            if (e.vectorSize != self.vectorSize and not e.isUnboundSize) or \
                e.vectorSize > self.vectorSize:

                ctx.Warning(
                    f"Arithmetic expression argument size mismatch: {e} size {e.vectorSize} "
                    f"bits, required {self.vectorSize} bits", self.srcFrame)
        return True


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


    def __init__(self, op: str, lhs: Expression, rhs: RawExpression, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Cmp({op})"
        self.op = op
        self.lhs = Expression._CheckType(lhs)
        self.rhs = Expression._FromRaw(rhs, frameDepth + 1)


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth + 1):
            return False

        if not Dimensions.MatchAny(self.lhs.dims, self.rhs.dims):
            CompileCtx.Current().Warning(
                "Comparing operands of different size: "
                f"{Dimensions.StrAny(self.lhs.dims, str(self.lhs))} <=> {Dimensions.StrAny(self.rhs.dims, str(self.rhs))}")

        return True


    def _GetChildren(self) -> Iterator["Expression"]:
        yield self.lhs
        yield self.rhs


    def Render(self, ctx: RenderCtx):
        self.lhs.RenderNested(ctx)
        ctx.Write(f" {self.op} ")
        self.rhs.RenderNested(ctx)


class ShiftExpr(Expression):
    op: str
    rhs: Expression
    lhs: Expression
    needParentheses = True


    def __init__(self, op: str, lhs: Expression, rhs: RawExpression, frameDepth: int):
        super().__init__(frameDepth + 1)
        self.strValue = f"Cmp({op})"
        self.op = op
        self.lhs = Expression._CheckType(lhs)
        self.rhs = Expression._FromRaw(rhs, frameDepth + 1)
        self.dims = self.lhs.dims


    def _Wire(self, isLhs: bool, frameDepth: int) -> bool:
        if not super()._Wire(isLhs, frameDepth + 1):
            return False

        if isinstance(self.rhs, Const) and self.rhs.value >= self.lhs.vectorSize:
            CompileCtx.Current().Warning(
                f"Shift amount reaches expression size: {self.rhs.value} >= {self.lhs.vectorSize}")

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
        self.dims = self.arg.dims


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
        if self.arg.isArray:
            raise ParseException("Reduction operator cannot be applied to unpacked array")
        self.dims = Dimensions.Vector(1)
        if self.arg.vectorSize == 1:
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
        if self.arg.dims is not None and self.arg.dims.isArray:
            raise ParseException("Replication operator cannot be applied to unpacked array")
        if self.arg.isUnboundSize:
            raise ParseException(f"Replication operand should have size bound: {arg}")
        self.dims = Dimensions.Vector(self.arg.vectorSize * count)


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
        if self.ifCase.isArray or self.elseCase.isArray:
            if not Dimensions.MatchAny(self.ifCase.dims, self.elseCase.dims):
                raise ParseException(
                    "Arrays of different shape in conditional expression: "
                    f"{Dimensions.StrAny(self.ifCase.dims)} <=> {Dimensions.StrAny(self.elseCase.dims)}")
            self.dims = self.ifCase.dims
        else:
            if Dimensions.MatchAny(self.ifCase.dims, self.elseCase.dims):
                self.dims = self.ifCase.dims
            else:
                # For different shapes make single dimension with max vector size.
                size = max(self.ifCase.vectorSize, self.elseCase.vectorSize)
                self.dims = Dimensions.Vector(size)


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


class FunctionCallExpr(Expression):
    funcName: str
    args: List[Expression]


    def __init__(self, funcName: str, args: Iterable[RawExpression], dims: Optional[Dimensions],
                 frameDepth: int):
        super().__init__(frameDepth + 1)
        self.funcName = funcName
        self.strValue = f"{funcName}()"
        self.args = [Expression._FromRaw(e, frameDepth + 1) for e in args]
        self.dims = dims


    def _GetChildren(self) -> Iterator["Expression"]:
        yield from self.args


    def Render(self, ctx: RenderCtx):
        ctx.Write(self.funcName)
        ctx.Write("(")
        isFirst = True
        for e in self.args:
            if isFirst:
                isFirst = False
            else:
                ctx.Write(", ")
            e.Render(ctx)
        ctx.Write(")")


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
                raise ParseException("Statement not allowed outside a procedural block")
            if (self.allowedScope.value & StatementScope.PROCEDURAL.value) == 0 and \
                ctx.isProceduralBlock:
                raise ParseException("Statement not allowed in procedural block")
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


    def Render(self, ctx: RenderCtx, indentOffset = 0):
        for stmt in self._statements:
            if ctx.options.sourceMap:
                ctx.WriteIndent(stmt.indent + indentOffset)
                ctx.Write(f"// {stmt.sourceMapEntry}\n")
            ctx.WriteIndent(stmt.indent + indentOffset)
            stmt.Render(ctx)
            ctx.Write("\n")


class AssignmentStatement(Statement):
    lhs: Expression
    rhs: Expression
    isBlocking: bool
    isProceduralBlock: bool
    isInitialBlock: bool
    isCombinationalBlock: bool


    def __init__(self, lhs: Expression, rhs: RawExpression, *, isBlocking: bool, frameDepth: int):
        super().__init__(frameDepth + 1)
        rhs = Expression._FromRaw(rhs, frameDepth + 1)
        self.lhs = Expression._CheckType(lhs)
        self.rhs = rhs
        self.isBlocking = isBlocking
        ctx = CompileCtx.Current()
        self.isProceduralBlock = ctx.isProceduralBlock
        self.isInitialBlock = ctx.isInitialBlock
        self.isCombinationalBlock = ctx.proceduralBlock is not None and \
            ctx.proceduralBlock.logicType == ProceduralBlock.LogicType.COMB
        lhs._Wire(True, frameDepth + 1)
        rhs._Wire(False, frameDepth + 1)
        lhs._Assign(None, frameDepth + 1)

        if self.isProceduralBlock:
            if not self.isBlocking and not self.isCombinationalBlock:
                for e in self.lhs._GetLeafNodes():
                    if isinstance(e, Net) and not e.isReg:
                        raise ParseException(f"Procedural assignment to wire {e}")
        else:
            for e in self.lhs._GetLeafNodes():
                if isinstance(e, Net) and e.isReg:
                    ctx.Warning(f"Continuous assignment to register {e}", self.srcFrame)

        assert not lhs.isUnboundSize
        if not Dimensions.MatchAny(self.lhs.dims, self.rhs.dims, True):
            raise ParseException(
                "Arrays of different shape in assignment statement: "
                f"{Dimensions.StrAny(self.lhs.dims)} <= {Dimensions.StrAny(self.rhs.dims)}")

        if rhs.vectorSize > lhs.vectorSize:
            raise ParseException(f"Assignment size exceeded: {lhs.vectorSize} bits <<= {rhs.vectorSize} bits")
        elif not rhs.isUnboundSize and rhs.vectorSize < lhs.vectorSize:
            ctx.Warning(f"Assignment of insufficient size: {lhs.vectorSize} bits <<= {rhs.vectorSize} bits",
                        self.srcFrame)


    def Render(self, ctx: RenderCtx):
        if self.isProceduralBlock:
            op = "=" if self.isBlocking or self.isInitialBlock or self.isCombinationalBlock else "<="
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
    # "z" or "x" for `casez` or `casex`
    flavour: Optional[str] = None
    switch: Expression
    conditions: List[Expression]
    blocks: List[Block]
    defaultBlock: Optional[Block] = None
    # Catches any statements inside `when` body. There should be nothing in normal case.
    dummyBlock: Block


    def __init__(self, flavour: Optional[str], switch: Expression, frameDepth):
        super().__init__(frameDepth + 1)
        if flavour is not None:
            if flavour != "z" and flavour != "x":
                raise Exception(f"Bad flavour: {flavour}")
            self.flavour = flavour
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
        flavour = self.flavour if self.flavour is not None else ""
        ctx.Write(f"case{flavour} (")
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
        ctx = CompileCtx.Current()
        for e in self.conditions:
            if (self.switch.isArray or e.isArray):
                if not Dimensions.MatchAny(self.switch.dims, e.dims):
                    raise ParseException(
                        f"Case expression shape mismatch: {Dimensions.StrAny(self.switch.dims)} != "
                        f"{Dimensions.StrAny(e.dims)} ({e})")
            elif (e.vectorSize != self.switch.vectorSize and not e.isUnboundSize) or \
                e.vectorSize > self.switch.vectorSize:
                ctx.Warning(f"'`_when` expression size mismatch: {self.switch.vectorSize} != {e.vectorSize} ({e})")


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
    class LogicType(Enum):
        NONE = auto()
        COMB = auto()
        FF = auto()
        LATCH = auto()


    allowedScope = StatementScope.NON_PROCEDURAL
    sensitivityList: Optional[SensitivityList | EdgeTrigger]
    body: Block
    logicType: LogicType



    def __init__(self, sensitivityList: Optional[SensitivityList | EdgeTrigger],
                 logicType=LogicType.NONE, *, frameDepth: int):
        super().__init__(frameDepth + 1, deferPush=True)
        self.sensitivityList = sensitivityList
        self.logicType = logicType
        if sensitivityList is not None and logicType != ProceduralBlock.LogicType.NONE and \
            logicType != ProceduralBlock.LogicType.FF:
            raise Exception(f"Sensitivity list cannot be specified for logic type {logicType}")


    def __enter__(self):
        self.body = Block(1)
        ctx = CompileCtx.Current()
        if ctx.isProceduralBlock:
            raise ParseException("Nested procedural block")
        ctx.PushBlock(self.body)
        ctx.proceduralBlock = self


    def __exit__(self, excType, excValue, tb):
        ctx = CompileCtx.Current()
        if ctx.PopBlock() is not self.body:
            raise Exception("Unexpected current block")
        ctx.proceduralBlock = None
        if len(self.body) == 0:
            ctx.Warning(f"Empty procedural block", self.srcFrame)
        else:
            if self.sensitivityList is not None:
                self.sensitivityList._Wire(1)
            ctx.PushStatement(self)


    def Render(self, ctx: RenderCtx):
        if self.logicType == ProceduralBlock.LogicType.NONE:
            if ctx.options.svProceduralBlocks:
                if self.sensitivityList is None:
                    ctx.Write("always_comb")
                else:
                    ctx.Write("always_ff @")
            else:
                ctx.Write("always @")
        elif self.logicType == ProceduralBlock.LogicType.COMB:
            ctx.Write("always_comb")
        elif self.logicType == ProceduralBlock.LogicType.FF:
            if self.sensitivityList is None:
                raise ParseException("Sensitivity list not specified for FF block")
            ctx.Write("always_ff @")
        elif self.logicType == ProceduralBlock.LogicType.LATCH:
            ctx.Write("always_latch")
        else:
            raise Exception(f"Unhandled logic type: {self.logicType}")

        if self.sensitivityList is None:
            if not ctx.options.svProceduralBlocks and \
               self.logicType == ProceduralBlock.LogicType.NONE:
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
            raise ParseException("Nested initial block")
        ctx.PushBlock(self.body)
        ctx.isInitialBlock = True


    def __exit__(self, excType, excValue, tb):
        ctx = CompileCtx.Current()
        if ctx.PopBlock() is not self.body:
            raise Exception("Unexpected current block")
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
            if not e.isUnboundSize and not Dimensions.MatchAny(port.dims, e.dims):
                CompileCtx.Current().Warning(
                    f"Port `{port}` binding shape mismatch: "
                    f"{Dimensions.StrAny(port.dims)} != {Dimensions.StrAny(e.dims)} ({e})")
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


class VerilatorLintOffStatement(Statement):
    warnNames: Sequence[str]
    body: Block


    def __init__(self, warnNames: Sequence[str], frameDepth: int):
        super().__init__(frameDepth + 1, deferPush=True)
        self.warnNames = warnNames


    def __enter__(self):
        self.body = Block(1)
        ctx = CompileCtx.Current()
        ctx.PushBlock(self.body)


    def __exit__(self, excType, excValue, tb):
        ctx = CompileCtx.Current()
        if ctx.PopBlock() is not self.body:
            raise Exception("Unexpected current block")
        if len(self.body) == 0:
            ctx.Warning(f"Empty `verilator_lint_off` block", self.srcFrame)
        else:
            ctx.PushStatement(self)


    def Render(self, ctx: RenderCtx):
        isFirst = True
        for name in self.warnNames:
            if isFirst:
                isFirst = False
            else:
                ctx.WriteIndent(self.indent)
            ctx.Write(f"// verilator lint_off {name}\n")
        self.body.Render(ctx, -1)
        isFirst = True
        for name in reversed(self.warnNames):
            if isFirst:
                isFirst = False
            else:
                ctx.Write("\n")
            ctx.WriteIndent(self.indent)
            ctx.Write(f"// verilator lint_on {name}")


class AssertStatement(Statement):
    allowedScope = StatementScope.PROCEDURAL
    condition: Expression


    def __init__(self, condition: Expression, frameDepth):
        super().__init__(frameDepth + 1)
        self.condition = Expression._CheckType(condition)
        self.condition._Wire(False, frameDepth + 1)


    def Render(self, ctx: RenderCtx):
        ctx.Write("`ASSERT(")
        self.condition.Render(ctx)
        ctx.Write(")")

import collections.abc
from typing import List, Optional, Tuple, Type, cast
from gateforge.core import ArithmeticExpr, CaseContext, CompileCtx, ConditionalExpr, Const, \
    Expression, IfContext, IfStatement, InitialBlock, Module, ModuleParameter, Namespace, Net, \
    NetProxy, ParseException, ProceduralBlock, Reg, SensitivityList, WhenStatement, Wire, \
    RawExpression


def const(value: str | int | bool, size: Optional[int] = None) -> Const:
    return Const(value, size, frameDepth=1)


def _CreateNet(isReg: bool,
               sizeOrName: int | List[int] | Tuple[int] | str | None,
               name: Optional[str]) -> Reg | Wire:
    size = 1
    baseIndex = 0
    if sizeOrName is not None:
        if isinstance(sizeOrName, str):
            if name is not None:
                raise ParseException("Name specified twice")
            name = sizeOrName
        elif isinstance(sizeOrName, int):
            size = sizeOrName
        else:
            if not isinstance(sizeOrName, collections.abc.Sequence):
                raise ParseException("Sequence expected for net indices range")
            idxHigh, baseIndex = sizeOrName # type: ignore
            if idxHigh < baseIndex:
                raise ParseException(f"Bad net indices range, {idxHigh} < {baseIndex}")
            size = idxHigh - baseIndex + 1

    ctr: Type[Reg | Wire] = Reg if isReg else Wire
    return ctr(size=size, baseIndex=baseIndex, isReg=isReg, name=name, frameDepth=2)


def wire(sizeOrName: int | List[int] | Tuple[int] | str | None = None, /,
         name: Optional[str] = None) -> Wire:
    return cast(Wire, _CreateNet(False, sizeOrName, name))


def reg(sizeOrName: int | List[int] | Tuple[int] | str | None = None, /,
        name: Optional[str] = None) -> Reg:
    return cast(Reg, _CreateNet(True, sizeOrName, name))


def always(sensitivityList: SensitivityList | Net | ArithmeticExpr | None = None) -> ProceduralBlock:
    sl: Optional[SensitivityList]
    if isinstance(sensitivityList, Net):
        sl = SensitivityList(1)
        sl.PushSignal(sensitivityList)
    elif isinstance(sensitivityList, ArithmeticExpr):
        # Should be wire | wire
        sl = sensitivityList._ToSensitivityList(1)
    else:
        sl = sensitivityList
    return ProceduralBlock(sl, 1)


def initial() -> InitialBlock:
    return InitialBlock(1)


def cond(condition: Expression, ifCase: RawExpression, elseCase: RawExpression) -> ConditionalExpr:
    return ConditionalExpr(condition, ifCase, elseCase, 1)


def _if(condition: Expression) -> IfContext:
    return IfStatement(1)._GetContext(condition)


def _elseif(condition: Expression) -> IfContext:
    block = CompileCtx.Current().curBlock
    stmt = block.lastStatement if len(block) > 0 else None
    if not isinstance(stmt, IfStatement):
        raise ParseException("No `_if` statement to apply `_elseif` onto")
    return stmt._GetContext(condition)


def _else() -> IfContext:
    block = CompileCtx.Current().curBlock
    stmt = block.lastStatement if len(block) > 0 else None
    if not isinstance(stmt, IfStatement):
        raise ParseException("No `_if` statement to apply `_else` onto")
    return stmt._GetContext(None)


def _when(switch: Expression) -> WhenStatement:
    return WhenStatement(switch, 1)


def _case(condition: RawExpression) -> CaseContext:
    ctx = CompileCtx.Current()
    block = ctx._blockStack[-2] if len(ctx._blockStack) > 1 else None
    stmt = block.lastStatement if block is not None and len(block) > 0 else None
    if not isinstance(stmt, WhenStatement):
        raise ParseException("No `_when` statement to apply `_case` onto")
    return stmt._GetContext(Expression._FromRaw(condition, 1))


def _default() -> CaseContext:
    ctx = CompileCtx.Current()
    block = ctx._blockStack[-2] if len(ctx._blockStack) > 1 else None
    stmt = block.lastStatement if block is not None and len(block) > 0 else None
    if not isinstance(stmt, WhenStatement):
        raise ParseException("No `_when` statement to apply `_default` onto")
    return stmt._GetContext(None)


def parameter(name: str) -> ModuleParameter:
    return ModuleParameter(name, 1)


def module(moduleName: str, *args: NetProxy | ModuleParameter) -> Module:
    ports: dict[str, NetProxy] = dict()
    params: dict[str, ModuleParameter] = dict()

    for arg in args:

        if isinstance(arg, ModuleParameter):
            if arg.name in ports:
                raise ParseException(f"Parameter name conflicts with port name: {arg.name}")
            if arg.name in params:
                raise ParseException(f"Duplicate parameter name: {arg.name}")
            params[arg.name] = arg
            continue

        if not isinstance(arg, NetProxy):
            raise ParseException(f"NetProxy instance expected, has `{type(arg).__name__}`")

        if arg.initialName is None:
            raise ParseException(f"Unnamed net cannot be used as module port in a declaration, {arg}")
        if arg.initialName in ports:
            raise ParseException(f"Duplicate port name in a module declaration: `{arg.initialName}`")
        if arg.initialName in params:
            raise ParseException(f"Port name conflicts with parameter name: {arg.initialName}")
        ports[arg.initialName] = arg

    return Module(moduleName, ports, params, 1)


def namespace(name: str) -> Namespace:
    return Namespace(name, 1)

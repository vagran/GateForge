from typing import List, Optional, Sequence, Type, cast
from gateforge.core import ArithmeticExpr, AssertStatement, CaseContext, CompileCtx, ConcatExpr, \
    ConditionalExpr, Const, Dimensions, EdgeTrigger, Expression, FunctionCallExpr, IfContext, \
    IfStatement, InitialBlock, Module, ModuleParameter, Namespace, Net, NetProxy, ParseException, \
    ProceduralBlock, Reg, SensitivityList, VerilatorLintOffStatement, WhenStatement, Wire, \
    RawExpression


def const(value: str | int | bool, size: Optional[int] = None) -> Const:
    return Const(value, size, frameDepth=1)


def _CreateNet(isReg: bool,
               dimsOrName: str | int | Sequence[int] | None,
               dims: Sequence[int | Sequence[int]]) -> Reg | Wire:
    name: Optional[str] = None
    _dims: Optional[List[int | Sequence[int]]] = None
    if dimsOrName is not None:
        if isinstance(dimsOrName, str):
            name = dimsOrName
        else:
            _dims = [dimsOrName]
        if len(dims) > 0:
            if _dims is None:
                _dims = list(dims)
            else:
                _dims.extend(dims)
    ctr: Type[Reg | Wire] = Reg if isReg else Wire
    return ctr(dims=Dimensions.Parse(_dims, None) if _dims is not None else None,
               isReg=isReg, name=name, frameDepth=2)


def wire(dimsOrName: str | int | Sequence[int] | None = None,
         *dims: int | Sequence[int]) -> Wire:
    return cast(Wire, _CreateNet(False, dimsOrName, dims))


def reg(dimsOrName: str | int | Sequence[int] | None = None,
        *dims: int | Sequence[int]) -> Reg:
    return cast(Reg, _CreateNet(True, dimsOrName, dims))


def concat(*items: RawExpression) -> ConcatExpr:
    return ConcatExpr(items, 1)


def _CreateProceduralBlock(sensitivityList: SensitivityList | EdgeTrigger | Net | ArithmeticExpr | None,
                           logicType: ProceduralBlock.LogicType) -> ProceduralBlock:
    sl: Optional[SensitivityList | EdgeTrigger]
    if isinstance(sensitivityList, Net):
        sl = SensitivityList(1)
        sl.PushSignal(sensitivityList)
    elif isinstance(sensitivityList, ArithmeticExpr):
        # Should be wire | wire
        sl = sensitivityList._ToSensitivityList(1)
    else:
        sl = sensitivityList
    return ProceduralBlock(sl, logicType, frameDepth=2)


def always(sensitivityList: SensitivityList | EdgeTrigger | Net | ArithmeticExpr | None = None) -> \
    ProceduralBlock:
    return _CreateProceduralBlock(sensitivityList, ProceduralBlock.LogicType.NONE)


def always_ff(sensitivityList: SensitivityList | Net | ArithmeticExpr) -> ProceduralBlock:
    return _CreateProceduralBlock(sensitivityList, ProceduralBlock.LogicType.FF)


def always_comb() -> ProceduralBlock:
    return _CreateProceduralBlock(None, ProceduralBlock.LogicType.COMB)


def always_latch() -> ProceduralBlock:
    return _CreateProceduralBlock(None, ProceduralBlock.LogicType.LATCH)


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
    return WhenStatement(None, switch, 1)


def _whenz(switch: Expression) -> WhenStatement:
    return WhenStatement("z", switch, 1)


def _whenx(switch: Expression) -> WhenStatement:
    return WhenStatement("x", switch, 1)


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


def verilator_lint_off(*warnNames: str) -> VerilatorLintOffStatement:
    return VerilatorLintOffStatement(warnNames, 1)


def call(funcName: str, *args: RawExpression, dims: Optional[Dimensions] = None) -> FunctionCallExpr:
    return FunctionCallExpr(funcName, args, dims, 1)


def _assert(condition: Expression) -> AssertStatement:
    return AssertStatement(condition, 1)

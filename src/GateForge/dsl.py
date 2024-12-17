import collections.abc
from typing import List, Optional, Tuple, Type, cast
from GateForge.core import ArithmeticExpr, ConditionalExpr, Const, Expression, Net, \
    ParseException, ProceduralBlock, Reg, SensitivityList, Wire


def const(value: str | int, size: Optional[int] = None) -> Const:
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


def cond(condition: Expression, ifCase: Expression | int, elseCase: Expression | int) -> ConditionalExpr:
    return ConditionalExpr(condition, ifCase, elseCase, 1)

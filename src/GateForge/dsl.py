from typing import Optional
from GateForge.core import Const


def const(value: str | int, size: Optional[int] = None):
    return Const(value, size, frameDepth=1)

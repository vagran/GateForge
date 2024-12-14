from dataclasses import dataclass
from io import TextIOBase
import io
from typing import Callable, List, Optional

from GateForge.core import CompileCtx, RenderOptions, WarningMsg


@dataclass
class CompileResult:
    warnings: List[WarningMsg]


def CompileModule(moduleFunc: Callable[[], None], outputStream: TextIOBase, *,
                  renderOptions: RenderOptions = RenderOptions(),
                  moduleName: Optional[str] = None) -> CompileResult:
    if moduleName is None:
        moduleName = moduleFunc.__name__
    compileCtx = CompileCtx(moduleName)
    CompileCtx.Open(compileCtx, 1)
    try:
        moduleFunc()
        compileCtx.Render(outputStream, renderOptions)
        return CompileResult(warnings=list(compileCtx.GetWarnings()))
    finally:
        CompileCtx.Close()


def CompileModuleToString(moduleFunc: Callable[[], None], *,
                          renderOptions: RenderOptions = RenderOptions(),
                          moduleName: Optional[str] = None) -> str:
    output = io.StringIO()
    CompileModule(moduleFunc, output, renderOptions=renderOptions, moduleName=moduleName)
    return output.getvalue()

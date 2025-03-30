import contextlib
from dataclasses import dataclass
import importlib.util
from io import TextIOBase
import io
import os.path
from typing import Any, Callable, List, Optional

from gateforge.core import CompileCtx, RenderOptions, WarningMsg
from gateforge.verilator import SimulationModel, Verilator, VerilatorParams


@dataclass
class CompileResult:
    warnings: List[WarningMsg]
    # Result returned by design module function
    result: Any
    # Simulation model if verilation requested
    simulationModel: Optional[SimulationModel]


class NullOutput(io.StringIO):
    def write(self, s, /):
        pass


def CompileModule(moduleFunc: Callable[[], Any], outputStream: TextIOBase = NullOutput(), *,
                  renderOptions: RenderOptions = RenderOptions(),
                  moduleName: Optional[str] = None,
                  moduleArgs: List[Any] = list(),
                  moduleKwargs: dict[str, Any] = dict(),
                  verilatorParams: Optional[VerilatorParams] = None) -> CompileResult:
    """Compile module into Verilog.

    :param moduleFunc: DSL function which defines the module.
    :param outputStream: Stream to output resulting Verilog to.
    :param renderOptions:
    :param moduleName: Module name, defaults to module function name.
    :param moduleArgs: Positional arguments to pass to module function.
    :param moduleKwargs: Keyword arguments to pass to module function.
    :param verilatorParams: Parameters for Verilator call. Does not produce simulation module if
        None.
    :return: CompileResult instance.
    """

    if moduleName is None:
        moduleName = moduleFunc.__name__
    compileCtx = CompileCtx(moduleName)
    if verilatorParams is not None:
        verilator = Verilator(verilatorParams, compileCtx)
    else:
        verilator = None
    CompileCtx.Open(compileCtx, 1)
    with (verilator if verilator is not None else contextlib.nullcontext()): # type: ignore
        try:
            result = moduleFunc(*moduleArgs, **moduleKwargs)
            compileCtx.Finish()
            compileCtx.Render(outputStream if verilator is None else verilator.GetOutputStream(outputStream),
                              renderOptions)
            if verilator is not None:
                verilator.Build()
            return CompileResult(warnings=list(compileCtx.GetWarnings()), result=result,
                                 simulationModel=verilator.GetModel() if verilator is not None else None)
        finally:
            CompileCtx.Close()


def CompileModuleToString(moduleFunc: Callable[[], Any], **kwargs) -> str:
    output = io.StringIO()
    CompileModule(moduleFunc, output, **kwargs)
    return output.getvalue()


def ResolveModuleSpec(moduleSpec: str) -> Callable[[], Any]:

    if ":" in moduleSpec:
        path, funcName = moduleSpec.split(":")
    else:
        path = moduleSpec
        funcName = None

    if os.path.exists(path):
        rootModuleName = os.path.basename(path)
        if funcName is None:
            moduleName = "__main__"
        else:
            moduleName = rootModuleName
        spec = importlib.util.spec_from_file_location(moduleName, path)
    else:
        rootModuleName = path
        spec = importlib.util.find_spec(path)

    if spec is None:
        raise Exception(f"Cannot create module spec from `{path}`")

    module = importlib.util.module_from_spec(spec)
    if funcName is not None:
        rootModuleName = funcName

    def Run(*args, **kwargs):
        spec.loader.exec_module(module)
        if funcName is not None:
            if hasattr(module, funcName):
                func = getattr(module, funcName)
                return func(*args, **kwargs)
            else:
                raise Exception(f"Module does not have function `{funcName}`")

    Run.__name__ = rootModuleName

    return Run


def CompileModuleByPath(moduleSpec: str, outputStream: TextIOBase, **kwargs):
    return CompileModule(ResolveModuleSpec(moduleSpec), outputStream, **kwargs)

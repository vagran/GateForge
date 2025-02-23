import ctypes
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
import subprocess
import tempfile
from typing import List, Optional

from gateforge.core import CompileCtx
from gateforge.verilator_cpp import CreateCppFile


@dataclass
class VerilatorParams:
    # Verilator executable path, use "verilator" system-installed executable if None.
    verilatorPath: Optional[str] = None
    # Directory to build the simulation library in. Some temporal location if None.
    buildDir: Optional[str] = None
    # Do not output Verilator console messages during build
    quite: bool = True

    #XXX include dirs
    #XXX defines


class SimulationModel:
    moduleName: str
    buildDir: Path
    portNames: List[str]
    _lib: ctypes.CDLL
    _ctx: ctypes.c_void_p


    def __init__(self, *, moduleName: str, buildDir: Path, portNames: List[str]):
        self.moduleName = moduleName
        self.buildDir = buildDir
        self.portNames = portNames
        self._lib = ctypes.CDLL(str(self.buildDir / "obj" / f"V{moduleName}"))

        self._lib.Construct.restype = ctypes.c_void_p
        self._lib.Destruct.argtypes = [ctypes.c_void_p]
        self._lib.Eval.argtypes = [ctypes.c_void_p]
        self._lib.TimeInc.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        self._lib.OpenVcd.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.CloseVcd.argtypes = [ctypes.c_void_p]
        self._lib.DumpVcd.argtypes = [ctypes.c_void_p]

        self.ports = self._CreatePorts()

        self._ctx = self._lib.Construct()


    def _CreatePorts(self):

        class Ports:
            pass

        for portName in self.portNames:

            cGetter = getattr(self._lib, f"GateForge_Get_{portName}")
            cGetter.argtypes = [ctypes.c_void_p]
            cGetter.restype = ctypes.c_ulonglong

            cSetter = getattr(self._lib, f"GateForge_Set_{portName}")
            cSetter.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]

            def getter(_self, cGetter=cGetter):
                return cGetter(self._ctx)

            def setter(_self, value, cSetter=cSetter):
                cSetter(self._ctx, value)

            setattr(Ports, portName, property(getter, setter))

        return Ports()


    def _CheckCtx(self):
        if self._ctx is None:
            raise Exception("Using closed context")


    def Close(self):
        self._CheckCtx()
        self._lib.Destruct(self._ctx)
        self._ctx = None


    def Reload(self):
        """
        Reload context, resetting simulation state.
        """
        self._CheckCtx()
        self._lib.Destruct(self._ctx)
        self._ctx = self._lib.Construct()


    def Eval(self):
        self._CheckCtx()
        self._lib.Eval(self._ctx)


    def TimeInc(self, add: int):
        self._CheckCtx()
        self._lib.TimeInc(self._ctx, add)


    def OpenVcd(self, path):
        self._CheckCtx()
        self._lib.OpenVcd(self._ctx, str(path).encode())


    def CloseVcd(self):
        self._CheckCtx()
        self._lib.CloseVcd(self._ctx)


    def DumpVcd(self):
        self._CheckCtx()
        self._lib.DumpVcd(self._ctx)


class Verilator:
    params: VerilatorParams
    compileCtx: CompileCtx
    buildDir: Path
    modulePath: Path
    outputStream: TextIOBase


    def __init__(self, params: VerilatorParams, compileCtx: CompileCtx):
        self.params = params
        self.compileCtx = compileCtx
        if params.buildDir is not None:
            self.buildDir = Path(params.buildDir)
            self.buildDir.mkdir(exist_ok=True)
        else:
            self.buildDir = Path(tempfile.mkdtemp(prefix=f"GateForge_{compileCtx.moduleName}"))
        self.modulePath = self.buildDir / f"{compileCtx.moduleName}.v"


    def __enter__(self):
        self.outputStream = open(self.modulePath, "w")


    def __exit__(self, excType, excValue, tb):
        self.outputStream.close()


    def GetOutputStream(self, originalStream: TextIOBase) -> TextIOBase:
        """Intercept original output stream to get the compiled result to Verilator.

        :param originalStream: Original stream passed into compilation function.
        :return: Wrapped stream to provide to renderer.
        """

        originalWrite = self.outputStream.write

        def Write(s, /):
            originalWrite(s)
            originalStream.write(s)

        setattr(self.outputStream, "write", Write)

        return self.outputStream


    def Build(self):
        self.outputStream.close()

        cppFilePath = self.buildDir / f"GateForge_{self.compileCtx.moduleName}.cpp"
        CreateCppFile(cppFilePath, self.compileCtx.moduleName, self.compileCtx.GetPortNames())

        verilatorExe = self.params.verilatorPath if self.params.verilatorPath is not None else "verilator"
        args = [verilatorExe, "--binary", "--trace", "--Mdir", str(self.buildDir / "obj"),
                "-j", "0",
                "-CFLAGS", "-fPIC -shared", "-LDFLAGS", "-fPIC -shared",
                str(self.modulePath), str(cppFilePath)]
        if self.params.quite:
            subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        else:
            subprocess.check_call(args)


    def GetModel(self) -> SimulationModel:
        return SimulationModel(moduleName=self.compileCtx.moduleName, buildDir=self.buildDir,
                               portNames=list(self.compileCtx.GetPortNames()))

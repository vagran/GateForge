from typing import Optional, Sequence
import unittest
from gateforge.compiler import CompileResult
from gateforge.core import CompileCtx, WarningMsg


class WarningTracker:

    def __init__(self,  tc: unittest.TestCase, ctx: Optional[CompileCtx] = None,
                 result: Optional[CompileResult] = None):
         self.tc = tc
         self.ctx = ctx
         self.result = result
         self.prevWarningsCount = 0


    def GetWarnings(self) -> Sequence[WarningMsg]:
        if self.ctx is not None:
            return self.ctx.GetWarnings()
        assert self.result is not None
        return self.result.warnings


    def Check(self, expectedWarnings: int | str | Sequence[str]):
        if isinstance(expectedWarnings, int):
            numExpectedWarnings = expectedWarnings
        elif isinstance(expectedWarnings, str):
            numExpectedWarnings = 1
            expectedWarnings = [expectedWarnings]
        else:
            numExpectedWarnings = len(expectedWarnings)

        def DumpWarnings():
            print("\n============================================================================\n"
                  f"Warnings in failed warning check ({len(self.GetWarnings()) - self.prevWarningsCount}):")
            for i, w in enumerate(self.GetWarnings()[self.prevWarningsCount:]):
                print(f"{i + 1}. {w}")
            print("============================================================================\n")

        if not isinstance(expectedWarnings, int):
            for warnText in expectedWarnings:
                for w in self.GetWarnings()[self.prevWarningsCount:]:
                    if warnText in w.msg:
                        break
                else:
                    DumpWarnings()
                    self.tc.fail(f"Expected warning `{warnText}` not found")

        if (len(self.GetWarnings()) - self.prevWarningsCount != numExpectedWarnings):
            DumpWarnings()
        self.tc.assertEqual(len(self.GetWarnings()) - self.prevWarningsCount,
                            numExpectedWarnings)
        self.prevWarningsCount += numExpectedWarnings

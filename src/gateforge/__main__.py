import argparse
import sys

from gateforge.compiler import CompileModuleByPath
from gateforge.core import RenderOptions


def Main():
    parser = argparse.ArgumentParser(prog="python -m gateforge", description="Hardware synthesis tool")

    parser.add_argument("input", metavar="INPUT_FILE_OR_MODULE", type=str,
                        help="Input top-level module. May be Python module or function in a module "
                             "(separate function name by colon from file path or module)")
    parser.add_argument("--output", "-o", metavar="OUTPUT_FILE", type=str,
                        help="Output Verilog file. Outputs to stdout if not specified.")
    parser.add_argument("--sourceMap", action="store_true",
                        help="Inject comments with Python source locations")
    parser.add_argument("--moduleName", type=str,
                        help="Module name for produced top-level module.")

    args = parser.parse_args()

    renderOptions = RenderOptions()
    if args.sourceMap:
        renderOptions.sourceMap = True

    options = {
        "renderOptions": renderOptions,
        "moduleName": args.moduleName
    }

    if args.output is None:
        CompileModuleByPath(args.input, sys.stdout, **options)
    else:
        with open(args.output, "w") as output:
            CompileModuleByPath(args.input, output, **options)


if __name__ == "__main__":
    Main()

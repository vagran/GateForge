import argparse
import sys

from GateForge.compiler import CompileModuleByPath


def Main():
    parser = argparse.ArgumentParser(prog="GateForge", description="Hardware synthesis tool")

    parser.add_argument("input", metavar="INPUT_FILE_OR_MODULE", type=str,
                        help="Input top-level module. May be Python module or function in a module "
                             "(separate function name by colon from file path or module)")
    parser.add_argument("--output", "-o", metavar="OUTPUT_FILE", type=str,
                        help="Output Verilog file. Outputs to stdout if not specified.")

    args = parser.parse_args()

    if args.output is None:
        CompileModuleByPath(args.input, sys.stdout)
    else:
        with open(args.output, "w") as output:
            CompileModuleByPath(args.input, output)


if __name__ == "__main__":
    Main()

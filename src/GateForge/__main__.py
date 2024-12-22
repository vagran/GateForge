import argparse


def Main():
    parser = argparse.ArgumentParser(prog="GateForge", description="Hardware synthesis tool")

    parser.add_argument("input", metavar="INPUT_FILE", type=str,
                        help="Input top-level module. May be Python module or function in a module.")
    parser.add_argument("--output", "-o", metavar="OUTPUT_FILE", type=str, required=True,
                        help="Output Verilog file")

    args = parser.parse_args()


if __name__ == "__main__":
    Main()

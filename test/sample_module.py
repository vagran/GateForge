from gateforge.concepts import Interface
from gateforge.core import Reg, Wire
from gateforge.dsl import _else, _elseif, _if, always, namespace, reg, wire


class Shifter:

    class _Interface(Interface["_Interface"]):
        clk: Wire
        setSig: Wire
        dir: Wire
        input: Wire
        output: Reg

    iface: _Interface


    def __init__(self, size: int = 32):
        self.size = size
        with namespace("Shifter"):
            self.iface = Shifter._Interface.Create(
                clk=wire("CLK").input,
                setSig=wire("SET").input,
                dir=wire("DIR").input,
                input=wire(size, "IN").input,
                output=reg(size, "OUT").output)


    def __call__(self):

        with always(self.iface.clk.negedge):
            with _if(self.iface.setSig):
                self.iface.output <<= self.iface.input
            with _elseif(self.iface.dir):
                self.iface.output <<= self.iface.output[self.size - 2:0] % self.iface.output[self.size - 1]
            with _else():
                self.iface.output <<= self.iface.output[0] % self.iface.output[self.size - 1:1]


def ShifterModule(size: int = 32):

    shifter = Shifter(size)

    shifter.iface.external.Assign(
        clk=wire("CLK").input.port,
        setSig=wire("SET").input.port,
        dir=wire("DIR").input.port,
        input=wire(shifter.size, "IN").input.port,
        output=reg(shifter.size, "OUT").output.port)

    shifter()


if __name__=="__main__":
    ShifterModule()

from GateForge.dsl import _else, _elseif, _if, always, reg, wire


class Shifter:
    def __init__(self, size: int = 32):
        self.size = size
        self.clk = wire("CLK")
        self.setSig = wire("SET")
        self.dir = wire("DIR")
        self.input = wire(size, "IN")
        self.output = reg(size, "OUT")


    def __call__(self):
        with always(self.clk.negedge):
            with _if(self.setSig):
                self.output <<= input
            with _elseif(self.dirSig):
                self.output <<= self.output[self.size - 2:0] % self.output[self.size - 1]
            with _else():
                self.output <<= self.output[0] % self.output[self.size - 1:1]


def ShifterModule():
    pass

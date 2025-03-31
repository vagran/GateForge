# GateForge: Python RTL hardware design framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Version](https://img.shields.io/pypi/v/gateforge)](https://pypi.org/project/gateforge/)


This is an open-source Python framework for designing Register-Transfer Level (RTL) hardware. It
provides a domain-specific language (DSL) for creating hardware descriptions that compile to
Verilog. The framework bridges the gap between high-level Python expressiveness and the constraints
of open-source hardware toolchains like Yosys, which lack full support for SystemVerilog.

It does not tend to introduce any new concepts, but mostly just wraps Verilog into Python DSL, so
that you can use any Python features you want for metaprogramming over your synthesizable logic. If
you are familiar with Verilog and Python, then you are mostly familiar with this framework.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Wires and Registers](#wires-and-registers)
  - [Net Declarations](#net-declarations)
    - [Basic Wire Creation](#basic-wire-creation)
    - [Multi-dimensional Packed Arrays](#multi-dimensional-packed-arrays)
    - [Unpacked Arrays](#unpacked-arrays)
  - [Dimension Properties](#dimension-properties)
  - [Register Declarations](#register-declarations)
  - [Key Features](#key-features)
- [Bit Selection and Slicing](#bit-selection-and-slicing)
  - [Accessing Net Elements](#accessing-net-elements)
    - [Single-Bit Access](#single-bit-access)
    - [Multi-Bit Slicing (MSB:LSB)](#multi-bit-slicing-msblsb)
    - [Dynamic Bit Selection](#dynamic-bit-selection)
  - [Key Rules and Conversions](#key-rules-and-conversions)
  - [Error Checking Examples](#error-checking-examples)
- [Signal Naming and Direction](#signal-naming-and-direction)
  - [Explicit Signal Naming](#explicit-signal-naming)
  - [Signal Directions](#signal-directions)
  - [Hierarchical Namespaces](#hierarchical-namespaces)
    - [Namespace Features](#namespace-features)
- [Assignments and Concatenation](#assignments-and-concatenation)
  - [Basic Assignments](#basic-assignments)
  - [Concatenation Operators](#concatenation-operators)
- [Constants and Literals](#constants-and-literals)
  - [Constant Declaration Methods](#constant-declaration-methods)
  - [Concatenation rules](#concatenation-rules)
- [Ternary Operator Implementation](#ternary-operator-implementation)
- [Procedural Logic](#procedural-logic)
  - [Sequential Logic](#sequential-logic)
  - [Combinational Logic](#combinational-logic)
  - [`_when` statement](#_when-statement)
  - [SystemVerilog procedural blocks](#systemverilog-procedural-blocks)
  - [Initial Blocks](#initial-blocks)
- [Operators](#operators)
  - [Bitwise Operators](#bitwise-operators)
  - [Shift Operations](#shift-operations)
  - [Reduction Operators](#reduction-operators)
  - [Replication Operator](#replication-operator)
  - [Operator Precedence Solutions](#operator-precedence-solutions)
  - [Operator Reference Table](#operator-reference-table)
- [Functions](#functions)
  - [Signed Signal Handling](#signed-signal-handling)
  - [Custom Function Calls](#custom-function-calls)
- [Verilator Warning Suppression](#verilator-warning-suppression)
- [External Modules](#external-modules)
  - [Module Definition](#module-definition)
  - [Module Instantiation](#module-instantiation)
- [Simulation-time assertion](#simulation-time-assertion)
- [Module Compilation and Structure](#module-compilation-and-structure)
  - [`CompileModule()` Parameters](#compilemodule-parameters)
  - [RenderOptions Configuration](#renderoptions-configuration)
  - [Compilation example](#compilation-example)
- [Verilator integration](#verilator-integration)
- [High-level helpers](#high-level-helpers)
  - [Typing](#typing)
  - [Nets construction by annotations](#nets-construction-by-annotations)
  - [Bus](#bus)
  - [Interface](#interface)
- [Advanced example](#advanced-example)
- [License](#license)

## Overview
GateForge enables hardware design through Python constructs that translate to Verilog RTL, bridging
Python's expressiveness with open-source toolchain capabilities. Key advantages include:
- High-level abstractions for complex hardware
- Native integration with Verilator for simulation
- Type-safe RTL generation compatible with Yosys-based flows
- Python-native testbench development

## Installation
```bash
pip install gateforge
```

## Getting Started

**Basic XOR Module:**
```python
import sys
from gateforge.dsl import wire
from gateforge.compiler import CompileModule

def SampleModule():
    in1 = wire("in1").input.port
    in2 = wire("in2").input.port
    out1 = wire("out1").output.port
    out1 <<= in1 ^ in2

CompileModule(SampleModule, sys.stdout)
```

**Verilator Test Case:**
```python
import unittest
import SampleModule from SampleModule
from gateforge.verilator import VerilatorParams

class TestXOR(unittest.TestCase):
    def setUp(self):
        vp = VerilatorParams(buildDir="build")
        self.sim = CompileModule(SampleModule, verilatorParams=vp).simulation_model

    def test_xor_behavior(self):
        self.sim.ports.in1 = 0
        self.sim.ports.in2 = 0
        self.sim.eval()
        self.assertEqual(self.sim.ports.out1, 0)

        self.sim.ports.in1 = 1
        self.sim.eval()
        self.assertEqual(self.sim.ports.out1, 1)
```

## Wires and Registers

### Net Declarations
The framework provides Python-idiomatic ways to create wires and registers with various dimensional
configurations.

#### Basic Wire Creation
```python
# Single-bit anonymous wire
w1 = wire()

# 4-bit wire (big-endian, indices 0-3)
w2 = wire(4)          # Verilog: wire [3:0] w2

# 5-bit wire with custom indices
w3 = wire([7, 3])     # Verilog: wire [7:3] w3

# Little-endian 5-bit wire
w4 = wire([3, 7])     # Verilog: wire [3:7] w4
```

#### Multi-dimensional Packed Arrays
```python
# 2D packed array (5x8 bits)
w5 = wire([4, 0], [7, 0])  # Verilog: wire [4:0][7:0] w5

# 3D packed structure
w6 = wire([3,0], [15,8], [7,4]) # Verilog: wire [3:0][15:8][7:4] w6
```

#### Unpacked Arrays
Same dimensions specifying scheme applies to unpacked arrays.
```python
# 1D unpacked array (2 elements)
arr1 = wire(4).array(2)     # Verilog: wire [3:0] arr1[1:0]

# 2D unpacked array with custom indices
arr2 = wire(8).array([6, 2], [15, 10])
# Verilog: wire [7:0] arr2[6:2][15:10]

# Mixed packed/unpacked
arr3 = wire([7,0], [3,0]).array(4)
# Verilog: wire [7:0][3:0] arr3[3:0]
```

### Dimension Properties
```python
# Vector size calculation
w = wire([7,0], [3,0]).array(4)
assert w.vector_size == 8 * 4  # 32 bits (packed dimensions only)
```

### Register Declarations
Registers follow identical declaration syntax to wires, using `reg()` instead:
```python
# 8-bit register
r1 = reg(8)        # Verilog: reg [7:0] r1

# Multi-dimensional register
r2 = reg([3,0], [7,4])      # Verilog: reg [3:0][7:4] r2
```

### Key Features

1. **Indexing Schemes**:
   - `wire(N)` creates 0-based big-endian vectors
   - `wire([high, low])` creates custom ranges
   - Little-endian semantics is propagated to Verilog corresponding declaration.

2. **Dimension Propagation**:
   ```python
   # Packed then unpacked dimensions
   bus = wire([3,0], [7,4]).array(2)
   # Verilog: wire [3:0][7:4] bus[1:0]
   ```

## Bit Selection and Slicing

### Accessing Net Elements
The framework provides flexible bit selection mechanisms that mirror Verilog's capabilities while
maintaining Pythonic syntax.

#### Single-Bit Access
```python
# Access bit at position 6 (actual hardware bit depends on declaration)
bit6 = w3[6]  # Verilog: w3[6]
```

#### Multi-Bit Slicing (MSB:LSB)

```python
# Standard Verilog-style slice (inclusive)
upper_bits = w3[7:3]  # Verilog: w3[7:3]

# Python-style open ranges
first_bits = w3[:3]   # Verilog: w3[<msb>:3]
last_bits = w3[5:]    # Verilog: w3[5:<lsb>]
```
Note that slice follows verilog notation, the first is MSB index, the second is LSB inclusive index.
Endianness should correspond to the net declaration - little-endian wire should be accessed as
`w3[3:7]`.

#### Dynamic Bit Selection
```python
# Single-bit selection using another wire
dynamic_bit = w3[selector]  # Verilog: w3[selector]

# Valid for unpacked arrays
array_element = arr[index]  # Verilog: arr[index]
```

Dynamic slicing is not supported (tools like Verilator do not support this case).

### Key Rules and Conversions

| Python Operation     | Verilog Equivalent       | Notes                                      |
|----------------------|--------------------------|--------------------------------------------|
| `wire[7]`            | `wire[7]`                | Actual bit position depends on declaration |
| `wire[7:3]`          | `wire[7:3]`              | Inclusive range, MSB first                 |
| `wire[:3]`           | `wire[<msb>:3]`          | Full range from MSB to 3                   |
| `wire[5:]`           | `wire[5:<lsb>]`          | From 5 to LSB inclusive                    |
| `wire[var]`          | `wire[var]`              | Dynamic single-bit selection               |


### Error Checking Examples
```python
# Valid
wire8 = wire(8)
wire8[7:0]    # Full range
wire8[3]      # Single bit

# Invalid
wire8[8]      # IndexError: Bit out of range
wire8[3:7]    # ValueError: Reverse slice (MSB < LSB), mismatched endianness.
wire8[myReg:1]  # ValueError: Non-constant slice indices
```

## Signal Naming and Direction

### Explicit Signal Naming
```python
# Anonymous single-bit wire (auto-generated name)
w1 = wire()

# Named wire with explicit 1-bit width
cs = wire("CS")        # Verilog: wire CS;
cs = wire("CS", 1)     # Equivalent explicit form

# Named 8-bit register
counter = reg("COUNT", 8)  # Verilog: reg [7:0] COUNT;
```

Names collisions are resolved automatically by appending number suffix for conflicting name in the
resulting Verilog.

Names are only required if a net is used to define module port (see below). All the internal nets
may be anonymous, however it might be more convenient to provide descriptive names for most internal
nets for debugging and waveforms analyzing.

### Signal Directions
Signal directions are specified using method chaining:
```python
# Input wire
clk = wire("CLK").input       # Verilog: input wire CLK;

# Output register
result = reg("RESULT", 8).output  # Verilog: output reg [7:0] RESULT;
```

This can be used either for module ports (described below), or for internal signals. In former case
it does not have any special effect on the generated Verilog, but used for internal checks to
validate usage.

### Hierarchical Namespaces
```python
with namespace("PCIe"):
    # Creates wire PCIe_REQ
    req = wire("REQ").input

    with namespace("Tx"):
        # Creates wire PCIe_Tx_DATA_VALID
        valid = wire("DATA_VALID")
```

#### Namespace Features

- Namespace prefixes are cumulative in nested contexts
- Supports arbitrary depth of nesting
- Affects all signal types (wires, registers, ports)
- Generated Verilog uses underscore concatenation


## Assignments and Concatenation

### Basic Assignments

Use `<<=` to assign signal. In non-procedural context it always corresponds to continuous assignment.
```python
# Continuous assignment
cs <<= w1  # Verilog: assign cs = w1;
```
Remember that regular Python assignment just assigns a Python reference to the specified signal.

This is alternative assignment syntax which might be useful in some cases:
```python
cs.assign(w1)
```

### Concatenation Operators

```python
# Basic concatenation
w1 % w2 % w3[7:5]  # Verilog: {w1, w2, w3[7:5]}

# Assignment requires special handling to overcome Python restriction on augmented operators.
# `w1 % w2 % w3[7:5] <<= c1 % r1` is compilation error in Python.
(w1 % w2 % w3[7:5]).assign(c1 % r1)     # Verilog: assign {w1, w2, w3[7:5]} = {c1, r1};

# Alternative using intermediate variable
result = w1 % w2 % w3[7:5]
result <<= c1 % r1

# Function-style concatenation
result <<= concat(w1, w2, w3[7:5])
```


## Constants and Literals

### Constant Declaration Methods

```python
# Verilog-style string declaration
hex_const = const("5'ha")       # 5-bit hex: 5'h0a
wide_const = const("16'hxz2")   # 16'bxxxx_zzzz_0000_0010

# Python numeric declaration
dec_const = const(0xaa, 8)      # 8-bit 0xaa
# Boolean type is converted to single bit constant.
bool_const = const(True)
```

In most places Python `int` and `bool` type values can be used as is, corresponding constant is
inferred.
```python
cs <<= True
address <<= 0x8000
```

When constant does not have size specified (either inferred from `int` or declared without size),
its size is unbound, and implies some consequences, mostly the same as in Verilog in the same
situation.
```python
unsized = const("'h800")
otherUnsized = const(someIntValue)
```

### Concatenation rules

```python
w4 <<= 5 % w1     # Allowed: 3-bit + 1-bit = 4-bit
w4 <<= w1 % 5     # Error: Right constant needs explicit width
w4 <<= w1 % const(5, 3)  # Valid: 1 + 3 = 4-bit
```

## Ternary Operator Implementation

The framework provides two equivalent syntaxes for conditional assignments:
```python
# Functional style
w1 <<= cond(condition, true_expr, false_expr)

# Method-chaining style
w1 <<= condition.cond(true_expr, false_expr)
```


## Procedural Logic


### Sequential Logic

```python
# Edge-triggered
with always(clk.posedge | rst.negedge):
    # Non-blocking assignment
    counter <<= next_counter
    # Blocking assignment
    temp //= a + b
```

### Combinational Logic

```python
with always():
    with _if(sel == 0):
        out <<= a
    # Note that using parenthesis is mandatory since Python bitwise operators
    # have higher precedence over comparison operators.
    with _elseif((sel == 1) | (sel == 3)):
        out <<= b
    with _else():
        out <<= c
```

### `_when` statement

Wrapper for Verilog `case` statement is `_when`:
```python
with _when(w2):
    with _case(1):
        r1 <<= w1
    with _default():
        r1 <<= 5
```
There are `_whenz` and `_whenx` versions for `casez` and `casex` correspondingly.

### SystemVerilog procedural blocks

```python
# Combinational logic (auto-sensitivity)
with always_comb():
    y <<= a & b

# Clock-driven sequential logic
with always_ff(clk.posedge):
    q <<= d

# Explicit latch declaration
with always_latch():
    if en:
        q <<= d
```

### Initial Blocks

```python
# Power-up initialization (FPGA synthesis)
with initial():
    r1 <<= 42        # Verilog: initial r1 = 42;
```



## Operators

### Bitwise Operators
```python
# Standard bitwise operations
and_result = a & b   # Verilog: a & b
or_result = a | b    # Verilog: a | b
xor_result = a ^ b   # Verilog: a ^ b
not_result = ~a      # Verilog: ~a

# XNOR operation (Verilog-specific)
xnor_result = a.xnor(b)  # Verilog: a ~^ b
```

### Shift Operations
```python
# Logical left shift
shift_left = w1.sll(2)   # Verilog: w1 << 2

# Logical right shift (zero fill)
shift_right_log = w1.srl(3)  # Verilog: w1 >> 3

# Arithmetic right shift (sign extend)
shift_right_arith = w1.signed.sra(1)  # Verilog: $signed(w1) >>> 1
```

### Reduction Operators
```python
# Single-bit results from vector operations
all_and = w8.reduce_and    # Verilog: &w8
any_or = w8.reduce_or      # Verilog: |w8
parity = w8.reduce_xor     # Verilog: ^w8

# Inverted reductions
nand = w8.reduce_nand      # Verilog: ~(&w8)
nor = w8.reduce_nor        # Verilog: ~(|w8)
xnor_red = w8.reduce_xnor  # Verilog: ~^(w8)
```

### Replication Operator
```python
# Create repeated patterns
replicated = w4.replicate(3)  # Verilog: {3{w4}}
```

### Operator Precedence Solutions
```python
# Dangerous chained comparison ("Comparison operators chaining" Python feature)
if w1 < w2 == w3:  # Python: (w1 < w2) and (w2 == w3)
    ...             # Not equivalent to Verilog!

# Correct Verilog-style comparison
if (w1 < w2) == w3:  # Verilog: (w1 < w2) == w3
    ...
```

### Operator Reference Table

| Python Expression      | Verilog Equivalent       | Notes                              |
|------------------------|--------------------------|------------------------------------|
| `a & b`                | `a & b`                  | Bitwise AND                        |
| `a \| b`               | `a \| b`                 | Bitwise OR                         |
| `a ^ b`                | `a ^ b`                  | Bitwise XOR                        |
| `~a`                   | `~a`                     | Bitwise NOT                        |
| `a.xnor(b)`            | `a ~^ b`                 | XNOR gate                          |
| `w.reduce_and`         | `&w`                     | Vector AND reduction               |
| `w.replicate(n)`       | `{n{w}}`                 | Replication operator               |
| `a.sll(3)`             | `a << 3`                 | Logical left shift                 |
| `a.srl(3)`             | `a >> 3`                 | Logical right shift                |
| `a.sra(2)`             | `a >>> 2`                | Arithmetic right shift             |


## Functions

### Signed Signal Handling
`.signed` property is a shorthand for calling Verilog `$signed` built-in function.
```python
# Convert wire to signed interpretation
signed_wire = w1.signed  # Verilog: $signed(w1)
```

### Custom Function Calls
```python
# Some function call which do not have pre-defined wrapper. Result dimensions should be
# specified (omitting produces dimensionless result).
checksum <<= call("calc_crc32", data, Dimensions.Vector(32))
```


## Verilator Warning Suppression
```python

w1 = wire("w1", 2)
w2 = wire("w2")

# Suppress specific warnings for a code block
with verilator_lint_off("WIDTH"):
    w1 <<= w2
    # Generates:
    # // verilator lint_off WIDTH
    # assign w1 = w2;
    # // verilator lint_on WIDTH
```
`verilator_lint_off` may take multiple arguments for suppressing multiple warning types.


## External Modules

### Module Definition

In order to use external modules (provided by target platform or defined in separate Verilog files)
they should be defined first.

```python
# Define module interface
UART = module("UART",
    # Port list
    wire("TX").output,
    wire("RX").input,
    wire("CLK").input,
    # Parameters
    parameter("BAUD_RATE", default=115200),
    parameter("DATA_BITS", default=8)
)
```

### Module Instantiation
```python
def TopModule():
    # Instantiate with port connections
    UART(
        TX=tx_wire,
        RX=rx_reg,
        CLK=clk,
        BAUD_RATE=9600,
        DATA_BITS=8
    )
```


## Simulation-time assertion

 `_assert` statement exists to validate conditions in simulator. It is compiled to
 Verilog-compatible check which calls `$fatal` if condition evaluates to false.
```python
with always_comb():
    _assert(w1 == 42)
```


## Module Compilation and Structure

A design is compiled into a single top-level Verilog module. Any part of the entire design can be
taken, just inputs and outputs should be provided.

Module-level IO ports should be defined by taking `.port` property of a signal. The signal direction
must be specified as well for each port. Port names should be unique, errors produced for name
conflicts.

Design internal structure may pass and store Python references to signals and expressions. Python
replaces Verilog functionality for components parametrization and configuring (i.e. Verilog
parameters and `generate` blocks).

```python
def MyComponent(cs: Wire, d: Reg):
    # Internal logic using provided ports
    cs <<= d.reduce_xor()

def TopModule():
    # Declare and expose top-level ports
    cs = wire("CS").input.port  # Becomes module input
    d_out = reg("D_OUT", 8).output.port

    # Instantiate component with ports
    MyComponent(cs, d_out)

# Compilation entry point
CompileModule(TopModule, sys.stdout)
```


### `CompileModule()` Parameters

| Parameter          | Description                                  | Default                  |
|--------------------|----------------------------------------------|--------------------------|
| `moduleFunc`       | Python function defining module structure    | Required                 |
| `outputStream`     | Text stream for Verilog output               | Null output              |
| `renderOptions`    | Code generation settings (see below)         | RenderOptions()          |
| `moduleName`       | Override generated module name               | Function name            |
| `moduleArgs`       | Positional args to pass to moduleFunc        | []                       |
| `moduleKwargs`     | Keyword args to pass to moduleFunc           | {}                       |
| `verilatorParams`  | Verilator configuration (enables simulation) | None                     |

### RenderOptions Configuration

```python

# Custom rendering settings
options = RenderOptions(
    indent="  ",  # 2-space indentation
    sourceMap=True,  # Generate source mapping
    prohibitUndeclaredNets=False,  # Allow implicit nets
    svProceduralBlocks=True  # Use `always_ff`/`always_comb` for `always(sensList)` and `always()`
)

CompileModule(MyModule, renderOptions=options)
```

### Compilation example

**Parameterized Modules**:
```python
def ParamModule(width=8):
    data = reg("DATA", width).output.port

# Compile with parameter override
CompileModule(ParamModule,
              module_kwargs={"width": 16},
              module_name="WideModule")
```


## Verilator integration

Providing `verilatorParams` argument for `CompileModule()` function enables simulation of the
module. Here is a complete example:
```python
from pathlib import Path
import unittest

from gateforge.compiler import CompileModule
from gateforge.dsl import wire
from gateforge.verilator import VerilatorParams


def SampleModule():
    in1 = wire("in1").input.port
    in2 = wire("in2").input.port
    out1 = wire("out1").output.port
    out1 <<= in1 ^ in2


class TestBase(unittest.TestCase):

    def setUp(self):
        verilatorParams = VerilatorParams(buildDir=str(Path(__file__).parent / "workspace"),
                                          quite=False)
        self.result = CompileModule(SampleModule, verilatorParams=verilatorParams)
        self.sim = self.result.simulationModel
        self.ports = self.sim.ports
        self.sim.OpenVcd(workspaceDir / "test.vcd")


class TestBasic(TestBase):

    def test_basic(self):

        self.ports.in1 = 0
        self.ports.in2 = 0
        self.sim.Eval()
        self.sim.DumpVcd()
        self.assertEqual(self.ports.out1, 0)

        self.ports.in1 = 1
        self.sim.Eval()
        self.sim.DumpVcd()
        self.assertEqual(self.ports.out1, 1)

        self.ports.in2 = 1
        self.sim.Eval()
        self.sim.DumpVcd()
        self.assertEqual(self.ports.out1, 0)
```
Use `.OpenVcd()` and `.DumpVcd()` methods if waveform dump is needed.


## High-level helpers

The above functionality is mostly one-to-one mapped to generated Verilog. It is up to the framework
user to decide how to organize the design at higher level using all the power of Python. However,
several helpers are provided for typical tasks.

### Typing

The helpers below assume type annotations used for class members to provide the functionality. You
can use types from `gateforge.core` package to annotate members, arguments and return values like
`Expression`, `Net`, `Wire`, `Reg`, etc. Besides a type we also use dimensions specification in type
annotation which is not compatible with conventions used in Python. It does not cause any runtime
failures because type annotation in Python can technically be any object, but it causes warnings for
some linting tools. So it may require to disable some warnings for those tools for convenient
development.

`mypy` requires this line in the beginning of file with GateForge annotations:
```python
# mypy: disable-error-code="type-arg, valid-type"
```

VSCode Pylance requires this entry in `settings.json`:
```json
"python.analysis.diagnosticSeverityOverrides": {
    "reportInvalidTypeForm": "none"
}
```

### Nets construction by annotations

You can use `ConstructNets()` function from `gateforge.concepts` package to create instances for all
nets declared in a class. It does not override existing attributes, so typically some non-trivially
constructed nets are created explicitly first, then  `ConstructNets()` is called. Size specification
follows the same approach as wires and registers creations by `wire()` and `reg()` functions.
Attribute name is used as net name. `ConstructNets()` may be called in namespace context to make
necessary prefix for the created net names.

```python
class MyComponent:
    w1: Wire # self.w1 = wire("w1)
    w2: Wire[32] # self.w2 = wire("w2", 32)
    # Single tuples cannot be used in type annotation due to Python limitations. Use list to provide single range.
    w3: Wire[[31, 16]] # self.w3 = wire("w3, [31, 16])
    # Single tuple is interpreted as two values
    w3_tuple: Wire[(31, 16)] # self.w3_tuple = wire("w3, 31, 16)
    w4: Wire[4, [31, 16]] # self.w4 = wire("w4", 4, [31, 16])
    r1: Reg[16].array(8) # self.r1 = reg("r1", 16).array(8)
    r2: Reg[16].array(8, [15, 8]) # self.r1 = reg("r1", 16).array(8, [15, 8])
    r3: Reg # Value assigned in constructor so it is untouched by `ConstructNets()`

    def __init__(self, size: int):
        with namespace("MyComponent"):
            # Dynamically sized so construct explicitly
            self.r3 = reg("r3", size)
            # Construct the rest
            ConstructNets(self)
```

### Bus

Bus is used to group nets into a class. Bus requires all nets have specified direction. Use proxy
types `InputNet` and `OutputNet` parametrized by net type and optional dimensions. The bus class
should be inherited from `Bus` class parameterized by your class name to ensure proper type
inference for provided methods.
```python
class SampleBus(Bus["SampleBus"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]

class SizedBus(Bus["SizedBus"]):
    w: InputNet[Wire, (11, 8)]
    r: OutputNet[Reg, 8]
    uw: InputNet[Wire]
    ur: OutputNet[Reg]
```

It provides static method `.Create()` to create an instance. It expects all member values are
provided as keyword arguments:
```python
b = SampleBus.Create(w=wire().input, r=reg().output)
```
Note, that each signal direction should be specified by corresponding `.input` or `.output` property.

`.CreateDefault()` creates missing nets like `ConstructNets()` does:
```python
b = SampleBus.CreateDefault(w=wire())
```

Use `.Construct()` method for calling it from the class constructor:
```python
class SampleBusConstr(Bus["SampleBusConstr"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]

    def __init__(self):
        self.Construct(w=wire().input, r=reg().output)
```
`.ConstructDefault()` creates missing nets.

`.Assign()` instance method used for bulk assignments. It validates directions and checks all the
specified nets are declared:
```python
b.Assign(w=True, r=myPort)
```

`.Adjacent()` instance method returns new bus instance which has direction inverted for all member
nets.

### Interface

Interface is a replacement for SystemVerilog interfaces which are, for example, not available in
Yosys. It looks very similar to bus:
```python
class SampleInterface(Interface["SampleInterface"]):
    w: InputNet[Wire]
    r: OutputNet[Reg]
```

In contrast with `Bus` it provides two properties - `.internal` and `.external` of type `Bus`, which
represent internal and external port of the interface. The directions specified in the interface
members declarations corresponds to internal port, i.e. looking towards a component implementation.
External port is adjacent, and is looking towards the component external periphery.

It has the same creation and construction methods as `Bus`. Typically you want use `.Assign()`
method of `.internal` and `.external` buses.

```python
self.memIface.internal.Assign(valid=self.memValid,
                              insn=~self.insnFetched,
                              address=self.memAddress,
                              dataWrite=self.memWData,
                              writeMask=self.memWriteMask)
```

## Advanced example

For more advanced example see [RISC-V core example implementation](https://github.com/vagran/gateforge-riscv).


## License
Apache 2.0 - See [LICENSE](LICENSE) for details

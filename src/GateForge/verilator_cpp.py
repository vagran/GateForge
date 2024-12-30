from pathlib import Path
from typing import List


template = """
#include <verilated.h>
#include <V{moduleName}.h>

class Context {{
public:
    VerilatedContext ctx;
    std::unique_ptr<V{moduleName}> module;

    Context():
        module(new V{moduleName}(&ctx))
    {{

    }}

    ~Context()
    {{
        module->final();
    }}
}};

extern "C" {{

Context *
Construct();

void
Destruct(Context *ctx);

void
Eval(Context *ctx);

void
TimeInc(Context *ctx, uint64_t add);

{getters}

{setters}

}} /* extern "C" */

Context *
Construct()
{{
    return new Context();
}}

void
Destruct(Context *ctx)
{{
    delete ctx;
}}

void
Eval(Context *ctx)
{{
    ctx->module->eval();
}}

void
TimeInc(Context *ctx, uint64_t add)
{{
    ctx->ctx.timeInc(add);
}}

{gettersImpl}

{settersImpl}
"""


def _GenerateGetters(ports: List[str]) -> str:
    result = ""
    for portName in ports:
        result += """
uint64_t
GateForge_Get_{portName}(Context *ctx);
""".format(portName=portName)
    return result


def _GenerateSetters(ports: List[str]) -> str:
    result = ""
    for portName in ports:
        result += """
void
GateForge_Set_{portName}(Context *ctx, uint64_t value);
""".format(portName=portName)
    return result


def _GenerateGettersImpl(ports: List[str]) -> str:
    result = ""
    for portName in ports:
        result += """
uint64_t
GateForge_Get_{portName}(Context *ctx)
{{
    return ctx->module->{portName};
}}
""".format(portName=portName)
    return result


def _GenerateSettersImpl(ports: List[str]) -> str:
    result = ""
    for portName in ports:
        result += """
void
GateForge_Set_{portName}(Context *ctx, uint64_t value)
{{
    ctx->module->{portName} = value;
}}
""".format(portName=portName)
    return result


def CreateCppFile(path: Path, moduleName: str, ports: List[str]):
    with open(path, "w") as f:
        f.write(template.format(moduleName=moduleName, getters=_GenerateGetters(ports),
                                setters=_GenerateSetters(ports),
                                gettersImpl=_GenerateGettersImpl(ports),
                                settersImpl=_GenerateSettersImpl(ports)))

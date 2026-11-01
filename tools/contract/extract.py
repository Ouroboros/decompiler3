'''Contract items as each layer of the pipeline represents them'''

import contextlib
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from codegen import generate_typescript
from falcom.ed9.disasm import ED9Opcode
from falcom.ed9.hlil_converter import convert_falcom_mlil_to_hlil
from falcom.ed9.lifters.vm_lifter import ED9VMLifter
from falcom.ed9.mlil_converter import convert_falcom_llil_to_mlil
from falcom.ed9.parser.types_scp import RawInt, ScpParamType, ScpValue
from ir.hlil import (
    HighLevelILFunction, HLILBlock, HLILComment, HLILDoWhile, HLILFor, HLILIf, HLILStatement, HLILSwitch, HLILWhile,
)
from ir.llil import LowLevelILDebug, LowLevelILFunction
from ir.mlil import MediumLevelILFunction, MLILDebug

from .model import TAG_FLOAT, TAG_INT, TAG_RAW, TAG_STR, NO_CALLEE, CodeItems, FunctionEntry, ModuleTables, Value


DEBUG_LINE  = 'line'
DEBUG_LOG   = 'log'

# Python parameter types of the lifter (ScpParamFlags.get_python_type) back to the raw flags
PYTHON_TYPE_FLAGS = {
    'Value32':      ScpParamType.Value,
    'str':          ScpParamType.Offset,
    'Nullable32':   ScpParamType.NullableValue,
    'NullableStr':  ScpParamType.NullableOffset,
    'Pointer':      ScpParamType.Pointer,
}

SCP_VALUE_TAGS = {
    ScpValue.Type.Raw:      TAG_RAW,
    ScpValue.Type.Integer:  TAG_INT,
    ScpValue.Type.Float:    TAG_FLOAT,
    ScpValue.Type.String:   TAG_STR,
}

HLIL_DEBUG_COMMENT  = re.compile(r'^(line|log)\((-?\d+)\)$')
TS_DEBUG_COMMENT    = re.compile(r'//\s*(line|log)\((-?\d+)\)')


# ----------------------------------------------------------------- values --

def value_of_scp(value: ScpValue) -> Value:
    return (SCP_VALUE_TAGS[value.type], value.value)


def value_of_python(value: Any) -> Value:
    '''Value of a default as the IR keeps it (a plain Python value; RawInt marks Raw)'''
    if isinstance(value, RawInt):
        return (TAG_RAW, int(value))

    if isinstance(value, bool):
        return (TAG_INT, int(value))

    if isinstance(value, int):
        return (TAG_INT, value)

    if isinstance(value, float):
        return (TAG_FLOAT, value)

    return (TAG_STR, value)


# ----------------------------------------------------------------- parser --

def tables_from_parser(parser) -> ModuleTables:
    index_of = {func.name: index for index, func in enumerate(parser.functions)}
    functions: Dict[str, FunctionEntry] = {}
    for func in parser.functions:
        records = tuple(
            (index_of.get(info.func_name, NO_CALLEE), int(info.call_type),
             tuple((value_of_scp(arg.value), arg.type) for arg in info.args))
            for info in func.debug_info
        )
        functions[func.name] = FunctionEntry(
            name = func.name,
            name_hash = func.name_hash(),
            is_common = bool(func.is_common_func),
            param_flags = tuple(param.type.flags for param in func.params),
            defaults = tuple(value_of_scp(param.default_value) if param.default_value is not None else None for param in func.params),
            debug_records = records,
        )

    return ModuleTables(
        dword_14 = parser.header.dword_14,
        globals = tuple((var.name, int(var.type)) for var in parser.global_vars),
        function_order = tuple(func.name for func in parser.functions),
        functions = functions,
    )


def code_from_instructions(func) -> CodeItems:
    '''Reference code items: reachable instructions of the disassembled function'''
    items = CodeItems()
    pending, seen = [func.entry_block], set()
    while pending:
        block = pending.pop()
        if block.offset in seen:
            continue

        seen.add(block.offset)
        pending.extend(block.succs)
        for inst in block.instructions:
            # Synthetic fallthrough jumps added by the disassembler have no bytes
            if inst.size == 0:
                continue

            if inst.opcode == ED9Opcode.DEBUG_SET_LINENO:
                items.lines.add((inst.offset, int(inst.operands[0].value)))

            elif inst.opcode == ED9Opcode.DEBUG_LOG:
                items.logs.add((inst.offset, int(inst.operands[0].value)))

    return items


# ------------------------------------------------------------------ layers --

@dataclass
class LayerForms:
    '''One function in every layer of the pipeline'''
    llil: LowLevelILFunction
    mlil: MediumLevelILFunction
    hlil: HighLevelILFunction
    ts: str
    llil_code: CodeItems
    llil_entry: FunctionEntry


def run_pipeline(parser, func) -> LayerForms:
    '''Lift and decompile one function, extracting LLIL items before later stages can touch the LLIL'''
    with contextlib.redirect_stdout(io.StringIO()):
        llil = ED9VMLifter(parser = parser).lift_function(func)
        llil_code = code_from_llil(llil)
        llil_entry = entry_from_params(llil.name, llil.params, llil.is_common_func)
        mlil = convert_falcom_llil_to_mlil(llil, parser)
        hlil = convert_falcom_mlil_to_hlil(mlil, func)
        ts = generate_typescript(hlil)

    return LayerForms(llil, mlil, hlil, ts, llil_code, llil_entry)


def entry_from_params(name: str, params, is_common: bool) -> FunctionEntry:
    '''Function table items an LLIL or MLIL function keeps: IRParameter type names and default values'''
    flags = tuple(PYTHON_TYPE_FLAGS.get(param.type_name) for param in params)
    return FunctionEntry(
        name = name,
        is_common = bool(is_common),
        param_flags = None if None in flags else tuple(int(flag) for flag in flags),
        defaults = tuple(value_of_python(param.default_value) if param.default_value is not None else None for param in params),
    )


def entry_from_hlil(func: HighLevelILFunction) -> FunctionEntry:
    return FunctionEntry(
        name = func.name,
        is_common = bool(func.is_common_func),
        defaults = tuple(value_of_python(param.default_value) if param.default_value is not None else None for param in func.parameters),
    )


def _debug_items(items: CodeItems, kind: str, address: Optional[int], value: int):
    if kind == DEBUG_LINE:
        items.lines.add((address, value))

    elif kind == DEBUG_LOG:
        items.logs.add((address, value))


def code_from_llil(func: LowLevelILFunction) -> CodeItems:
    items = CodeItems()
    for block in func.basic_blocks:
        for inst in block.instructions:
            if isinstance(inst, LowLevelILDebug):
                _debug_items(items, inst.debug_type, inst.address, int(inst.value))

    return items


def code_from_mlil(func: MediumLevelILFunction) -> CodeItems:
    items = CodeItems()
    for block in func.basic_blocks:
        for inst in block.instructions:
            if isinstance(inst, MLILDebug) and inst.debug_type in (DEBUG_LINE, DEBUG_LOG):
                _debug_items(items, inst.debug_type, inst.address, int(inst.value))

    return items


def hlil_statements(block: HLILBlock) -> Iterator[HLILStatement]:
    for stmt in block.statements:
        yield stmt
        if isinstance(stmt, HLILBlock):
            yield from hlil_statements(stmt)

        elif isinstance(stmt, HLILIf):
            yield from hlil_statements(stmt.true_block)
            if stmt.false_block is not None:
                yield from hlil_statements(stmt.false_block)

        elif isinstance(stmt, (HLILWhile, HLILDoWhile, HLILFor)):
            yield from hlil_statements(stmt.body)

        elif isinstance(stmt, HLILSwitch):
            for case in stmt.cases:
                yield from hlil_statements(case.body)


def code_from_hlil(func: HighLevelILFunction) -> CodeItems:
    items = CodeItems()
    for stmt in hlil_statements(func.body):
        if isinstance(stmt, HLILComment):
            match = HLIL_DEBUG_COMMENT.match(stmt.text)
            if match:
                _debug_items(items, match.group(1), stmt.address, int(match.group(2)))

    return items


def code_from_ts(text: str) -> CodeItems:
    '''The TypeScript output keeps no bytecode addresses'''
    items = CodeItems()
    for match in TS_DEBUG_COMMENT.finditer(text):
        _debug_items(items, match.group(1), None, int(match.group(2)))

    return items

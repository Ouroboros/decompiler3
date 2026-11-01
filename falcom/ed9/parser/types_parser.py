from .types_scp import *
from .crc32 import hash_func_Name
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..disasm import BasicBlock

class GlobalVar:
    index       : int
    name        : str
    type        : ScpGlobalVar.Type
    name_offset : int

    def __init__(self, index: int, name: str, type: ScpGlobalVar.Type, name_offset: int = None):
        self.index = index
        self.name = name
        self.type = type
        self.name_offset = name_offset  # File offset of the name string

    def __str__(self) -> str:
        return f'global(index = {self.index}, name = {self.name!r}, type = {self.type})'

    __repr__ = __str__


class FunctionCallDebugInfo:
    class ArgInfo(StrictBase):
        value : ScpValue
        type  : int

        def __init__(self, type: int, value: ScpValue):
            self.type = type
            self.value = value

    func_name   : str
    func_id     : int       # Callee function index, NO_FUNCTION_ID for script and system calls
    call_type   : ScpFunctionCallDebugInfo.CallType
    args        : list[ArgInfo]
    info_offset : int       # File offset of the argument records

    NO_FUNCTION_ID = 0xFFFFFFFF

    def __init__(self):
        self.args = []

    def __str__(self) -> str:
        lines = [f'debug_info(func_name={self.func_name}, call_type={self.call_type})']
        if self.args:
            indent = default_indent()
            for arg in self.args:
                lines.append(f'{indent}arg(type={arg.type}, value={arg.value})')
        return '\n'.join(lines)

    __repr__ = __str__


class FunctionParam:
    type            : ScpParamFlags
    default_value   : ScpValue | None

    def __init__(self, type: ScpParamFlags, default_value: ScpValue | None = None):
        self.type           = type
        self.default_value  = default_value

    def __str__(self) -> str:
        if self.default_value is not None:
            return f'param({self.type.get_python_type()}, default = {self.default_value})'
        else:
            return f'param({self.type.get_python_type()})'

    __repr__ = __str__

class ScriptLayout:
    '''File-level layout: everything after the tables (code and the string pool) is kept as bytes'''

    def __init__(self, size: int, body_offset: int, body: bytes):
        self.size           = size
        self.body_offset    = body_offset
        self.body           = body
        # Start of the string pool after the code; set once the code is decoded
        self.string_pool_offset = size


class FunctionLayout:
    '''File offsets of a function entry's arrays and name, kept to write the script back unchanged'''

    def __init__(self, name_offset: int, defaults_offset: int, param_flags_offset: int, debug_info_offset: int):
        self.name_offset        = name_offset
        self.defaults_offset    = defaults_offset
        self.param_flags_offset = param_flags_offset
        self.debug_info_offset  = debug_info_offset


class Function:
    name            : str
    offset          : int
    params          : list[FunctionParam]
    is_common_func  : bool
    byte06          : int
    debug_info      : list[FunctionCallDebugInfo]
    layout          : FunctionLayout
    entry_block     : BasicBlock | None
    # Every instruction from the entry up to the next function, dead code included, in offset order
    instructions    : list
    code_end        : int

    def __init__(self):
        self.params     = []
        self.debug_info = []
        self.byte06     = 0
        self.layout     = None
        self.entry_block = None
        self.instructions = []
        self.code_end   = 0

    def name_hash(self) -> int:
        return hash_func_Name(self.name)

    def __str__(self) -> str:
        params = ', '.join([
            param.type.get_python_type() if param.default_value is None
                else f'{param.type.get_python_type()} = {param.default_value.value!r}'
            for param in self.params
        ])

        lines = [
            f'{self.name}({params})',
        ]

        indent = default_indent()

        # if self.debug_info:
        #     for dbg in self.debug_info:
        #         dbg_lines = str(dbg).splitlines()
        #         for dbg_line in dbg_lines:
        #             lines.append(f'{indent}{dbg_line}')

        #         lines.append('')

        return '\n'.join(lines)

    __repr__ = __str__

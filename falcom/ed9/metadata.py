'''ED9 file-level metadata of a script module (ir/core/target.py ModuleMetadata)'''

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ir.core import ModuleMetadata

from .parser.types_parser import FunctionCallDebugInfo


@dataclass(frozen = True)
class ED9GlobalVar:
    name: str
    type: int


@dataclass(frozen = True)
class ED9FunctionMetadata:
    '''Function table items the IL functions do not represent themselves'''
    index: int
    name_hash: int
    byte06: int
    # Call debug records exactly as stored, including those without arguments
    debug_info: Tuple[FunctionCallDebugInfo, ...]


@dataclass(frozen = True)
class ED9ModuleMetadata(ModuleMetadata):
    '''File-level items of a script; shared by the module of every layer'''
    dword_14: int
    globals: Tuple[ED9GlobalVar, ...]
    # Function names in table order (the function index of calls)
    function_order: Tuple[str, ...]
    functions: Dict[str, ED9FunctionMetadata] = field(default_factory = dict)

    @classmethod
    def fields(cls) -> List[str]:
        return [item.name for item in dataclasses.fields(cls)]

    @classmethod
    def from_parser(cls, parser) -> 'ED9ModuleMetadata':
        return cls(
            dword_14 = parser.header.dword_14,
            globals = tuple(ED9GlobalVar(var.name, int(var.type)) for var in parser.global_vars),
            function_order = tuple(func.name for func in parser.functions),
            functions = {
                func.name: ED9FunctionMetadata(index, func.name_hash(), func.byte06, tuple(func.debug_info))
                for index, func in enumerate(parser.functions)
            },
        )

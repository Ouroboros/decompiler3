'''ED9 script modules: file-level metadata and the module-by-module conversion pipeline'''

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ir.core import ILModule
from ir.hlil import HighLevelILFunction
from ir.llil import LowLevelILFunction
from ir.mlil import MediumLevelILFunction

from .hlil_converter import convert_falcom_mlil_to_hlil
from .lifters.vm_lifter import ED9VMLifter
from .mlil_converter import convert_falcom_llil_to_mlil
from .parser.types_parser import Function, FunctionCallDebugInfo


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
class ED9ModuleMetadata:
    '''File-level items of a script; shared by the module of every layer'''
    dword_14: int
    globals: Tuple[ED9GlobalVar, ...]
    # Function names in table order (the function index of calls)
    function_order: Tuple[str, ...]
    functions: Dict[str, ED9FunctionMetadata] = field(default_factory = dict)

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


class ED9ModuleDecompiler:
    '''Converts the functions of one parsed script layer by layer into LLIL, MLIL and HLIL modules

    The three modules share the script's ED9ModuleMetadata. decompile() keeps only the HLIL of a function;
    the single-layer steps keep what they produce.
    '''

    def __init__(self, parser):
        self.parser = parser
        self.lifter = ED9VMLifter(parser = parser)
        self.llil = ILModule(parser.name, ED9ModuleMetadata.from_parser(parser))
        self.mlil = self.llil.derive()
        self.hlil = self.llil.derive()

    def lift(self, func: Function) -> LowLevelILFunction:
        return self.llil.add_function(self.lifter.lift_function(func))

    def to_mlil(self, func: Function) -> MediumLevelILFunction:
        return self.mlil.add_function(convert_falcom_llil_to_mlil(self.llil[func.name], self.parser))

    def to_hlil(self, func: Function) -> HighLevelILFunction:
        return self.hlil.add_function(convert_falcom_mlil_to_hlil(self.mlil[func.name], func))

    def decompile(self, func: Function) -> HighLevelILFunction:
        self.lift(func)
        self.to_mlil(func)
        hlil = self.to_hlil(func)
        self.llil.remove_function(func.name)
        self.mlil.remove_function(func.name)
        return hlil

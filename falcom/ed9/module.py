'''ED9 script modules: the module-by-module conversion pipeline (metadata in falcom/ed9/metadata.py)'''

from typing import Optional, Tuple

from ir.core import ILModule
from ir.pipeline import PassObserver
from ir.hlil import HighLevelILFunction
from ir.llil import LowLevelILFunction
from ir.mlil import MediumLevelILFunction

from .hlil_converter import convert_falcom_mlil_to_hlil
from .lifters.vm_lifter import ED9VMLifter
from .mlil_converter import convert_falcom_llil_to_mlil
from .metadata import ED9FunctionMetadata, ED9GlobalVar, ED9ModuleMetadata
from .parser.types_parser import Function


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

    def to_mlil(self, func: Function, observer: Optional[PassObserver] = None) -> MediumLevelILFunction:
        '''observer sees the MLIL after every pass (ir/pipeline.py)'''
        return self.mlil.add_function(convert_falcom_llil_to_mlil(self.llil[func.name], self.parser, observer = observer))

    def to_hlil(self, func: Function, observer: Optional[PassObserver] = None) -> HighLevelILFunction:
        '''observer sees the HLIL after every pass (ir/pipeline.py)'''
        return self.hlil.add_function(convert_falcom_mlil_to_hlil(self.mlil[func.name], func, observer = observer))

    def decompile_layers(self, func: Function) -> Tuple[LowLevelILFunction, MediumLevelILFunction, HighLevelILFunction]:
        '''The function in every layer; the modules keep only the HLIL'''
        llil = self.lift(func)
        mlil = self.to_mlil(func)
        hlil = self.to_hlil(func)
        self.llil.remove_function(func.name)
        self.mlil.remove_function(func.name)
        return llil, mlil, hlil

    def decompile(self, func: Function) -> HighLevelILFunction:
        return self.decompile_layers(func)[-1]

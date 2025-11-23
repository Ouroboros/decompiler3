'''HLIL Passes'''

from ir.pipeline import Pass
from ir.mlil.mlil import MediumLevelILFunction
from .hlil import HighLevelILFunction
from .mlil_to_hlil import MLILToHLILConverter


class MLILToHLILPass(Pass):
    '''MLIL to HLIL conversion pass'''

    def run(self, mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
        '''Convert MLIL function to HLIL'''
        converter = MLILToHLILConverter(mlil_func)
        return converter.convert()

'''
HLIL Passes

HLIL transformation and optimization passes.
'''

from ir.pipeline import Pass
from ir.mlil.mlil import MediumLevelILFunction
from .hlil import HighLevelILFunction
from .mlil_to_hlil import MLILToHLILConverter


class MLILToHLILPass(Pass):
    '''MLIL to HLIL conversion pass'''

    def run(self, mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
        '''Convert MLIL function to HLIL

        Args:
            mlil_func: MLIL function

        Returns:
            HLIL function
        '''
        converter = MLILToHLILConverter(mlil_func)
        return converter.convert()

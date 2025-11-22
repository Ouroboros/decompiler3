'''
Falcom MLIL to HLIL Converter

Provides convenient function for converting MLIL to HLIL with Falcom-specific passes.
'''

from typing import Optional
from ir.mlil.mlil import MediumLevelILFunction
from ir.hlil import HighLevelILFunction
from ir.pipeline import Pipeline
from ir.hlil.hlil_passes import MLILToHLILPass
from .hlil_passes import FalcomTypeInferencePass
from .parser.types_parser import Function


def convert_falcom_mlil_to_hlil(mlil_func: MediumLevelILFunction, scp_func: Optional[Function] = None) -> HighLevelILFunction:
    '''Convert MLIL function to HLIL with Falcom-specific type information

    Uses a pipeline with the following passes:
    1. MLIL to HLIL conversion
    2. Falcom type inference (if scp_func provided)

    Args:
        mlil_func: MLIL function to convert
        scp_func: Optional SCP function with parameter type information

    Returns:
        Converted HLIL function with type annotations
    '''
    pipeline = Pipeline()
    pipeline.add_pass(MLILToHLILPass())

    if scp_func:
        pipeline.add_pass(FalcomTypeInferencePass(scp_func))

    return pipeline.run(mlil_func)

'''Falcom LLIL to MLIL Converter'''

from ir.llil import *
from ir.mlil import *
from ir.pipeline import *
from ir.mlil.mlil_passes import *
from .mlil_passes import *


def convert_falcom_llil_to_mlil(llil_func: LowLevelILFunction,
                                 parser=None,
                                 optimize: bool = True,
                                 infer_types: bool = True) -> MediumLevelILFunction:
    '''Convert LLIL function to MLIL with Falcom-specific handling

    Args:
        llil_func: LLIL function to convert
        parser: Optional ScpParser for extracting function signatures
        optimize: Whether to run SSA optimizations (default True)
        infer_types: Whether to run type inference (default True)
    '''
    print(f'Translating {llil_func.name} @ 0x{llil_func.start_addr:08X}')

    pipeline = Pipeline()

    # Phase 1: LLIL to MLIL conversion
    pipeline.add_pass(ED9LLILToMLILPass())

    # Phase 2: SSA-based optimization (optional)
    if optimize:
        pipeline.add_pass(SSAConversionPass())
        pipeline.add_pass(SSAOptimizationPass())

        # Type inference (on SSA form)
        if infer_types:
            pipeline.add_pass(ED9TypeInferencePass(parser))

        pipeline.add_pass(SSADeconstructionPass())
        pipeline.add_pass(DeadCodeEliminationPass())
        pipeline.add_pass(RegGlobalValuePropagationPass())

    return pipeline.run(llil_func)

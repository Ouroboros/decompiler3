'''HLIL Passes - Barrel file for all HLIL optimization passes'''

from .pass_mlil_to_hlil import MLILToHLILPass
from .pass_expression_simplification import ExpressionSimplificationPass
from .pass_control_flow_optimization import ControlFlowOptimizationPass
from .pass_common_return_extraction import CommonReturnExtractionPass
from .pass_copy_propagation import CopyPropagationPass
from .pass_dead_code_elimination import DeadCodeEliminationPass

__all__ = [
    'MLILToHLILPass',
    'ExpressionSimplificationPass',
    'ControlFlowOptimizationPass',
    'CommonReturnExtractionPass',
    'CopyPropagationPass',
    'DeadCodeEliminationPass',
]

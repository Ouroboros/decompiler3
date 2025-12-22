'''MLIL Passes - Pass-based MLIL processing'''

# Re-export all passes from passes/
from .passes import (
    # Lifting
    LLILToMLILPass,
    # SSA conversion
    SSAConversionPass,
    SSAOptimizationPass,
    SSADeconstructionPass,
    # SSA type inference
    TypeInferencePass,
    # Non-SSA
    DeadCodeEliminationPass,
    StorageKind,
    StorageKey,
    RegGlobalState,
    RegGlobalValuePropagator,
    RegGlobalValuePropagationPass,
)

__all__ = [
    'LLILToMLILPass',
    'SSAConversionPass',
    'SSAOptimizationPass',
    'SSADeconstructionPass',
    'TypeInferencePass',
    'DeadCodeEliminationPass',
    'StorageKind',
    'StorageKey',
    'RegGlobalState',
    'RegGlobalValuePropagator',
    'RegGlobalValuePropagationPass',
]

'''
Medium Level IL - Stack-free intermediate representation
'''

from .mlil_builder import MLILBuilder
from .llil_to_mlil import LLILToMLILTranslator, translate_llil_to_mlil
from .mlil_formatter import MLILFormatter, format_mlil_function
from .mlil_optimizer import MLILOptimizer, optimize_mlil

from .mlil import (
    # Naming utilities
    mlil_stack_var_name,
    mlil_arg_var_name,

    # Core
    MediumLevelILOperation,
    MediumLevelILInstruction,
    MediumLevelILFunction,
    MediumLevelILBasicBlock,

    # Variables
    MLILVariable,

    # Constants
    MLILConst,

    # Variable operations
    MLILVar,
    MLILSetVar,

    # Binary operations
    MLILBinaryOp,
    MLILAdd,
    MLILSub,
    MLILMul,
    MLILDiv,
    MLILMod,
    MLILAnd,
    MLILOr,
    MLILXor,
    MLILShl,
    MLILShr,
    MLILLogicalAnd,
    MLILLogicalOr,

    # Comparison operations
    MLILEq,
    MLILNe,
    MLILLt,
    MLILLe,
    MLILGt,
    MLILGe,

    # Unary operations
    MLILUnaryOp,
    MLILNeg,
    MLILLogicalNot,
    MLILTestZero,

    # Control flow
    MLILGoto,
    MLILIf,
    MLILRet,
    MLILRetVar,

    # Function calls
    MLILCall,
    MLILSyscall,
    MLILCallScript,

    # Globals
    MLILLoadGlobal,
    MLILStoreGlobal,

    # Registers
    MLILLoadReg,
    MLILStoreReg,

    # Debug
    MLILNop,
    MLILDebug,
)

__all__ = [
    # Naming utilities
    'mlil_stack_var_name',
    'mlil_arg_var_name',

    # Builder, Translator, Formatter, and Optimizer
    'MLILBuilder',
    'LLILToMLILTranslator',
    'translate_llil_to_mlil',
    'MLILFormatter',
    'format_mlil_function',
    'MLILOptimizer',
    'optimize_mlil',

    # Core
    'MediumLevelILOperation',
    'MediumLevelILInstruction',
    'MediumLevelILFunction',
    'MediumLevelILBasicBlock',

    # Variables
    'MLILVariable',

    # Constants
    'MLILConst',

    # Variable operations
    'MLILVar',
    'MLILSetVar',

    # Binary operations
    'MLILBinaryOp',
    'MLILAdd',
    'MLILSub',
    'MLILMul',
    'MLILDiv',
    'MLILMod',
    'MLILAnd',
    'MLILOr',
    'MLILXor',
    'MLILShl',
    'MLILShr',
    'MLILLogicalAnd',
    'MLILLogicalOr',

    # Comparison operations
    'MLILEq',
    'MLILNe',
    'MLILLt',
    'MLILLe',
    'MLILGt',
    'MLILGe',

    # Unary operations
    'MLILUnaryOp',
    'MLILNeg',
    'MLILLogicalNot',
    'MLILTestZero',

    # Control flow
    'MLILGoto',
    'MLILIf',
    'MLILRet',
    'MLILRetVar',

    # Function calls
    'MLILCall',
    'MLILSyscall',
    'MLILCallScript',

    # Globals
    'MLILLoadGlobal',
    'MLILStoreGlobal',

    # Registers
    'MLILLoadReg',
    'MLILStoreReg',

    # Debug
    'MLILNop',
    'MLILDebug',
]

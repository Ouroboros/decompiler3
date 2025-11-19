'''
Medium Level IL - Stack-free intermediate representation
'''

from .mlil_builder import MLILBuilder
from .llil_to_mlil import LLILToMLILTranslator, translate_llil_to_mlil
from .mlil_formatter import MLILFormatter, format_mlil_function
from .mlil_optimizer import MLILOptimizer, optimize_mlil
from .mlil_ssa_optimizer import SSAOptimizer
from .mlil_ssa import (
    MLILVariableSSA,
    MLILVarSSA,
    MLILSetVarSSA,
    MLILPhi,
    DominanceAnalysis,
    SSAConstructor,
    SSADeconstructor,
    convert_to_ssa,
    convert_from_ssa,
)
from .mlil_types import (
    MLILType,
    MLILTypeKind,
    MLILVariantType,
    unify_types,
    get_operation_result_type,
)
from .mlil_type_inference import MLILTypeInference, infer_types

from .mlil import (
    # Naming utilities
    mlil_stack_var_name,
    mlil_arg_var_name,

    # Core
    MediumLevelILOperation,
    MediumLevelILInstruction,
    MediumLevelILExpr,
    MediumLevelILStatement,
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

    # Builder, Translator, Formatter, and Optimizers
    'MLILBuilder',
    'LLILToMLILTranslator',
    'translate_llil_to_mlil',
    'MLILFormatter',
    'format_mlil_function',
    'MLILOptimizer',
    'SSAOptimizer',
    'optimize_mlil',

    # SSA
    'MLILVariableSSA',
    'MLILVarSSA',
    'MLILSetVarSSA',
    'MLILPhi',
    'DominanceAnalysis',
    'SSAConstructor',
    'SSADeconstructor',
    'convert_to_ssa',
    'convert_from_ssa',

    # Type System
    'MLILType',
    'MLILTypeKind',
    'MLILVariantType',
    'unify_types',
    'get_operation_result_type',
    'MLILTypeInference',
    'infer_types',

    # Core
    'MediumLevelILOperation',
    'MediumLevelILInstruction',
    'MediumLevelILExpr',
    'MediumLevelILStatement',
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

'''
MLIL Optimizer - SSA-based optimization

Performs SSA-based optimization passes on MLIL:
- Constant Propagation: Propagate constant values through SSA variables
- Copy Propagation: Eliminate copy assignments
- Dead Code Elimination: Remove unused SSA variable assignments
- Type Inference: Infer types for variables using SSA def-use chains
'''

from typing import Dict
from .mlil import *
from .mlil_ssa import convert_to_ssa, convert_from_ssa, MLILVariableSSA
from .mlil_ssa_optimizer import SSAOptimizer
from .mlil_type_inference import infer_types
from .mlil_types import MLILType, unify_types


def _map_ssa_types_to_base(ssa_var_types: Dict[MLILVariableSSA, MLILType],
                           function: MediumLevelILFunction) -> Dict[str, MLILType]:
    '''Map SSA variable types back to base variables

    Args:
        ssa_var_types: Mapping from SSA variables to types
        function: MLIL function (still in SSA form)

    Returns:
        Mapping from variable names to unified types

    Strategy:
        For each base variable, unify all its SSA versions' types
        Example: var_x#0: int, var_x#1: int → var_x: int
    '''
    base_types: Dict[str, MLILType] = {}

    for ssa_var, typ in ssa_var_types.items():
        base_name = ssa_var.base_var.name

        if base_name in base_types:
            # Unify with existing type
            base_types[base_name] = unify_types(base_types[base_name], typ)

        else:
            base_types[base_name] = typ

    return base_types


def _eliminate_dead_code_post_ssa(function: MediumLevelILFunction) -> bool:
    '''Remove unused variable assignments after de-SSA

    Args:
        function: MLIL function in non-SSA form

    Returns:
        True if any changes were made
    '''
    # Build use set for all variables
    var_uses: Dict[str, int] = {}

    # Count uses
    def count_uses(expr):
        if isinstance(expr, MLILVar):
            var_name = expr.var.name
            var_uses[var_name] = var_uses.get(var_name, 0) + 1

        elif isinstance(expr, MLILBinaryOp):
            count_uses(expr.lhs)
            count_uses(expr.rhs)

        elif isinstance(expr, MLILUnaryOp):
            count_uses(expr.operand)

        elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
            for arg in expr.args:
                count_uses(arg)

    # Scan all instructions to count uses
    for block in function.basic_blocks:
        for inst in block.instructions:
            if isinstance(inst, MLILSetVar):
                count_uses(inst.value)

            elif isinstance(inst, MLILIf):
                count_uses(inst.condition)

            elif isinstance(inst, MLILRet):
                if inst.value:
                    count_uses(inst.value)

            elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in inst.args:
                    count_uses(arg)

            elif isinstance(inst, MLILStoreGlobal):
                count_uses(inst.value)

            elif isinstance(inst, MLILStoreReg):
                count_uses(inst.value)

    # Remove unused assignments
    changed = False
    for block in function.basic_blocks:
        new_instructions = []
        for inst in block.instructions:
            # Keep non-assignment instructions
            if not isinstance(inst, MLILSetVar):
                new_instructions.append(inst)
                continue

            # For assignments, check if variable is used
            var_name = inst.var.name
            if var_uses.get(var_name, 0) > 0:
                new_instructions.append(inst)

            else:
                # Dead code
                changed = True

        block.instructions = new_instructions

    return changed


def optimize_mlil(function: MediumLevelILFunction, infer_types_enabled: bool = True) -> MediumLevelILFunction:
    '''Optimize MLIL function using SSA-based analysis

    Args:
        function: MLIL function to optimize
        infer_types_enabled: Whether to infer types (default: True)

    Returns:
        Optimized MLIL function

    Pipeline:
        MLIL → SSA → SSA-based optimizations → type inference → de-SSA → optimized MLIL
    '''
    # Convert to SSA form
    convert_to_ssa(function)

    # Run SSA-based optimizations (includes DCE on SSA variables)
    ssa_optimizer = SSAOptimizer(function)
    function = ssa_optimizer.optimize()

    # Infer types (before de-SSA, so we have def-use chains)
    if infer_types_enabled:
        ssa_var_types = infer_types(function)
        # Map SSA types back to base variables
        function.var_types = _map_ssa_types_to_base(ssa_var_types, function)

    # Convert back from SSA
    convert_from_ssa(function)

    # Final DCE pass: Clean up any remaining dead code
    # Catches: 1) Dead code that existed before SSA conversion
    #          2) Any residual copies from Phi elimination (rare, as we skip self-assignments)
    _eliminate_dead_code_post_ssa(function)

    return function

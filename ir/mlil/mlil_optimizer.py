'''
MLIL Optimizer - Dead Code Elimination and Variable Propagation

Performs optimization passes on MLIL to remove redundant operations:
- Dead Code Elimination: Remove unused variable assignments
- Copy Propagation: Inline single-use variables
'''

from typing import Dict, Set
from .mlil import *


class MLILOptimizer:
    '''MLIL optimizer for dead code elimination and copy propagation'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.var_uses: Dict[MLILVariable, int] = {}  # Variable -> use count
        self.var_defs: Dict[MLILVariable, MLILSetVar] = {}  # Variable -> last definition

    def optimize(self) -> MediumLevelILFunction:
        '''Run all optimization passes'''
        # Multiple passes until no changes
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            changed |= self.eliminate_dead_code()
            changed |= self.propagate_copies()
            iterations += 1

        return self.function

    def eliminate_dead_code(self) -> bool:
        '''Remove assignments to variables that are never read

        Returns:
            True if any code was eliminated
        '''
        changed = False

        # First pass: count variable uses
        self._analyze_variable_uses()

        # Second pass: remove dead assignments
        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                # Keep non-SetVar instructions
                if not isinstance(inst, MLILSetVar):
                    new_instructions.append(inst)
                    continue

                # Keep if variable is used
                if self.var_uses.get(inst.var, 0) > 0:
                    new_instructions.append(inst)
                else:
                    # Dead code - variable assigned but never read
                    changed = True

            block.instructions = new_instructions

        return changed

    def propagate_copies(self) -> bool:
        '''Inline variables that are only used once

        Returns:
            True if any propagation occurred
        '''
        changed = False

        # Analyze variable uses and definitions
        self._analyze_variable_uses()
        self._analyze_variable_defs()

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                # Try to inline variables in this instruction
                modified_inst = self._inline_single_use_vars(inst)
                if modified_inst is not inst:
                    changed = True

                new_instructions.append(modified_inst)

            block.instructions = new_instructions

        return changed

    def _analyze_variable_uses(self):
        '''Count how many times each variable is read'''
        self.var_uses = {}

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                self._count_uses_in_expr(inst)

    def _count_uses_in_expr(self, expr: MediumLevelILInstruction):
        '''Recursively count variable uses in expression'''
        if isinstance(expr, MLILVar):
            self.var_uses[expr.var] = self.var_uses.get(expr.var, 0) + 1

        # Recurse into sub-expressions
        elif isinstance(expr, MLILSetVar):
            self._count_uses_in_expr(expr.value)

        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            self._count_uses_in_expr(expr.lhs)
            self._count_uses_in_expr(expr.rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILTestZero)):
            self._count_uses_in_expr(expr.operand)

        elif isinstance(expr, MLILIf):
            self._count_uses_in_expr(expr.condition)

        elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
            for arg in expr.args:
                self._count_uses_in_expr(arg)

        elif isinstance(expr, (MLILStoreGlobal, MLILStoreReg)):
            self._count_uses_in_expr(expr.value)

    def _analyze_variable_defs(self):
        '''Track the last definition of each variable'''
        self.var_defs = {}

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar):
                    self.var_defs[inst.var] = inst

    def _inline_single_use_vars(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Inline variables that are only used once in this instruction'''

        if isinstance(inst, MLILSetVar):
            # Inline in the value expression
            new_value = self._inline_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVar(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._inline_expr(inst.condition)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._inline_expr(arg) for arg in inst.args]
            if any(new_args[i] is not inst.args[i] for i in range(len(inst.args))):
                if isinstance(inst, MLILCall):
                    return MLILCall(inst.target, new_args)
                elif isinstance(inst, MLILSyscall):
                    return MLILSyscall(inst.subsystem, inst.cmd, new_args)
                elif isinstance(inst, MLILCallScript):
                    return MLILCallScript(inst.module, inst.func, new_args)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._inline_expr(inst.value)
            if new_value is not inst.value:
                if isinstance(inst, MLILStoreGlobal):
                    return MLILStoreGlobal(inst.index, new_value)
                else:
                    return MLILStoreReg(inst.reg_index, new_value)

        return inst

    def _inline_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Inline single-use variables in expression'''

        # If this is a variable reference
        if isinstance(expr, MLILVar):
            # Check if it's only used once and has a simple definition
            if self.var_uses.get(expr.var, 0) == 1:
                definition = self.var_defs.get(expr.var)
                if definition and self._is_inlinable(definition.value):
                    # Inline the definition
                    return definition.value

        # Recurse into binary operations
        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            new_lhs = self._inline_expr(expr.lhs)
            new_rhs = self._inline_expr(expr.rhs)
            if new_lhs is not expr.lhs or new_rhs is not expr.rhs:
                # Create new instance with inlined operands
                return type(expr)(new_lhs, new_rhs)

        # Recurse into unary operations
        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILTestZero)):
            new_operand = self._inline_expr(expr.operand)
            if new_operand is not expr.operand:
                return type(expr)(new_operand)

        return expr

    def _is_inlinable(self, expr: MediumLevelILInstruction) -> bool:
        '''Check if expression is safe to inline'''
        # Don't inline complex expressions or function calls
        if isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
            return False

        # Don't inline loads (may have side effects)
        if isinstance(expr, (MLILLoadGlobal, MLILLoadReg)):
            return False

        # Constants and simple operations are safe
        return True


def optimize_mlil(function: MediumLevelILFunction) -> MediumLevelILFunction:
    '''Convenience function to optimize MLIL function'''
    optimizer = MLILOptimizer(function)
    return optimizer.optimize()

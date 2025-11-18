'''
MLIL Optimizer - Expression simplification and optimization

Performs optimization passes on MLIL:
- Constant Folding: Evaluate constant expressions at compile time
- Expression Simplification: Apply algebraic identities (x+0=x, x*1=x, etc.)
- Condition Simplification: Simplify boolean expressions
- Copy Propagation: Inline single-use variables
- Dead Code Elimination: Remove unused variable assignments
'''

from typing import Dict, Set, Optional
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
            # changed |= self.fold_constants()
            changed |= self.simplify_expressions()
            changed |= self.simplify_conditions()
            changed |= self.propagate_copies()
            changed |= self.eliminate_dead_code()
            iterations += 1

        return self.function

    def simplify_conditions(self) -> bool:
        '''Simplify condition expressions in If instructions

        Transforms patterns like ((a >= b) == 0) to (a < b)

        Returns:
            True if any simplification occurred
        '''
        changed = False

        for block in self.function.basic_blocks:
            for i, inst in enumerate(block.instructions):
                if not isinstance(inst, MLILIf):
                    continue

                # Check if condition needs simplification
                condition = inst.condition
                new_condition = None

                # Pattern: !(comparison) -> invert comparison
                if isinstance(condition, MLILLogicalNot):
                    inverted = self._invert_comparison(condition.operand)
                    if inverted is not None:
                        new_condition = inverted

                # Pattern: (expr == 0)
                elif isinstance(condition, MLILEq):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        # Try to invert comparison
                        inverted = self._invert_comparison(condition.lhs)
                        if inverted is not None:
                            # Simple comparison - invert it
                            # (a >= b) == 0 is equivalent to a < b
                            new_condition = inverted

                        else:
                            # Complex expression - use LogicalNot
                            # (expr) == 0 is equivalent to !expr
                            new_condition = MLILLogicalNot(condition.lhs)

                # Pattern: (expr != 0)
                elif isinstance(condition, MLILNe):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        # For (expr != 0), just use expr directly if it's a comparison
                        if isinstance(condition.lhs, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
                            new_condition = condition.lhs

                # Apply simplification
                if new_condition is not None:
                    # Simplified condition has same semantics, keep targets unchanged
                    block.instructions[i] = MLILIf(new_condition, inst.true_target, inst.false_target)
                    changed = True

        return changed

    def _invert_comparison(self, expr: MediumLevelILInstruction) -> Optional[MediumLevelILInstruction]:
        '''Invert a comparison expression

        Examples:
            a >= b  ->  a < b
            a > b   ->  a <= b
            a < b   ->  a >= b
            a <= b  ->  a > b
            a == b  ->  a != b
            a != b  ->  a == b

        Returns:
            Inverted expression, or None if not a comparison
        '''
        if isinstance(expr, MLILGe):
            return MLILLt(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILGt):
            return MLILLe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILLt):
            return MLILGe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILLe):
            return MLILGt(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILEq):
            return MLILNe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILNe):
            return MLILEq(expr.lhs, expr.rhs)

        else:
            return None

    def fold_constants(self) -> bool:
        '''Fold constant expressions at compile time

        Returns:
            True if any folding occurred
        '''
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._fold_constants_in_inst(inst)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _fold_constants_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Fold constants in a single instruction'''
        if isinstance(inst, MLILSetVar):
            new_value = self._fold_expr(inst.value)

            if new_value is not inst.value:
                return MLILSetVar(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._fold_expr(inst.condition)

            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._fold_expr(inst.value)

                if new_value is not inst.value:
                    return MLILRet(new_value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._fold_expr(arg) for arg in inst.args]

            if any(new_args[i] is not inst.args[i] for i in range(len(inst.args))):
                if isinstance(inst, MLILCall):
                    return MLILCall(inst.target, new_args)

                elif isinstance(inst, MLILSyscall):
                    return MLILSyscall(inst.subsystem, inst.cmd, new_args)

                elif isinstance(inst, MLILCallScript):
                    return MLILCallScript(inst.module, inst.func, new_args)

        elif isinstance(inst, MLILStoreGlobal):
            new_value = self._fold_expr(inst.value)

            if new_value is not inst.value:
                return MLILStoreGlobal(inst.index, new_value)

        elif isinstance(inst, MLILStoreReg):
            new_value = self._fold_expr(inst.value)

            if new_value is not inst.value:
                return MLILStoreReg(inst.index, new_value)

        return inst

    def _fold_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively fold constants in expression'''
        # First fold children
        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                             MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                             MLILLogicalAnd, MLILLogicalOr,
                             MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._fold_expr(expr.lhs)
            rhs = self._fold_expr(expr.rhs)

            # Try to evaluate if both are constants
            if isinstance(lhs, MLILConst) and isinstance(rhs, MLILConst):
                result = self._eval_binary_const(type(expr), lhs.value, rhs.value)

                if result is not None:
                    return MLILConst(result, is_hex = False)

            # Rebuild if children changed
            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot)):
            operand = self._fold_expr(expr.operand)

            # Try to evaluate if operand is constant
            if isinstance(operand, MLILConst):
                result = self._eval_unary_const(type(expr), operand.value)

                if result is not None:
                    return MLILConst(result, is_hex = False)

            # Rebuild if operand changed
            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def _eval_binary_const(self, op_type, lhs: int, rhs: int) -> Optional[int]:
        '''Evaluate binary operation on constants'''
        try:
            if op_type == MLILAdd:
                return lhs + rhs

            elif op_type == MLILSub:
                return lhs - rhs

            elif op_type == MLILMul:
                return lhs * rhs

            elif op_type == MLILDiv:
                return lhs // rhs if rhs != 0 else None

            elif op_type == MLILMod:
                return lhs % rhs if rhs != 0 else None

            elif op_type == MLILAnd:
                return lhs & rhs

            elif op_type == MLILOr:
                return lhs | rhs

            elif op_type == MLILXor:
                return lhs ^ rhs

            elif op_type == MLILShl:
                return lhs << rhs

            elif op_type == MLILShr:
                return lhs >> rhs

            elif op_type == MLILLogicalAnd:
                return 1 if (lhs and rhs) else 0

            elif op_type == MLILLogicalOr:
                return 1 if (lhs or rhs) else 0

            elif op_type == MLILEq:
                return 1 if lhs == rhs else 0

            elif op_type == MLILNe:
                return 1 if lhs != rhs else 0

            elif op_type == MLILLt:
                return 1 if lhs < rhs else 0

            elif op_type == MLILLe:
                return 1 if lhs <= rhs else 0

            elif op_type == MLILGt:
                return 1 if lhs > rhs else 0

            elif op_type == MLILGe:
                return 1 if lhs >= rhs else 0

        except (OverflowError, ZeroDivisionError):
            return None

        return None

    def _eval_unary_const(self, op_type, operand: int) -> Optional[int]:
        '''Evaluate unary operation on constant'''
        try:
            if op_type == MLILNeg:
                return -operand

            elif op_type == MLILLogicalNot:
                return 1 if not operand else 0

        except OverflowError:
            return None

        return None

    def simplify_expressions(self) -> bool:
        '''Apply algebraic simplification rules

        Returns:
            True if any simplification occurred
        '''
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._simplify_expr_in_inst(inst)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _simplify_expr_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Simplify expressions in a single instruction'''
        if isinstance(inst, MLILSetVar):
            new_value = self._simplify_expr(inst.value)

            if new_value is not inst.value:
                return MLILSetVar(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._simplify_expr(inst.condition)

            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._simplify_expr(inst.value)

                if new_value is not inst.value:
                    return MLILRet(new_value)

        return inst

    def _simplify_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively apply algebraic simplifications'''
        # First simplify children
        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                             MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr)):
            lhs = self._simplify_expr(expr.lhs)
            rhs = self._simplify_expr(expr.rhs)

            # Apply algebraic identities
            simplified = self._apply_algebraic_identity(type(expr), lhs, rhs)

            if simplified is not None:
                return simplified

            # Rebuild if children changed
            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot)):
            operand = self._simplify_expr(expr.operand)

            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def _apply_algebraic_identity(self, op_type, lhs, rhs) -> Optional[MediumLevelILInstruction]:
        '''Apply algebraic identity rules'''
        # x + 0 = x
        if op_type == MLILAdd:
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs

            if isinstance(lhs, MLILConst) and lhs.value == 0:
                return rhs

        # x - 0 = x
        elif op_type == MLILSub:
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs

        # x * 0 = 0, x * 1 = x
        elif op_type == MLILMul:
            if isinstance(rhs, MLILConst):
                if rhs.value == 0:
                    return MLILConst(0, is_hex = False)

                elif rhs.value == 1:
                    return lhs

            if isinstance(lhs, MLILConst):
                if lhs.value == 0:
                    return MLILConst(0, is_hex = False)

                elif lhs.value == 1:
                    return rhs

        # x / 1 = x
        elif op_type == MLILDiv:
            if isinstance(rhs, MLILConst) and rhs.value == 1:
                return lhs

        # x & 0 = 0, x & 0xFFFFFFFF = x
        elif op_type == MLILAnd:
            if isinstance(rhs, MLILConst):
                if rhs.value == 0:
                    return MLILConst(0, is_hex = False)

                elif rhs.value == 0xFFFFFFFF:
                    return lhs

            if isinstance(lhs, MLILConst):
                if lhs.value == 0:
                    return MLILConst(0, is_hex = False)

                elif lhs.value == 0xFFFFFFFF:
                    return rhs

        # x | 0 = x, x | 0xFFFFFFFF = 0xFFFFFFFF
        elif op_type == MLILOr:
            if isinstance(rhs, MLILConst):
                if rhs.value == 0:
                    return lhs

                elif rhs.value == 0xFFFFFFFF:
                    return MLILConst(0xFFFFFFFF, is_hex = True)

            if isinstance(lhs, MLILConst):
                if lhs.value == 0:
                    return rhs

                elif lhs.value == 0xFFFFFFFF:
                    return MLILConst(0xFFFFFFFF, is_hex = True)

        # x ^ 0 = x
        elif op_type == MLILXor:
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs

            if isinstance(lhs, MLILConst) and lhs.value == 0:
                return rhs

        # x << 0 = x, x >> 0 = x
        elif op_type in (MLILShl, MLILShr):
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs

        return None

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

        elif isinstance(expr, MLILRet):
            if expr.value is not None:
                self._count_uses_in_expr(expr.value)

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
                    return MLILStoreReg(inst.index, new_value)

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
                # Create new instance with inlined operands (explicit type checking)
                return self._reconstruct_binary_op(expr, new_lhs, new_rhs)

        # Recurse into unary operations
        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILTestZero, MLILAddressOf)):
            new_operand = self._inline_expr(expr.operand)
            if new_operand is not expr.operand:
                return self._reconstruct_unary_op(expr, new_operand)

        return expr

    def _reconstruct_binary_op(self, expr: MediumLevelILInstruction,
                               lhs: MediumLevelILInstruction,
                               rhs: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Reconstruct binary operation with new operands (explicit type checking)'''
        # Arithmetic operations
        if isinstance(expr, MLILAdd):
            return MLILAdd(lhs, rhs)

        elif isinstance(expr, MLILSub):
            return MLILSub(lhs, rhs)

        elif isinstance(expr, MLILMul):
            return MLILMul(lhs, rhs)

        elif isinstance(expr, MLILDiv):
            return MLILDiv(lhs, rhs)

        elif isinstance(expr, MLILMod):
            return MLILMod(lhs, rhs)

        # Bitwise operations
        elif isinstance(expr, MLILAnd):
            return MLILAnd(lhs, rhs)

        elif isinstance(expr, MLILOr):
            return MLILOr(lhs, rhs)

        elif isinstance(expr, MLILXor):
            return MLILXor(lhs, rhs)

        elif isinstance(expr, MLILShl):
            return MLILShl(lhs, rhs)

        elif isinstance(expr, MLILShr):
            return MLILShr(lhs, rhs)

        # Logical operations
        elif isinstance(expr, MLILLogicalAnd):
            return MLILLogicalAnd(lhs, rhs)

        elif isinstance(expr, MLILLogicalOr):
            return MLILLogicalOr(lhs, rhs)

        # Comparison operations
        elif isinstance(expr, MLILEq):
            return MLILEq(lhs, rhs)

        elif isinstance(expr, MLILNe):
            return MLILNe(lhs, rhs)

        elif isinstance(expr, MLILLt):
            return MLILLt(lhs, rhs)

        elif isinstance(expr, MLILLe):
            return MLILLe(lhs, rhs)

        elif isinstance(expr, MLILGt):
            return MLILGt(lhs, rhs)

        elif isinstance(expr, MLILGe):
            return MLILGe(lhs, rhs)

        else:
            raise NotImplementedError(f'Unhandled binary operation type: {type(expr).__name__}')

    def _reconstruct_unary_op(self, expr: MediumLevelILInstruction,
                              operand: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Reconstruct unary operation with new operand (explicit type checking)'''
        if isinstance(expr, MLILNeg):
            return MLILNeg(operand)

        elif isinstance(expr, MLILLogicalNot):
            return MLILLogicalNot(operand)

        elif isinstance(expr, MLILTestZero):
            return MLILTestZero(operand)

        elif isinstance(expr, MLILAddressOf):
            return MLILAddressOf(operand)

        else:
            raise NotImplementedError(f'Unhandled unary operation type: {type(expr).__name__}')

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

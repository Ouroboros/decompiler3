'''
MLIL SSA Optimizer - SSA-based optimizations using def-use chains

Performs SSA-based optimization passes:
- Constant Propagation: Propagate constant values through SSA variables
- Copy Propagation: Eliminate copy assignments (x#1 = x#0)
- Dead Code Elimination: Remove unused SSA variable assignments
'''

from typing import Dict, List, Optional
from .mlil import *
from .mlil_ssa import *


class SSAOptimizer:
    '''SSA-based optimizer using def-use chains'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.ssa_defs: Dict[MLILVariableSSA, MLILSetVarSSA] = {}  # SSA var -> definition
        self.ssa_uses: Dict[MLILVariableSSA, List[MediumLevelILInstruction]] = {}  # SSA var -> uses
        self.constants: Dict[MLILVariableSSA, int] = {}  # SSA var -> constant value

    def optimize(self) -> MediumLevelILFunction:
        '''Run all SSA-based optimization passes'''
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            changed |= self.propagate_constants()
            # changed |= self.fold_constants()     # Constant folding - temporarily disabled
            changed |= self.simplify_expressions()
            changed |= self.simplify_conditions()
            changed |= self.propagate_copies()
            changed |= self.eliminate_dead_code()
            iterations += 1

        return self.function

    def _build_def_use_chains(self):
        '''Build SSA def-use chains'''
        self.ssa_defs = {}
        self.ssa_uses = {}
        self.constants = {}

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                # Track definitions
                if isinstance(inst, MLILSetVarSSA):
                    self.ssa_defs[inst.var] = inst

                    # Track constants
                    if isinstance(inst.value, MLILConst):
                        self.constants[inst.var] = inst.value.value

                elif isinstance(inst, MLILPhi):
                    self.ssa_defs[inst.dest] = inst

                # Track uses
                self._collect_uses_in_inst(inst)

    def _collect_uses_in_inst(self, inst: MediumLevelILInstruction):
        '''Recursively collect SSA variable uses'''
        if isinstance(inst, MLILVarSSA):
            if inst.var not in self.ssa_uses:
                self.ssa_uses[inst.var] = []
            self.ssa_uses[inst.var].append(inst)

        # Recurse into sub-expressions
        elif isinstance(inst, MLILSetVarSSA):
            self._collect_uses_in_expr(inst.value)

        elif isinstance(inst, MLILPhi):
            for source_var, _ in inst.sources:
                if source_var not in self.ssa_uses:
                    self.ssa_uses[source_var] = []
                self.ssa_uses[source_var].append(inst)

        elif isinstance(inst, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            self._collect_uses_in_expr(inst.lhs)
            self._collect_uses_in_expr(inst.rhs)

        elif isinstance(inst, (MLILNeg, MLILLogicalNot, MLILTestZero)):
            self._collect_uses_in_expr(inst.operand)

        elif isinstance(inst, MLILIf):
            self._collect_uses_in_expr(inst.condition)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                self._collect_uses_in_expr(inst.value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            for arg in inst.args:
                self._collect_uses_in_expr(arg)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            self._collect_uses_in_expr(inst.value)

    def _collect_uses_in_expr(self, expr: MediumLevelILInstruction):
        '''Helper to collect uses in expression'''
        self._collect_uses_in_inst(expr)

    def propagate_constants(self) -> bool:
        '''Propagate constant values through SSA variables

        If x#0 = 5, replace all uses of x#0 with 5
        Does NOT fold constants (5 + 3 stays as 5 + 3)
        '''
        self._build_def_use_chains()
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._propagate_constants_in_inst(inst)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _propagate_constants_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Propagate constants in a single instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._propagate_constants_in_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._propagate_constants_in_expr(inst.condition)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._propagate_constants_in_expr(inst.value)
                if new_value is not inst.value:
                    return MLILRet(new_value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._propagate_constants_in_expr(arg) for arg in inst.args]
            if any(new_args[i] is not inst.args[i] for i in range(len(inst.args))):
                if isinstance(inst, MLILCall):
                    return MLILCall(inst.target, new_args)
                elif isinstance(inst, MLILSyscall):
                    return MLILSyscall(inst.subsystem, inst.cmd, new_args)
                elif isinstance(inst, MLILCallScript):
                    return MLILCallScript(inst.module, inst.func, new_args)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._propagate_constants_in_expr(inst.value)
            if new_value is not inst.value:
                if isinstance(inst, MLILStoreGlobal):
                    return MLILStoreGlobal(inst.index, new_value)
                else:
                    return MLILStoreReg(inst.index, new_value)

        return inst

    def _propagate_constants_in_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Replace SSA variables with their constant values (NO folding)'''
        # Replace SSA variable with constant
        if isinstance(expr, MLILVarSSA):
            if expr.var in self.constants:
                return MLILConst(self.constants[expr.var], is_hex = False)

        # Recurse into binary operations
        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._propagate_constants_in_expr(expr.lhs)
            rhs = self._propagate_constants_in_expr(expr.rhs)

            # Rebuild if children changed (NO folding)
            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        # Recurse into unary operations
        elif isinstance(expr, (MLILNeg, MLILLogicalNot)):
            operand = self._propagate_constants_in_expr(expr.operand)

            # Rebuild if operand changed (NO folding)
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

    def fold_constants(self) -> bool:
        '''Fold constant expressions at compile time

        Evaluates expressions like (5 + 3) to 8
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
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._fold_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

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

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._fold_expr(inst.value)
            if new_value is not inst.value:
                if isinstance(inst, MLILStoreGlobal):
                    return MLILStoreGlobal(inst.index, new_value)
                else:
                    return MLILStoreReg(inst.index, new_value)

        return inst

    def _fold_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively fold constant expressions'''
        # Recurse into binary operations
        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                             MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                             MLILLogicalAnd, MLILLogicalOr,
                             MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._fold_expr(expr.lhs)
            rhs = self._fold_expr(expr.rhs)

            # Fold if both sides are constants
            if isinstance(lhs, MLILConst) and isinstance(rhs, MLILConst):
                result = self._eval_binary_const(type(expr), lhs.value, rhs.value)
                if result is not None:
                    return MLILConst(result, is_hex = False)

            # Rebuild if children changed
            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        # Recurse into unary operations
        elif isinstance(expr, (MLILNeg, MLILLogicalNot)):
            operand = self._fold_expr(expr.operand)

            # Fold if operand is constant
            if isinstance(operand, MLILConst):
                result = self._eval_unary_const(type(expr), operand.value)
                if result is not None:
                    return MLILConst(result, is_hex = False)

            # Rebuild if operand changed
            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def propagate_copies(self) -> bool:
        '''Propagate SSA variable copies

        If x#1 = x#0, replace all uses of x#1 with x#0
        '''
        self._build_def_use_chains()
        changed = False

        # Find copy assignments
        copies: Dict[MLILVariableSSA, MLILVariableSSA] = {}
        for ssa_var, defn in self.ssa_defs.items():
            if isinstance(defn, MLILSetVarSSA):
                if isinstance(defn.value, MLILVarSSA):
                    copies[ssa_var] = defn.value.var

        # Replace uses
        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._replace_copies_in_inst(inst, copies)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _replace_copies_in_inst(self, inst: MediumLevelILInstruction,
                                copies: Dict[MLILVariableSSA, MLILVariableSSA]) -> MediumLevelILInstruction:
        '''Replace copy variables in instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._replace_copies_in_expr(inst.value, copies)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._replace_copies_in_expr(inst.condition, copies)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._replace_copies_in_expr(inst.value, copies)
                if new_value is not inst.value:
                    return MLILRet(new_value)

        return inst

    def _replace_copies_in_expr(self, expr: MediumLevelILInstruction,
                                copies: Dict[MLILVariableSSA, MLILVariableSSA]) -> MediumLevelILInstruction:
        '''Replace copy variables in expression'''
        if isinstance(expr, MLILVarSSA):
            if expr.var in copies:
                return MLILVarSSA(copies[expr.var])

        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._replace_copies_in_expr(expr.lhs, copies)
            rhs = self._replace_copies_in_expr(expr.rhs, copies)
            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot)):
            operand = self._replace_copies_in_expr(expr.operand, copies)
            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def eliminate_dead_code(self) -> bool:
        '''Remove SSA variable assignments that are never used'''
        self._build_def_use_chains()
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                # Keep non-assignment instructions
                if not isinstance(inst, (MLILSetVarSSA, MLILPhi)):
                    new_instructions.append(inst)
                    continue

                # For assignments, check if the variable is used
                if isinstance(inst, MLILSetVarSSA):
                    if len(self.ssa_uses.get(inst.var, [])) > 0:
                        new_instructions.append(inst)
                    else:
                        # Dead code - variable never used
                        changed = True

                elif isinstance(inst, MLILPhi):
                    if len(self.ssa_uses.get(inst.dest, [])) > 0:
                        new_instructions.append(inst)
                    else:
                        # Dead phi node
                        changed = True

            block.instructions = new_instructions

        return changed

    def _reconstruct_binary_op(self, expr: MediumLevelILInstruction,
                               lhs: MediumLevelILInstruction,
                               rhs: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Reconstruct binary operation with new operands'''
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
        elif isinstance(expr, MLILLogicalAnd):
            return MLILLogicalAnd(lhs, rhs)
        elif isinstance(expr, MLILLogicalOr):
            return MLILLogicalOr(lhs, rhs)
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
            raise NotImplementedError(f'Unhandled binary operation: {type(expr).__name__}')

    def _reconstruct_unary_op(self, expr: MediumLevelILInstruction,
                              operand: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Reconstruct unary operation with new operand'''
        if isinstance(expr, MLILNeg):
            return MLILNeg(operand)
        elif isinstance(expr, MLILLogicalNot):
            return MLILLogicalNot(operand)
        else:
            raise NotImplementedError(f'Unhandled unary operation: {type(expr).__name__}')

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
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._simplify_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

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
                            new_condition = inverted
                        else:
                            # Complex expression - use LogicalNot
                            new_condition = MLILLogicalNot(condition.lhs)

                # Pattern: (expr != 0)
                elif isinstance(condition, MLILNe):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        # For (expr != 0), just use expr directly if it's a comparison
                        if isinstance(condition.lhs, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
                            new_condition = condition.lhs

                # Apply simplification
                if new_condition is not None:
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

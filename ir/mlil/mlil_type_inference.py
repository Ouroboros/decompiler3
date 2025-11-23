'''MLIL Type Inference - SSA-based type propagation'''

from typing import Dict, Set, List
from .mlil import *
from .mlil_ssa import MLILVariableSSA, MLILVarSSA, MLILSetVarSSA, MLILPhi
from .mlil_types import MLILType, unify_types, get_operation_result_type


class MLILTypeInference:
    '''Type inference for MLIL using SSA form'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.var_types: Dict[MLILVariableSSA, MLILType] = {}

    def infer_types(self) -> Dict[MLILVariableSSA, MLILType]:
        '''Infer types for all SSA variables'''
        # Initialize all variables to unknown
        self._initialize_types()

        # Iterate until types converge
        changed = True
        iterations = 0
        max_iterations = 20

        while changed and iterations < max_iterations:
            changed = self._propagate_types()
            iterations += 1

        # Backward pass: infer types from usage (e.g., comparisons, arithmetic)
        for _ in range(3):
            if self._infer_from_usage():
                changed = True

        return self.var_types

    def _initialize_types(self):
        '''Initialize all SSA variables to unknown type'''
        self.var_types = {}

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVarSSA):
                    self.var_types[inst.var] = MLILType.unknown()
                elif isinstance(inst, MLILPhi):
                    self.var_types[inst.dest] = MLILType.unknown()

    def _propagate_types(self) -> bool:
        '''Propagate types through SSA variables

        Returns:
            True if any type changed
        '''
        changed = False

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVarSSA):
                    new_type = self._infer_expr_type(inst.value)
                    if self._update_var_type(inst.var, new_type):
                        changed = True

                elif isinstance(inst, MLILPhi):
                    new_type = self._infer_phi_type(inst)
                    if self._update_var_type(inst.dest, new_type):
                        changed = True

        return changed

    def _update_var_type(self, var: MLILVariableSSA, new_type: MLILType) -> bool:
        '''Update variable type, unifying with existing type

        Returns:
            True if type changed
        '''
        old_type = self.var_types.get(var, MLILType.unknown())
        unified_type = unify_types(old_type, new_type)

        if unified_type != old_type:
            self.var_types[var] = unified_type
            return True

        return False

    def _infer_expr_type(self, expr: MediumLevelILInstruction) -> MLILType:
        '''Infer type of an expression

        Returns:
            Inferred type of the expression
        '''
        # Constant
        if isinstance(expr, MLILConst):
            return self._infer_const_type(expr)

        # SSA variable reference
        elif isinstance(expr, MLILVarSSA):
            return self.var_types.get(expr.var, MLILType.unknown())

        # Binary operations
        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod)):
            lhs_type = self._infer_expr_type(expr.lhs)
            rhs_type = self._infer_expr_type(expr.rhs)
            return get_operation_result_type(self._get_op_name(expr), lhs_type, rhs_type)

        elif isinstance(expr, (MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr)):
            return MLILType.int_type()

        elif isinstance(expr, (MLILLogicalAnd, MLILLogicalOr)):
            return MLILType.bool_type()

        # Comparison operations
        elif isinstance(expr, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            return MLILType.bool_type()

        # Unary operations
        elif isinstance(expr, MLILNeg):
            operand_type = self._infer_expr_type(expr.operand)
            return operand_type if operand_type.is_numeric() else MLILType.int_type()

        elif isinstance(expr, (MLILLogicalNot, MLILTestZero)):
            return MLILType.bool_type()

        # Function calls
        elif isinstance(expr, MLILCall):
            return self._infer_call_type(expr)

        elif isinstance(expr, MLILSyscall):
            return self._infer_syscall_type(expr)

        elif isinstance(expr, MLILCallScript):
            return self._infer_script_call_type(expr)

        # Memory operations
        elif isinstance(expr, MLILLoadReg):
            # Registers in this VM typically hold integers (syscall returns, indices, etc.)
            return MLILType.int_type()

        elif isinstance(expr, MLILLoadGlobal):
            return MLILType.unknown()  # Global variables could be anything

        # Unknown
        else:
            return MLILType.unknown()

    def _infer_const_type(self, const: MLILConst) -> MLILType:
        '''Infer type from constant value

        Returns:
            Type inferred from constant
        '''
        # Check if it's a string (heuristic: non-numeric representation)
        if not const.is_hex and isinstance(const.value, str):
            return MLILType.string_type()

        # Numeric constant
        if isinstance(const.value, int):
            # Small values might be bool
            if const.value in (0, 1):
                return MLILType.bool_type()
            return MLILType.int_type()

        elif isinstance(const.value, float):
            return MLILType.float_type()

        # Unknown
        return MLILType.unknown()

    def _infer_phi_type(self, phi: MLILPhi) -> MLILType:
        '''Infer type from Phi node by unifying source types

        Returns:
            Unified type of all Phi sources
        '''
        result_type = MLILType.unknown()

        for source_var, _ in phi.sources:
            source_type = self.var_types.get(source_var, MLILType.unknown())
            result_type = unify_types(result_type, source_type)

        return result_type

    def _infer_call_type(self, call: MLILCall) -> MLILType:
        '''Infer return type from function call

        Returns:
            Inferred return type
        '''
        # TODO: Add function signature database
        # For now, assume unknown
        return MLILType.unknown()

    def _infer_syscall_type(self, syscall: MLILSyscall) -> MLILType:
        '''Infer return type from syscall

        Returns:
            Inferred return type based on syscall
        '''
        # Heuristic: some syscalls return specific types
        # This would need a syscall signature database
        return MLILType.unknown()

    def _infer_script_call_type(self, call: MLILCallScript) -> MLILType:
        '''Infer return type from script call

        Returns:
            Inferred return type
        '''
        # TODO: Add script function signature database
        return MLILType.unknown()

    def _infer_from_usage(self) -> bool:
        '''Infer types from how variables are used (backward inference)

        Returns:
            True if any type changed
        '''
        changed = False

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                # Check comparisons: if var compared with int/float, it's probably that type
                if isinstance(inst, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
                    changed |= self._infer_from_comparison(inst)

                # Check arithmetic: if var used in arithmetic, it's numeric
                elif isinstance(inst, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod)):
                    changed |= self._infer_from_arithmetic(inst)

                # Check conditional: if var used as condition, it's likely int/bool
                elif isinstance(inst, MLILIf):
                    changed |= self._infer_from_condition(inst.condition)

        return changed

    def _infer_from_comparison(self, comp: MediumLevelILInstruction) -> bool:
        '''Infer type from comparison operation'''
        changed = False

        # Comparison operations have lhs and rhs
        if not isinstance(comp, MLILBinaryOp):
            return changed

        lhs = comp.lhs
        rhs = comp.rhs

        # If comparing with constant, infer type from constant
        if isinstance(rhs, MLILConst):
            const_type = self._infer_const_type(rhs)
            if isinstance(lhs, MLILVarSSA) and not const_type.is_unknown():
                changed |= self._update_var_type(lhs.var, const_type)

        elif isinstance(lhs, MLILConst):
            const_type = self._infer_const_type(lhs)
            if isinstance(rhs, MLILVarSSA) and not const_type.is_unknown():
                changed |= self._update_var_type(rhs.var, const_type)

        return changed

    def _infer_from_arithmetic(self, arith: MediumLevelILInstruction) -> bool:
        '''Infer that variables in arithmetic are numeric'''
        changed = False

        # Arithmetic operations have lhs and rhs
        if not isinstance(arith, MLILBinaryOp):
            return changed

        lhs = arith.lhs
        rhs = arith.rhs

        # Variables in arithmetic should be numeric (at least int)
        if isinstance(lhs, MLILVarSSA):
            var_type = self.var_types.get(lhs.var, MLILType.unknown())
            if var_type.is_unknown():
                changed |= self._update_var_type(lhs.var, MLILType.int_type())

        if isinstance(rhs, MLILVarSSA):
            var_type = self.var_types.get(rhs.var, MLILType.unknown())
            if var_type.is_unknown():
                changed |= self._update_var_type(rhs.var, MLILType.int_type())

        return changed

    def _infer_from_condition(self, cond: MediumLevelILInstruction) -> bool:
        '''Infer type from conditional expression'''
        changed = False

        # Check if condition is a comparison
        if isinstance(cond, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            changed |= self._infer_from_comparison(cond)

        # Check if condition is arithmetic
        elif isinstance(cond, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod)):
            changed |= self._infer_from_arithmetic(cond)

        # Variable used directly as condition - likely int or bool
        elif isinstance(cond, MLILVarSSA):
            var_type = self.var_types.get(cond.var, MLILType.unknown())
            if var_type.is_unknown():
                changed |= self._update_var_type(cond.var, MLILType.int_type())

        return changed

    def _get_op_name(self, expr: MediumLevelILInstruction) -> str:
        '''Get operation name for type inference

        Returns:
            Operation name (lowercase)
        '''
        if isinstance(expr, MLILAdd):
            return 'add'
        elif isinstance(expr, MLILSub):
            return 'sub'
        elif isinstance(expr, MLILMul):
            return 'mul'
        elif isinstance(expr, MLILDiv):
            return 'div'
        elif isinstance(expr, MLILMod):
            return 'mod'
        elif isinstance(expr, MLILAnd):
            return 'and'
        elif isinstance(expr, MLILOr):
            return 'or'
        elif isinstance(expr, MLILXor):
            return 'xor'
        elif isinstance(expr, MLILShl):
            return 'shl'
        elif isinstance(expr, MLILShr):
            return 'shr'
        elif isinstance(expr, MLILEq):
            return 'eq'
        elif isinstance(expr, MLILNe):
            return 'ne'
        elif isinstance(expr, MLILLt):
            return 'lt'
        elif isinstance(expr, MLILLe):
            return 'le'
        elif isinstance(expr, MLILGt):
            return 'gt'
        elif isinstance(expr, MLILGe):
            return 'ge'
        elif isinstance(expr, MLILNeg):
            return 'neg'
        elif isinstance(expr, MLILLogicalNot):
            return 'logical_not'
        elif isinstance(expr, MLILLogicalAnd):
            return 'logical_and'
        elif isinstance(expr, MLILLogicalOr):
            return 'logical_or'
        else:
            return 'unknown'


def infer_types(function: MediumLevelILFunction) -> Dict[MLILVariableSSA, MLILType]:
    '''Convenience function to infer types for MLIL function

    Args:
        function: MLIL function in SSA form

    Returns:
        Mapping from SSA variables to their inferred types
    '''
    inference = MLILTypeInference(function)
    return inference.infer_types()

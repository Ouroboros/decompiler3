'''Expression Simplification Pass

Apply algebraic simplifications:
- x + 0 → x
- x * 1 → x
- x * 0 → 0
- x & 0 → 0
- x | 0 → x
etc.
'''

from typing import Optional
from ir.pipeline import Pass
from ..mlil import (
    MediumLevelILFunction,
    MediumLevelILInstruction,
    MLILConst,
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
    MLILNeg,
    MLILLogicalNot,
    MLILBitwiseNot,
    MLILTestZero,
    MLILEq,
    MLILNe,
    MLILLt,
    MLILLe,
    MLILGt,
    MLILGe,
    MLILRet,
)
from ..mlil_ssa import MLILSetVarSSA, MLILIf


ALL_BITS = 0xFFFFFFFF

# x OP neutral == x for an integer x
NEUTRAL_OPERANDS = {
    MLILAdd:    0,
    MLILSub:    0,
    MLILMul:    1,
    MLILDiv:    1,
    MLILAnd:    ALL_BITS,
    MLILOr:     0,
    MLILXor:    0,
    MLILShl:    0,
    MLILShr:    0,
}

COMMUTATIVE_OPS = (MLILAdd, MLILMul, MLILAnd, MLILOr, MLILXor)

# Operators whose result is an integer unless the VM logs a type error
INTEGER_RESULT_OPS = (
    MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe, MLILLogicalAnd, MLILLogicalOr, MLILLogicalNot,
    MLILTestZero, MLILMod, MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr, MLILBitwiseNot,
)


class ExpressionSimplificationPass(Pass):
    '''Apply algebraic simplifications'''

    def run(self, func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Apply simplification to all instructions'''
        for block in func.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._simplify_expr_in_inst(inst)
                new_instructions.append(new_inst)

            block.instructions = new_instructions

        return func

    def _simplify_expr_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Simplify expressions in a single instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._simplify_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value, address = inst.address).copy_metadata_from(inst)

        elif isinstance(inst, MLILIf):
            new_condition = self._simplify_expr(inst.condition)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target, address = inst.address).copy_metadata_from(inst)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._simplify_expr(inst.value)
                if new_value is not inst.value:
                    return MLILRet(new_value, address = inst.address).copy_metadata_from(inst)

        return inst

    def _simplify_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively apply algebraic simplifications'''
        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                             MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr)):
            lhs = self._simplify_expr(expr.lhs)
            rhs = self._simplify_expr(expr.rhs)

            simplified = self._apply_algebraic_identity(type(expr), lhs, rhs)
            if simplified is not None:
                return simplified

            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILBitwiseNot)):
            operand = self._simplify_expr(expr.operand)
            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def _apply_algebraic_identity(self, op_type, lhs, rhs) -> Optional[MediumLevelILInstruction]:
        '''Apply algebraic identity rules that keep the value and its type

        Only integer constants match: 0.0 or 1.0 would turn an integer operand into a float.
        x & 0 and x | ALL_BITS are integers whatever x is; every other rule needs an integer x,
        because a float or string x would come back as a different value or type.
        '''
        if op_type == MLILAnd and (self._is_int_const(lhs, 0) or self._is_int_const(rhs, 0)):
            return MLILConst(0, is_hex = False)

        if op_type == MLILOr and (self._is_int_const(lhs, ALL_BITS) or self._is_int_const(rhs, ALL_BITS)):
            return MLILConst(ALL_BITS, is_hex = True)

        if op_type == MLILMul and (self._is_int_const(lhs, 0) or self._is_int_const(rhs, 0)):
            other = rhs if self._is_int_const(lhs, 0) else lhs
            return MLILConst(0, is_hex = False) if self._is_integer(other) else None

        neutral = NEUTRAL_OPERANDS.get(op_type)
        if neutral is None:
            return None

        if self._is_int_const(rhs, neutral) and self._is_integer(lhs):
            return lhs

        if op_type in COMMUTATIVE_OPS and self._is_int_const(lhs, neutral) and self._is_integer(rhs):
            return rhs

        return None

    @classmethod
    def _is_int_const(cls, expr: MediumLevelILInstruction, value: int) -> bool:
        return isinstance(expr, MLILConst) and type(expr.value) is int and expr.value == value

    @classmethod
    def _is_integer(cls, expr: MediumLevelILInstruction) -> bool:
        '''Whether expr is an integer on every path that does not log a VM type error'''
        if isinstance(expr, MLILConst):
            return type(expr.value) is int

        if isinstance(expr, INTEGER_RESULT_OPS):
            return True

        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv)):
            return cls._is_integer(expr.lhs) and cls._is_integer(expr.rhs)

        if isinstance(expr, MLILNeg):
            return cls._is_integer(expr.operand)

        return False

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

        elif isinstance(expr, MLILBitwiseNot):
            return MLILBitwiseNot(operand)

        elif isinstance(expr, MLILTestZero):
            return MLILTestZero(operand)

        else:
            raise NotImplementedError(f'Unhandled unary operation: {type(expr).__name__}')

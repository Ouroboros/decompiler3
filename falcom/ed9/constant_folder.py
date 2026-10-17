'''Constant folding with the ED9 VM value semantics'''

import math
from typing import Any, Callable, Dict

from ir.mlil import (
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
    MLILEq,
    MLILNe,
    MLILLt,
    MLILLe,
    MLILGt,
    MLILGe,
    MLILNeg,
    MLILLogicalNot,
    MLILBitwiseNot,
    MLILTestZero,
)
from ir.mlil.passes import ConstantFolder, NotFoldable

from .vm_values import EQUALITY_OPS, Op, VMFault, binary_op, is_zero, normalize, unary_op


BINARY_OPS = {
    MLILAdd:        Op.ADD,
    MLILSub:        Op.SUB,
    MLILMul:        Op.MUL,
    MLILDiv:        Op.DIV,
    MLILMod:        Op.MOD,
    MLILAnd:        Op.BIT_AND,
    MLILOr:         Op.BIT_OR,
    MLILXor:        Op.BIT_XOR,
    MLILShl:        Op.SHL,
    MLILShr:        Op.SHR,
    MLILLogicalAnd: Op.LOGICAL_AND,
    MLILLogicalOr:  Op.LOGICAL_OR,
    MLILEq:         Op.EQ,
    MLILNe:         Op.NE,
    MLILLt:         Op.LT,
    MLILLe:         Op.LE,
    MLILGt:         Op.GT,
    MLILGe:         Op.GE,
}

UNARY_OPS = {
    MLILNeg:        Op.NEG,
    MLILLogicalNot: Op.LOGICAL_NOT,
    MLILBitwiseNot: Op.BIT_NOT,
    MLILTestZero:   Op.TEST_ZERO,
}

# Values an MLIL constant can hold and print as a literal
LITERAL_TYPES = (int, float, str)


class ED9ConstantFolder(ConstantFolder):
    '''Folds constants as VMContext::execute computes them (falcom/ed9/spec/vm_spec.md section 7)

    Nothing is folded when the result depends on a string's file offset, faults at run time
    (integer division by zero) or has no literal form (Raw words, infinities, NaN).
    '''

    @classmethod
    def binary(cls, op_type: type, lhs: Any, rhs: Any) -> Any:
        op = cls._vm_op(BINARY_OPS, op_type)
        left, right = cls._operand(lhs), cls._operand(rhs)
        if op in EQUALITY_OPS and isinstance(left, str) != isinstance(right, str):
            # A string equals the integer holding its file offset, which is unknown here
            raise NotFoldable('string offset')

        return cls._evaluate(binary_op, op, left, right)

    @classmethod
    def unary(cls, op_type: type, operand: Any) -> Any:
        return cls._evaluate(unary_op, cls._vm_op(UNARY_OPS, op_type), cls._operand(operand))

    @classmethod
    def is_true(cls, value: Any) -> bool:
        # POP_JMP_ZERO / POP_JMP_NOT_ZERO test the 30-bit payload; string offsets are never zero
        return not is_zero(cls._operand(value))

    @classmethod
    def _vm_op(cls, table: Dict[type, Op], op_type: type) -> Op:
        op = table.get(op_type)
        if op is None:
            raise NotFoldable(op_type.__name__)

        return op

    @classmethod
    def _operand(cls, value: Any) -> Any:
        if not isinstance(value, LITERAL_TYPES):
            raise NotFoldable(type(value).__name__)

        return normalize(value)

    @classmethod
    def _evaluate(cls, evaluate: Callable[..., Any], op: Op, *operands: Any) -> Any:
        try:
            result = evaluate(op, *operands)

        except VMFault as fault:
            raise NotFoldable(fault.kind) from fault

        if not isinstance(result, LITERAL_TYPES) or (isinstance(result, float) and not math.isfinite(result)):
            raise NotFoldable(repr(result))

        return result

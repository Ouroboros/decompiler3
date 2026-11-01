'''ED9 value semantics for the IR (ir/core/target.py ValueSemantics)'''

import math
from typing import Any, Callable, Dict, Iterable, Optional

from ir.core import ValueOp, ValueSemantics, NotFoldable

from .vm_values import (
    EQUALITY_OPS, PAYLOAD_SIGN_BIT, Op, Raw, VMFault, binary_op, is_zero, normalize, unary_op,
)


VM_OPS = {
    ValueOp.ADD:            Op.ADD,
    ValueOp.SUB:            Op.SUB,
    ValueOp.MUL:            Op.MUL,
    ValueOp.DIV:            Op.DIV,
    ValueOp.MOD:            Op.MOD,
    ValueOp.BIT_AND:        Op.BIT_AND,
    ValueOp.BIT_OR:         Op.BIT_OR,
    ValueOp.BIT_XOR:        Op.BIT_XOR,
    ValueOp.SHL:            Op.SHL,
    ValueOp.SHR:            Op.SHR,
    ValueOp.LOGICAL_AND:    Op.LOGICAL_AND,
    ValueOp.LOGICAL_OR:     Op.LOGICAL_OR,
    ValueOp.EQ:             Op.EQ,
    ValueOp.NE:             Op.NE,
    ValueOp.LT:             Op.LT,
    ValueOp.LE:             Op.LE,
    ValueOp.GT:             Op.GT,
    ValueOp.GE:             Op.GE,
    ValueOp.NEG:            Op.NEG,
    ValueOp.LOGICAL_NOT:    Op.LOGICAL_NOT,
    ValueOp.BIT_NOT:        Op.BIT_NOT,
    ValueOp.TEST_ZERO:      Op.TEST_ZERO,
}

# Values an IR constant can hold and print as a literal
LITERAL_TYPES = (int, float, str, Raw)

# Signed range of a 30-bit Integer payload; integer literals outside it wrap in the VM
INT_PAYLOAD_MIN = -PAYLOAD_SIGN_BIT
INT_PAYLOAD_MAX = PAYLOAD_SIGN_BIT - 1

LITERAL_COMPARISONS = {
    ValueOp.EQ: lambda lhs, rhs: lhs == rhs,
    ValueOp.NE: lambda lhs, rhs: lhs != rhs,
    ValueOp.LT: lambda lhs, rhs: lhs < rhs,
    ValueOp.LE: lambda lhs, rhs: lhs <= rhs,
    ValueOp.GT: lambda lhs, rhs: lhs > rhs,
    ValueOp.GE: lambda lhs, rhs: lhs >= rhs,
}


class ED9ConstantFolder(ValueSemantics):
    '''Computes on constants as VMContext::execute does (falcom/ed9/spec/vm_spec.md section 7)

    Nothing is folded when the result depends on a string's file offset, faults at run time
    (integer division by zero, type errors) or has no literal form (infinities, NaN).
    '''

    @classmethod
    def binary(cls, op: ValueOp, lhs: Any, rhs: Any) -> Any:
        vm_op = cls._vm_op(op)
        left, right = cls._operand(lhs), cls._operand(rhs)
        if vm_op in EQUALITY_OPS and isinstance(left, str) != isinstance(right, str):
            # A string equals the integer holding its file offset, which is unknown here
            raise NotFoldable('string offset')

        return cls._evaluate(binary_op, vm_op, left, right)

    @classmethod
    def unary(cls, op: ValueOp, operand: Any) -> Any:
        return cls._evaluate(unary_op, cls._vm_op(op), cls._operand(operand))

    @classmethod
    def is_true(cls, value: Any) -> bool:
        # POP_JMP_ZERO / POP_JMP_NOT_ZERO test the 30-bit payload; string offsets are never zero
        return not is_zero(cls._operand(value))

    @classmethod
    def labels_distinct(cls, values: Iterable[Any]) -> bool:
        '''No two labels compare equal under VM equality (an integer equals the same-valued float);
        strings and numbers are not mixed because a string compares by file offset'''
        values = list(values)
        if any(isinstance(value, str) for value in values) and not all(isinstance(value, str) for value in values):
            return False

        # Raw words have no common equality with numbers here
        if not all(isinstance(value, (str, int, float)) for value in values):
            return False

        keys = [value if isinstance(value, str) else float(value) for value in values]
        return len(set(keys)) == len(keys)

    @classmethod
    def compare_literals(cls, op: ValueOp, lhs: Any, rhs: Any) -> Optional[bool]:
        '''Only where the comparison matches the VM: both Integers (signed payloads) or both Floats.
        Strings compare by file offset and mixed operands through float32, so those stay.'''
        if type(lhs) is not type(rhs) or type(lhs) not in (int, float):
            return None

        if type(lhs) is int and not all(INT_PAYLOAD_MIN <= value <= INT_PAYLOAD_MAX for value in (lhs, rhs)):
            return None

        compare = LITERAL_COMPARISONS.get(op)
        return None if compare is None else compare(lhs, rhs)

    @classmethod
    def _vm_op(cls, op: ValueOp) -> Op:
        vm_op = VM_OPS.get(op)
        if vm_op is None:
            raise NotFoldable(op.name)

        return vm_op

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

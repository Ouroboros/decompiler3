'''TVM as an IR target: the only way TVM reaches the decompiler core (ir/core/target.py)'''

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from ir.core import BackendHints, CallModel, CallTarget, ModuleMetadata, NotFoldable, StorageModel, Target, ValueOp, ValueSemantics

from .isa import WORD_SIZE


INT_BITS        = 32
INT_MODULUS     = 1 << INT_BITS
INT_SIGN_BIT    = 1 << (INT_BITS - 1)
TRUE_VALUE      = 1
FALSE_VALUE     = 0
REGISTER_COUNT  = 8
RESULT_REGISTER = 3


class TVMFault(Exception):
    '''Run-time error of a TVM operation (division by zero, operand types)'''


def wrap(value: int) -> int:
    return ((value + INT_SIGN_BIT) % INT_MODULUS) - INT_SIGN_BIT


def truth(value: Any) -> bool:
    '''Branch test: strings are true, integers when nonzero'''
    return isinstance(value, str) or value != 0


def _int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TVMFault(f'integer operand expected, got {value!r}')

    return value


def binary(op: ValueOp, lhs: Any, rhs: Any) -> Any:
    if op in (ValueOp.EQ, ValueOp.NE):
        equal = type(lhs) is type(rhs) and lhs == rhs
        return TRUE_VALUE if equal == (op == ValueOp.EQ) else FALSE_VALUE

    if op == ValueOp.LOGICAL_AND:
        return TRUE_VALUE if truth(lhs) and truth(rhs) else FALSE_VALUE

    if op == ValueOp.LOGICAL_OR:
        return TRUE_VALUE if truth(lhs) or truth(rhs) else FALSE_VALUE

    left, right = _int(lhs), _int(rhs)
    match op:
        case ValueOp.ADD:
            return wrap(left + right)

        case ValueOp.SUB:
            return wrap(left - right)

        case ValueOp.MUL:
            return wrap(left * right)

        case ValueOp.DIV | ValueOp.MOD:
            if right == 0:
                raise TVMFault('division by zero')

            return wrap(left // right if op == ValueOp.DIV else left % right)

        case ValueOp.BIT_AND:
            return wrap(left & right)

        case ValueOp.BIT_OR:
            return wrap(left | right)

        case ValueOp.LT:
            return TRUE_VALUE if left < right else FALSE_VALUE

        case ValueOp.LE:
            return TRUE_VALUE if left <= right else FALSE_VALUE

        case ValueOp.GT:
            return TRUE_VALUE if left > right else FALSE_VALUE

        case ValueOp.GE:
            return TRUE_VALUE if left >= right else FALSE_VALUE

    raise TVMFault(f'no TVM operator {op.name}')


def unary(op: ValueOp, value: Any) -> Any:
    if op == ValueOp.LOGICAL_NOT:
        return FALSE_VALUE if truth(value) else TRUE_VALUE

    operand = _int(value)
    match op:
        case ValueOp.NEG:
            return wrap(-operand)

        case ValueOp.BIT_NOT:
            return wrap(~operand)

        case ValueOp.TEST_ZERO:
            return TRUE_VALUE if operand == 0 else FALSE_VALUE

    raise TVMFault(f'no TVM operator {op.name}')


class TVMValues(ValueSemantics):

    @classmethod
    def binary(cls, op: ValueOp, lhs: Any, rhs: Any) -> Any:
        try:
            return binary(op, lhs, rhs)

        except TVMFault as fault:
            raise NotFoldable(str(fault)) from fault

    @classmethod
    def unary(cls, op: ValueOp, operand: Any) -> Any:
        try:
            return unary(op, operand)

        except TVMFault as fault:
            raise NotFoldable(str(fault)) from fault

    @classmethod
    def is_true(cls, value: Any) -> bool:
        if not isinstance(value, (int, str)):
            raise NotFoldable(type(value).__name__)

        return truth(value)

    @classmethod
    def labels_distinct(cls, values: Iterable[Any]) -> bool:
        values = list(values)
        return all(type(value) is int for value in values) and len(set(values)) == len(values)

    @classmethod
    def compare_literals(cls, op: ValueOp, lhs: Any, rhs: Any) -> Optional[bool]:
        if type(lhs) is not int or type(rhs) is not int:
            return None

        return binary(op, lhs, rhs) == TRUE_VALUE


class TVMStorage(StorageModel):
    global_name     = 'G'
    register_name   = 'R'
    register_count  = REGISTER_COUNT


class TVMCalls(CallModel):
    # Parameter 1 is pushed first, the base default
    return_register = RESULT_REGISTER


@dataclass(frozen = True)
class TVMHost(CallTarget):
    '''Host function called by name'''
    name: str

    def format_call(self, args: List[str]) -> str:
        return f'host.{self.name}({", ".join(args)})'


class TVMHints(BackendHints):

    @classmethod
    def typescript_declarations(cls) -> str:
        return '// Host functions\ndeclare const host: any;\n\n'


@dataclass(frozen = True)
class TVMMetadata(ModuleMetadata):
    version: int
    global_count: int
    strings: Tuple[str, ...]
    function_order: Tuple[str, ...]

    @classmethod
    def fields(cls) -> List[str]:
        return [item.name for item in dataclasses.fields(cls)]

    @classmethod
    def from_image(cls, image) -> 'TVMMetadata':
        return cls(image.version, image.global_count, tuple(image.strings), tuple(func.name for func in image.functions))


class TVMTarget(Target):
    name        = 'tvm'
    word_size   = WORD_SIZE
    values      = TVMValues
    storage     = TVMStorage
    calls       = TVMCalls
    hints       = TVMHints
    metadata    = TVMMetadata

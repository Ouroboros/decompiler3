'''VM value semantics shared by the bytecode and HLIL interpreters'''

import math
import struct
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable


# Integer arithmetic wraps to a signed 32-bit word (assumption, verify against the VM)
INT_BITS            = 32
INT_MODULUS         = 1 << INT_BITS
INT_SIGN_BIT        = 1 << (INT_BITS - 1)
SHIFT_COUNT_MASK    = INT_BITS - 1

# Float arithmetic is rounded to IEEE single precision (assumption, verify against the VM)
FLOAT32_FORMAT      = '<f'

TRUE_VALUE          = 1
FALSE_VALUE         = 0


class VMFault(Exception):
    '''Execution fault that terminates a run and is recorded as a trace event'''

    def __init__(self, kind: str, detail: str = ''):
        super().__init__(f'{kind}: {detail}' if detail else kind)
        self.kind = kind
        self.detail = detail


class _Undefined:
    '''Value of a storage cell that was read before any write'''

    def __repr__(self) -> str:
        return 'UNDEF'


UNDEF = _Undefined()


@dataclass(frozen = True)
class Pointer:
    '''Address of a storage cell; space and key are owned by the interpreter that created it'''
    space: str
    key: Any


class Op(Enum):
    '''Operations with VM semantics'''
    ADD         = auto()
    SUB         = auto()
    MUL         = auto()
    DIV         = auto()
    MOD         = auto()
    EQ          = auto()
    NE          = auto()
    LT          = auto()
    LE          = auto()
    GT          = auto()
    GE          = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR  = auto()
    BIT_AND     = auto()
    BIT_OR      = auto()
    BIT_XOR     = auto()
    SHL         = auto()
    SHR         = auto()
    NEG         = auto()
    BIT_NOT     = auto()
    LOGICAL_NOT = auto()


COMPARISON_OPS  = frozenset({Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE})
EQUALITY_OPS    = frozenset({Op.EQ, Op.NE})
ARITHMETIC_OPS  = frozenset({Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD})
BITWISE_OPS     = frozenset({Op.BIT_AND, Op.BIT_OR, Op.BIT_XOR, Op.SHL, Op.SHR})


def wrap_int(value: int) -> int:
    return ((value + INT_SIGN_BIT) % INT_MODULUS) - INT_SIGN_BIT


def to_float32(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return value

    try:
        return struct.unpack(FLOAT32_FORMAT, struct.pack(FLOAT32_FORMAT, value))[0]

    except OverflowError:
        return math.copysign(math.inf, value)


def normalize(value: Any) -> Any:
    '''Canonical representation of a freshly produced value'''
    if isinstance(value, bool):
        return TRUE_VALUE if value else FALSE_VALUE

    if isinstance(value, int):
        return wrap_int(int(value))

    if isinstance(value, float):
        return to_float32(value)

    return value


def bool_value(flag: bool) -> int:
    return TRUE_VALUE if flag else FALSE_VALUE


def is_zero(value: Any) -> bool:
    '''Branch condition test used by POP_JMP_ZERO and structured conditions'''
    if value is UNDEF:
        raise VMFault('undefined_condition')

    if isinstance(value, (int, float)):
        return value == 0

    # Strings and pointers are non-null references
    return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _trunc_div(lhs: int, rhs: int) -> int:
    quotient = abs(lhs) // abs(rhs)
    return quotient if (lhs >= 0) == (rhs >= 0) else -quotient


def _compare(op: Op, lhs: Any, rhs: Any) -> int:
    if op in EQUALITY_OPS:
        equal = lhs == rhs if type(lhs) is type(rhs) or (_is_number(lhs) and _is_number(rhs)) else False
        return bool_value(equal if op == Op.EQ else not equal)

    if not ((_is_number(lhs) and _is_number(rhs)) or (isinstance(lhs, str) and isinstance(rhs, str))):
        raise VMFault('type_mismatch', f'{op.name} {type(lhs).__name__} {type(rhs).__name__}')

    match op:
        case Op.LT:
            return bool_value(lhs < rhs)

        case Op.LE:
            return bool_value(lhs <= rhs)

        case Op.GT:
            return bool_value(lhs > rhs)

        case _:
            return bool_value(lhs >= rhs)


def _arithmetic(op: Op, lhs: Any, rhs: Any) -> Any:
    if not (_is_number(lhs) and _is_number(rhs)):
        raise VMFault('type_mismatch', f'{op.name} {type(lhs).__name__} {type(rhs).__name__}')

    if isinstance(lhs, float) or isinstance(rhs, float):
        match op:
            case Op.ADD:
                return to_float32(lhs + rhs)

            case Op.SUB:
                return to_float32(lhs - rhs)

            case Op.MUL:
                return to_float32(lhs * rhs)

            case _:
                if rhs == 0:
                    raise VMFault('div_by_zero')

                return to_float32(lhs / rhs if op == Op.DIV else math.fmod(lhs, rhs))

    match op:
        case Op.ADD:
            return wrap_int(lhs + rhs)

        case Op.SUB:
            return wrap_int(lhs - rhs)

        case Op.MUL:
            return wrap_int(lhs * rhs)

        case _:
            if rhs == 0:
                raise VMFault('div_by_zero')

            quotient = _trunc_div(lhs, rhs)
            return wrap_int(quotient if op == Op.DIV else lhs - rhs * quotient)


def _bitwise(op: Op, lhs: Any, rhs: Any) -> int:
    if not (isinstance(lhs, int) and isinstance(rhs, int)):
        raise VMFault('type_mismatch', f'{op.name} {type(lhs).__name__} {type(rhs).__name__}')

    match op:
        case Op.BIT_AND:
            return wrap_int(lhs & rhs)

        case Op.BIT_OR:
            return wrap_int(lhs | rhs)

        case Op.BIT_XOR:
            return wrap_int(lhs ^ rhs)

        case Op.SHL:
            return wrap_int(lhs << (rhs & SHIFT_COUNT_MASK))

        case _:
            return wrap_int(lhs >> (rhs & SHIFT_COUNT_MASK))


def binary_op(op: Op, lhs: Any, rhs: Any) -> Any:
    if lhs is UNDEF or rhs is UNDEF:
        return UNDEF

    if op == Op.LOGICAL_AND:
        return bool_value(not is_zero(lhs) and not is_zero(rhs))

    if op == Op.LOGICAL_OR:
        return bool_value(not is_zero(lhs) or not is_zero(rhs))

    if isinstance(lhs, Pointer) or isinstance(rhs, Pointer):
        if op in EQUALITY_OPS:
            return _compare(op, lhs, rhs)

        raise VMFault('pointer_arithmetic', op.name)

    if op in COMPARISON_OPS:
        return _compare(op, lhs, rhs)

    if op in ARITHMETIC_OPS:
        return _arithmetic(op, lhs, rhs)

    if op in BITWISE_OPS:
        return _bitwise(op, lhs, rhs)

    raise VMFault('unsupported_op', op.name)


def unary_op(op: Op, value: Any) -> Any:
    if value is UNDEF:
        return UNDEF

    if op == Op.LOGICAL_NOT:
        return bool_value(is_zero(value))

    if op == Op.NEG:
        if not _is_number(value):
            raise VMFault('type_mismatch', f'NEG {type(value).__name__}')

        return to_float32(-value) if isinstance(value, float) else wrap_int(-value)

    if op == Op.BIT_NOT:
        if not isinstance(value, int):
            raise VMFault('type_mismatch', f'BIT_NOT {type(value).__name__}')

        return wrap_int(~value)

    raise VMFault('unsupported_op', op.name)


def observable(value: Any, deref: Callable[[Pointer], Any], depth: int = 0) -> Any:
    '''Hashable, representation-independent form of a value for trace comparison'''
    if value is UNDEF:
        return ('undef',)

    if isinstance(value, Pointer):
        # Pointer identity differs between interpreters; compare the referenced value instead
        return ('ptr', observable(deref(value), deref, depth + 1) if depth == 0 else ('nested',))

    if isinstance(value, (bool, int, float)):
        return normalize(value)

    if isinstance(value, str):
        return value

    return ('opaque', type(value).__name__)

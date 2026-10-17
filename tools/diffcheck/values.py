'''VM value semantics shared by the bytecode and HLIL interpreters

Verified against VMContext::execute and its helpers VMContext::Equal / NotEqual in
ed9_crack.exe. Every stack slot is a 32-bit word whose top two bits are a type tag:

- tag 0 Raw:     30-bit payload (PUSH_RAW)
- tag 1 Integer: 30-bit signed integer; every integer result wraps to 30 bits
- tag 2 Float:   float32 stored as (bits >> 2), so the two lowest bits are dropped
- tag 3 String:  offset of the string in the script file

Python representation: Raw -> Raw, Integer -> int, Float -> float (already truncated),
String -> str (content; the file offset is not modeled), stack address -> Pointer.
'''

import math
import struct
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable


PAYLOAD_BITS        = 30
PAYLOAD_MASK        = (1 << PAYLOAD_BITS) - 1
PAYLOAD_MODULUS     = 1 << PAYLOAD_BITS
PAYLOAD_SIGN_BIT    = 1 << (PAYLOAD_BITS - 1)
FLOAT_DROPPED_BITS  = 2
FLOAT_KEPT_MASK     = ~((1 << FLOAT_DROPPED_BITS) - 1)
WORD_MASK           = 0xFFFFFFFF
FLOAT32_FORMAT      = '<f'
WORD_FORMAT         = '<I'

# Shift operators have no VM opcode; they only appear if the decompiler synthesizes them
SHIFT_COUNT_MASK    = 31

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
    '''Address of a storage cell; space and key are owned by the interpreter that created it

    In the VM a stack address is an integer-tagged byte offset.
    '''
    space: str
    key: Any


@dataclass(frozen = True)
class Raw:
    '''Tag-0 word; payload holds the low 30 bits read as a signed integer'''
    payload: int


@dataclass(frozen = True)
class ObservedFloat:
    '''Float as compared in traces: by bit pattern, so -0.0 and NaN compare exactly'''
    bits: int

    def __repr__(self) -> str:
        return f'{_bits_to_float(self.bits)!r}f'


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
    TEST_ZERO   = auto()    # EZ opcode
    LOGICAL_NOT = auto()    # structural negation of a branch condition


EQUALITY_OPS    = frozenset({Op.EQ, Op.NE})
ORDER_OPS       = frozenset({Op.LT, Op.LE, Op.GT, Op.GE})
COMPARISON_OPS  = EQUALITY_OPS | ORDER_OPS
ARITHMETIC_OPS  = frozenset({Op.ADD, Op.SUB, Op.MUL, Op.DIV})
INTEGER_OPS     = frozenset({Op.MOD, Op.BIT_AND, Op.BIT_OR})
SYNTHETIC_OPS   = frozenset({Op.BIT_XOR, Op.SHL, Op.SHR})
LOGICAL_OPS     = frozenset({Op.LOGICAL_AND, Op.LOGICAL_OR})


# ------------------------------------------------------------- encoding --

def wrap_int(value: int) -> int:
    return ((value + PAYLOAD_SIGN_BIT) % PAYLOAD_MODULUS) - PAYLOAD_SIGN_BIT


def _float32_bits(value: float) -> int:
    try:
        return struct.unpack(WORD_FORMAT, struct.pack(FLOAT32_FORMAT, value))[0]

    except OverflowError:
        return struct.unpack(WORD_FORMAT, struct.pack(FLOAT32_FORMAT, math.copysign(math.inf, value)))[0]


def _bits_to_float(bits: int) -> float:
    return struct.unpack(FLOAT32_FORMAT, struct.pack(WORD_FORMAT, bits & WORD_MASK))[0]


def to_vm_float(value: float) -> float:
    '''Round to float32, then drop the two lowest bits as the VM does when storing a float'''
    return _bits_to_float(_float32_bits(value) & FLOAT_KEPT_MASK)


def int_to_float(value: int) -> float:
    '''Integer operand of a float operation (cvtdq2ps rounds to the nearest float32)'''
    return _bits_to_float(_float32_bits(float(value)))


def normalize(value: Any) -> Any:
    '''Canonical representation of a freshly produced value'''
    if isinstance(value, bool):
        return TRUE_VALUE if value else FALSE_VALUE

    if isinstance(value, Raw):
        return Raw(wrap_int(value.payload))

    if isinstance(value, int):
        return wrap_int(int(value))

    if isinstance(value, float):
        return to_vm_float(value)

    return value


def bool_value(flag: bool) -> int:
    return TRUE_VALUE if flag else FALSE_VALUE


# -------------------------------------------------------------- operands --

def _is_integer_tagged(value: Any) -> bool:
    return isinstance(value, (int, Pointer))


def _float_operand(value: Any) -> float:
    '''Operand of a float operation: integers are converted, other words are reinterpreted'''
    if isinstance(value, float):
        return value

    if isinstance(value, int):
        return int_to_float(value)

    if isinstance(value, Raw):
        return _bits_to_float((value.payload & PAYLOAD_MASK) << FLOAT_DROPPED_BITS)

    if isinstance(value, Pointer):
        raise VMFault('pointer_arithmetic')

    raise VMFault('unmodeled_string_word', 'string used as float')


def _payload(value: Any) -> int:
    '''Low 30 bits of a word read as a signed integer'''
    if isinstance(value, int):
        return value

    if isinstance(value, Raw):
        return value.payload

    if isinstance(value, float):
        return wrap_int(_float32_bits(value) >> FLOAT_DROPPED_BITS)

    if isinstance(value, Pointer):
        raise VMFault('pointer_arithmetic')

    raise VMFault('unmodeled_string_word', 'string used as integer')


def is_zero(value: Any) -> bool:
    '''Branch test of POP_JMP_ZERO / POP_JMP_NOT_ZERO: the 30-bit payload is zero

    Among floats only +0.0 is zero (-0.0 has a nonzero payload); strings are nonzero offsets.
    '''
    if value is UNDEF:
        raise VMFault('undefined_condition')

    if isinstance(value, float):
        return ((_float32_bits(value) >> FLOAT_DROPPED_BITS) & PAYLOAD_MASK) == 0

    if isinstance(value, int):
        return value == 0

    if isinstance(value, Raw):
        return value.payload == 0

    return False


def logical_truth(value: Any) -> bool:
    '''Operand test of LOGICAL_AND / LOGICAL_OR for one operand: floats compare with 0.0'''
    if value is UNDEF:
        raise VMFault('undefined_condition')

    if isinstance(value, float):
        return value != 0.0

    return not is_zero(value)


def _operand_truth(value: Any, float_mode: bool) -> bool:
    '''LOGICAL_AND / LOGICAL_OR compare both operands as floats when either one is a float'''
    if float_mode and not isinstance(value, (str, Pointer)):
        return _float_operand(value) != 0.0

    return not is_zero(value)


# ------------------------------------------------------------ operations --

def _trunc_div(lhs: int, rhs: int) -> int:
    quotient = abs(lhs) // abs(rhs)
    return quotient if (lhs >= 0) == (rhs >= 0) else -quotient


def _float_divide(lhs: float, rhs: float) -> float:
    if rhs != 0.0:
        return lhs / rhs

    if lhs == 0.0 or math.isnan(lhs):
        return math.nan

    return math.copysign(math.inf, lhs) * math.copysign(1.0, rhs)


def _equality(op: Op, lhs: Any, rhs: Any) -> int:
    if isinstance(lhs, str) and isinstance(rhs, str):
        equal = lhs == rhs

    elif isinstance(lhs, (str, Pointer)) or isinstance(rhs, (str, Pointer)):
        # String offsets and stack offsets are not modeled; they never equal a plain number in practice
        equal = lhs == rhs

    elif isinstance(lhs, float) or isinstance(rhs, float):
        equal = _float_operand(lhs) == _float_operand(rhs)

    else:
        equal = _payload(lhs) == _payload(rhs)

    return bool_value(equal if op == Op.EQ else not equal)


def _order(op: Op, lhs: Any, rhs: Any) -> int:
    if isinstance(lhs, float) or isinstance(rhs, float):
        left, right = _float_operand(lhs), _float_operand(rhs)

    else:
        left, right = _payload(lhs), _payload(rhs)

    match op:
        case Op.LT:
            return bool_value(left < right)

        case Op.LE:
            return bool_value(left <= right)

        case Op.GT:
            return bool_value(left > right)

        case _:
            return bool_value(left >= right)


def _arithmetic(op: Op, lhs: Any, rhs: Any) -> Any:
    if isinstance(lhs, float) or isinstance(rhs, float):
        left, right = _float_operand(lhs), _float_operand(rhs)
        match op:
            case Op.ADD:
                return to_vm_float(left + right)

            case Op.SUB:
                return to_vm_float(left - right)

            case Op.MUL:
                return to_vm_float(left * right)

            case _:
                return to_vm_float(_float_divide(left, right))

    left, right = _payload(lhs), _payload(rhs)
    match op:
        case Op.ADD:
            return wrap_int(left + right)

        case Op.SUB:
            return wrap_int(left - right)

        case Op.MUL:
            return wrap_int(left * right)

        case _:
            if right == 0:
                raise VMFault('div_by_zero')

            return wrap_int(_trunc_div(left, right))


def _integer_only(op: Op, lhs: Any, rhs: Any) -> Any:
    '''MOD / BITWISE_AND / BITWISE_OR: the VM logs an error and keeps the left operand
    when neither side is an integer; otherwise both payloads are used as integers'''
    if not (_is_integer_tagged(lhs) or _is_integer_tagged(rhs)):
        return lhs

    left, right = _payload(lhs), _payload(rhs)
    match op:
        case Op.MOD:
            if right == 0:
                raise VMFault('div_by_zero')

            return wrap_int(left - right * _trunc_div(left, right))

        case Op.BIT_AND:
            return wrap_int(left & right)

        case _:
            return wrap_int(left | right)


def _synthetic(op: Op, lhs: Any, rhs: Any) -> int:
    left, right = _payload(lhs), _payload(rhs)
    match op:
        case Op.BIT_XOR:
            return wrap_int(left ^ right)

        case Op.SHL:
            return wrap_int(left << (right & SHIFT_COUNT_MASK))

        case _:
            return wrap_int(left >> (right & SHIFT_COUNT_MASK))


def binary_op(op: Op, lhs: Any, rhs: Any) -> Any:
    if lhs is UNDEF or rhs is UNDEF:
        return UNDEF

    if op in LOGICAL_OPS:
        float_mode = isinstance(lhs, float) or isinstance(rhs, float)
        left, right = _operand_truth(lhs, float_mode), _operand_truth(rhs, float_mode)
        return bool_value(left and right if op == Op.LOGICAL_AND else left or right)

    if op in EQUALITY_OPS:
        return _equality(op, lhs, rhs)

    if op in ORDER_OPS:
        return _order(op, lhs, rhs)

    if op in ARITHMETIC_OPS:
        return _arithmetic(op, lhs, rhs)

    if op in INTEGER_OPS:
        return _integer_only(op, lhs, rhs)

    if op in SYNTHETIC_OPS:
        return _synthetic(op, lhs, rhs)

    raise VMFault('unsupported_op', op.name)


def unary_op(op: Op, value: Any) -> Any:
    if value is UNDEF:
        return UNDEF

    if op == Op.LOGICAL_NOT:
        return bool_value(is_zero(value))

    if op == Op.NEG and isinstance(value, float):
        return to_vm_float(-value)

    # NEG / EZ / NOT accept integers only; the VM logs an error and leaves other operands unchanged
    if not _is_integer_tagged(value):
        if op in (Op.NEG, Op.TEST_ZERO, Op.BIT_NOT):
            return value

        raise VMFault('unsupported_op', op.name)

    match op:
        case Op.NEG:
            return wrap_int(-_payload(value))

        case Op.TEST_ZERO:
            return bool_value(_payload(value) == 0)

        case Op.BIT_NOT:
            return wrap_int(~_payload(value))

        case _:
            raise VMFault('unsupported_op', op.name)


def observable(value: Any, deref: Callable[[Pointer], Any], depth: int = 0) -> Any:
    '''Hashable, representation-independent form of a value for trace comparison

    Raw words are observed by payload: script output cannot distinguish them from integers.
    '''
    if value is UNDEF:
        return ('undef',)

    if isinstance(value, Pointer):
        # Pointer identity differs between interpreters; compare the referenced value instead
        return ('ptr', observable(deref(value), deref, depth + 1) if depth == 0 else ('nested',))

    if isinstance(value, Raw):
        return normalize(value).payload

    if isinstance(value, float):
        return ObservedFloat(_float32_bits(normalize(value)))

    if isinstance(value, (bool, int)):
        return normalize(value)

    if isinstance(value, str):
        return value

    return ('opaque', type(value).__name__)

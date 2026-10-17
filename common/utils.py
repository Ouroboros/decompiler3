import math
import struct

from .config import *


FLOAT32_FORMAT          = '<f'
WORD_FORMAT             = '<I'
# Any float32 round-trips through 9 significant decimal digits
MAX_SIGNIFICANT_DIGITS  = 9


def stored_float_bits(value: float) -> int:
    '''Bits of value as the VM stores it: float32 with the configured low bits cleared'''
    bits = struct.unpack(WORD_FORMAT, struct.pack(FLOAT32_FORMAT, value))[0]
    return bits & ~((1 << default_float_dropped_bits()) - 1)


def _encodes_to(text: str, target_bits: int) -> bool:
    try:
        return stored_float_bits(float(text)) == target_bits

    except OverflowError:
        return False


def format_float(value: float) -> str:
    '''Shortest decimal text that encodes to the same stored float as value'''
    if not math.isfinite(value):
        return repr(value)

    try:
        target_bits = stored_float_bits(value)

    except OverflowError:
        return repr(value)

    for digits in range(1, MAX_SIGNIFICANT_DIGITS + 1):
        text = f'{value:.{digits}g}'
        if _encodes_to(text, target_bits):
            return repr(float(text))

    return repr(value)

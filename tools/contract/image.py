'''Independent decoding of the ED9 script tables from raw bytes

Deliberately does not use falcom.ed9.parser: it is the reference the parser is audited against.
Layout per falcom/ed9/spec/vm_spec.md section 9 and the record formats of the script image.
'''

import struct
from typing import Dict

from common.config import default_encoding

from .model import TAG_FLOAT, TAG_INT, TAG_RAW, TAG_STR, NO_CALLEE, FunctionEntry, ModuleTables, Value


MAGIC                   = b'#scp'
HEADER_FORMAT           = '<4sIIIII'
ENTRY_FORMAT            = '<IBBBBIIIIII'
ENTRY_SIZE              = struct.calcsize(ENTRY_FORMAT)
GLOBAL_FORMAT           = '<II'
GLOBAL_SIZE             = struct.calcsize(GLOBAL_FORMAT)
DEBUG_FORMAT            = '<IHHI'
DEBUG_SIZE              = struct.calcsize(DEBUG_FORMAT)
DEBUG_ARG_FORMAT        = '<II'
DEBUG_ARG_SIZE          = struct.calcsize(DEBUG_ARG_FORMAT)
WORD_FORMAT             = '<I'
WORD_BYTES              = struct.calcsize(WORD_FORMAT)
FLOAT_FORMAT            = '<f'

TAG_SHIFT               = 30
PAYLOAD_MASK            = (1 << TAG_SHIFT) - 1
PAYLOAD_SIGN            = 1 << (TAG_SHIFT - 1)
FLOAT_DROPPED_BITS      = 2
WORD_MASK               = 0xFFFFFFFF
STRING_TERMINATOR       = b'\0'
NO_FUNCTION_ID          = 0xFFFFFFFF
TAGS                    = (TAG_RAW, TAG_INT, TAG_FLOAT, TAG_STR)


class ImageError(Exception):
    pass


def read_string(data: bytes, offset: int) -> str:
    end = data.find(STRING_TERMINATOR, offset)
    if end < 0:
        raise ImageError(f'unterminated string at 0x{offset:X}')

    return data[offset:end].decode(default_encoding())


def decode_word(data: bytes, word: int) -> Value:
    '''Tagged 32-bit script value'''
    tag = TAGS[word >> TAG_SHIFT]
    payload = word & PAYLOAD_MASK
    if tag == TAG_RAW:
        return (TAG_RAW, word)

    if tag == TAG_INT:
        return (TAG_INT, payload - (payload & PAYLOAD_SIGN) * 2)

    if tag == TAG_FLOAT:
        bits = (payload << FLOAT_DROPPED_BITS) & WORD_MASK
        return (TAG_FLOAT, struct.unpack(FLOAT_FORMAT, struct.pack(WORD_FORMAT, bits))[0])

    return (TAG_STR, read_string(data, payload))


def read_word(data: bytes, offset: int) -> int:
    return struct.unpack_from(WORD_FORMAT, data, offset)[0]


def read_tables(data: bytes) -> ModuleTables:
    magic, entry_offset, function_count, global_offset, global_count, dword_14 = struct.unpack_from(HEADER_FORMAT, data, 0)
    if magic != MAGIC:
        raise ImageError(f'bad magic {magic!r}')

    globals_ = []
    for index in range(global_count):
        name_word, gtype = struct.unpack_from(GLOBAL_FORMAT, data, global_offset + GLOBAL_SIZE * index)
        name = decode_word(data, name_word)
        if name[0] != TAG_STR:
            raise ImageError(f'global {index} name is not a string: {name}')

        globals_.append((name[1], gtype))

    functions: Dict[str, FunctionEntry] = {}
    order = []
    for index in range(function_count):
        (offset, param_count, is_common, byte06, default_count, default_offset, flags_offset,
         debug_count, debug_offset, name_hash, name_word) = struct.unpack_from(ENTRY_FORMAT, data, entry_offset + ENTRY_SIZE * index)
        name = decode_word(data, name_word)
        if name[0] != TAG_STR:
            raise ImageError(f'function {index} name is not a string: {name}')

        flags = tuple(read_word(data, flags_offset + WORD_BYTES * slot) for slot in range(param_count))
        stored_defaults = [decode_word(data, read_word(data, default_offset + WORD_BYTES * slot)) for slot in range(default_count)]
        if default_count > param_count:
            raise ImageError(f'{name[1]}: {default_count} defaults for {param_count} parameters')

        # Defaults belong to the trailing parameters
        defaults = (None,) * (param_count - default_count) + tuple(stored_defaults)

        records = []
        for record in range(debug_count):
            func_id, call_type, arg_count, info_offset = struct.unpack_from(DEBUG_FORMAT, data, debug_offset + DEBUG_SIZE * record)
            args = []
            for arg in range(arg_count):
                word, arg_type = struct.unpack_from(DEBUG_ARG_FORMAT, data, info_offset + DEBUG_ARG_SIZE * arg)
                args.append((decode_word(data, word), arg_type))

            callee = NO_CALLEE if func_id == NO_FUNCTION_ID else func_id
            records.append((callee, call_type, tuple(args)))

        order.append(name[1])
        functions[name[1]] = FunctionEntry(
            name = name[1],
            name_hash = name_hash,
            is_common = bool(is_common),
            byte06 = byte06,
            param_flags = flags,
            defaults = defaults,
            debug_records = tuple(records),
        )

    if len(functions) != len(order):
        raise ImageError('duplicate function names')

    return ModuleTables(dword_14 = dword_14, globals = tuple(globals_), function_order = tuple(order), functions = functions)

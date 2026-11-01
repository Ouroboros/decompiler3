'''TVM image: header, string table, function table, code

    magic 'TVM1' | u16 version | u16 global_count | u16 string_count | u16 function_count
    strings:   u16 length, UTF-8 bytes
    functions: u16 name string, u8 param_count, param_count x u8 parameter flags, u32 code offset, u32 code size
    code:      the functions' code, at the recorded offsets
'''

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from .isa import Instruction, Op, decode, encode


MAGIC           = b'TVM1'
HEADER_FORMAT   = '<4sHHHH'
LENGTH_FORMAT   = '<H'
ENTRY_FORMAT    = '<HB'
FLAG_FORMAT     = '<B'
FLAGS_FORMAT    = '<{count}B'
RANGE_FORMAT    = '<II'
ENCODING        = 'utf-8'

# Parameter flags
PARAM_INT       = 0x10
PARAM_STR       = 0x20


@dataclass
class Function:
    name: str
    param_flags: Tuple[int, ...]
    code: bytes
    offset: int = 0

    @property
    def instructions(self) -> List[Instruction]:
        return decode(self.code)


@dataclass
class Image:
    version: int
    global_count: int
    strings: List[str]
    functions: List[Function] = field(default_factory = list)

    def function_index(self, name: str) -> int:
        return next(index for index, func in enumerate(self.functions) if func.name == name)


def read_image(data: bytes) -> Image:
    magic, version, global_count, string_count, function_count = struct.unpack_from(HEADER_FORMAT, data)
    if magic != MAGIC:
        raise ValueError(f'not a TVM image: {magic!r}')

    position = struct.calcsize(HEADER_FORMAT)
    strings = []
    for _ in range(string_count):
        (length,) = struct.unpack_from(LENGTH_FORMAT, data, position)
        position += struct.calcsize(LENGTH_FORMAT)
        strings.append(data[position:position + length].decode(ENCODING))
        position += length

    image = Image(version, global_count, strings)
    for _ in range(function_count):
        name_index, param_count = struct.unpack_from(ENTRY_FORMAT, data, position)
        position += struct.calcsize(ENTRY_FORMAT)
        flags = struct.unpack_from(FLAGS_FORMAT.format(count = param_count), data, position)
        position += struct.calcsize(FLAG_FORMAT) * param_count
        offset, size = struct.unpack_from(RANGE_FORMAT, data, position)
        position += struct.calcsize(RANGE_FORMAT)
        image.functions.append(Function(strings[name_index], tuple(flags), data[offset:offset + size], offset))

    return image


def write_image(image: Image) -> bytes:
    '''Bytes of an image; functions are laid out in order right after the tables'''
    header = struct.pack(HEADER_FORMAT, MAGIC, image.version, image.global_count, len(image.strings), len(image.functions))
    strings = b''.join(struct.pack(LENGTH_FORMAT, len(text.encode(ENCODING))) + text.encode(ENCODING) for text in image.strings)
    entry_size = lambda func: struct.calcsize(ENTRY_FORMAT) + struct.calcsize(FLAG_FORMAT) * len(func.param_flags) + struct.calcsize(RANGE_FORMAT)
    offset = len(header) + len(strings) + sum(entry_size(func) for func in image.functions)
    entries = b''
    for func in image.functions:
        func.offset = offset
        entries += struct.pack(ENTRY_FORMAT, image.strings.index(func.name), len(func.param_flags))
        entries += b''.join(struct.pack(FLAG_FORMAT, flag) for flag in func.param_flags)
        entries += struct.pack(RANGE_FORMAT, offset, len(func.code))
        offset += len(func.code)

    return header + strings + entries + b''.join(func.code for func in image.functions)


# Assembler: (op, operand, ...) tuples and label names; jump operands may name a label
Line = Union[str, Tuple]


def assemble(lines: List[Line]) -> bytes:
    labels: Dict[str, int] = {}
    position = 0
    for line in lines:
        if isinstance(line, str):
            labels[line] = position

        else:
            position += Instruction(position, line[0], tuple(0 for _ in line[1:])).size

    instructions = []
    position = 0
    for line in lines:
        if isinstance(line, str):
            continue

        operands = tuple(labels[value] if isinstance(value, str) else value for value in line[1:])
        inst = Instruction(position, line[0], operands)
        instructions.append(inst)
        position += inst.size

    return encode(instructions)

'''TVM instruction set: one opcode byte followed by fixed operands (little endian)'''

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple


# Bytes per stack slot; LDL / STL operands are byte offsets from sp
WORD_SIZE = 8


class Op(IntEnum):
    PUSH    = 0x01     # i32 integer
    PUSHS   = 0x02     # u16 string index
    LDL     = 0x03     # i16 byte offset from sp: push that slot
    STL     = 0x04     # i16 byte offset from sp after the pop: pop into that slot
    LDG     = 0x05     # u16 global index
    STG     = 0x06     # u16 global index
    LDR     = 0x07     # u8 register
    STR     = 0x08     # u8 register
    DROP    = 0x09     # u8 slot count
    ADD     = 0x10
    SUB     = 0x11
    MUL     = 0x12
    DIV     = 0x13     # floor division
    MOD     = 0x14     # floor modulo
    EQ      = 0x15
    NE      = 0x16
    LT      = 0x17
    LE      = 0x18
    GT      = 0x19
    GE      = 0x1A
    AND     = 0x1B     # bitwise
    OR      = 0x1C     # bitwise
    NEG     = 0x1E
    NOT     = 0x1F     # bitwise
    JMP     = 0x20     # u32 code offset within the function
    JZ      = 0x21     # u32: pop, jump when zero
    JNZ     = 0x22     # u32: pop, jump when nonzero
    CALL    = 0x23     # u16 function index, u8 argc: pops the arguments, result in RESULT_REGISTER
    HOST    = 0x24     # u16 string index (host function name), u8 argc: like CALL
    RET     = 0x25     # returns RESULT_REGISTER
    LINE    = 0x26     # u16 source line
    LOG     = 0x27     # i32 debug value


OPERAND_FORMATS = {
    Op.PUSH:    '<i',
    Op.PUSHS:   '<H',
    Op.LDL:     '<h',
    Op.STL:     '<h',
    Op.LDG:     '<H',
    Op.STG:     '<H',
    Op.LDR:     '<B',
    Op.STR:     '<B',
    Op.DROP:    '<B',
    Op.JMP:     '<I',
    Op.JZ:      '<I',
    Op.JNZ:     '<I',
    Op.CALL:    '<HB',
    Op.HOST:    '<HB',
    Op.LINE:    '<H',
    Op.LOG:     '<i',
}

OPCODE_FORMAT   = '<B'
JUMP_OPS        = (Op.JMP, Op.JZ, Op.JNZ)
# Instructions after which execution does not simply continue in the same block
BLOCK_END_OPS   = JUMP_OPS + (Op.CALL, Op.RET)


@dataclass
class Instruction:
    offset: int
    op: Op
    operands: Tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return struct.calcsize(OPCODE_FORMAT) + struct.calcsize(OPERAND_FORMATS.get(self.op, '<'))

    def encode(self) -> bytes:
        return struct.pack(OPCODE_FORMAT, self.op) + struct.pack(OPERAND_FORMATS.get(self.op, '<'), *self.operands)


def decode(code: bytes) -> List[Instruction]:
    '''Decode a function's code linearly'''
    instructions = []
    position = 0
    while position < len(code):
        op = Op(code[position])
        operand_format = OPERAND_FORMATS.get(op, '<')
        start = position + struct.calcsize(OPCODE_FORMAT)
        operands = struct.unpack_from(operand_format, code, start)
        instructions.append(Instruction(position, op, operands))
        position = start + struct.calcsize(operand_format)

    return instructions


def encode(instructions: List[Instruction]) -> bytes:
    return b''.join(inst.encode() for inst in instructions)

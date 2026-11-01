"""Instruction and Operand definitions"""

from common import *
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .instruction_table import InstructionDescriptor, OperandDescriptor


@dataclass
class Operand:
    """Generic operand representation"""
    descriptor  : 'OperandDescriptor'
    value       : Any


@dataclass
class Instruction:
    """Generic instruction representation"""
    offset        : int                                             # Offset in bytecode
    opcode        : int                                             # Raw opcode value
    descriptor    : 'InstructionDescriptor'                         # Instruction descriptor from table
    operands      : list[Operand] = field(default_factory = list)
    size          : int = 0                                         # Instruction size in bytes
    # Encoding of a PUSH that decoding turned into a typed pseudo-instruction (PUSH_INT, PUSH_RET_ADDR, ...)
    push_size     : int = 0
    push_value    : Any = None

    @property
    def mnemonic(self) -> str:
        """Instruction mnemonic"""
        return self.descriptor.mnemonic

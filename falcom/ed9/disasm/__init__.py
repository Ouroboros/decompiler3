"""
Falcom ED9 Disassembler

Generic bytecode disassembler framework with pluggable instruction tables.
"""

from .instruction import *
from .basic_block import *
from .instruction_table import *
from .disassembler import *
from .ed9_optable import *

__all__ = [
    'Instruction',
    'Operand',
    'OperandType',
    'BasicBlock',
    'BranchKind',
    'InstructionTable',
    'InstructionDescriptor',
    'InstructionFlags',
    'BranchTarget',
    'Disassembler',
    'ED9Opcode',
    'ED9InstructionTable',
    'ED9_INSTRUCTION_TABLE',
]

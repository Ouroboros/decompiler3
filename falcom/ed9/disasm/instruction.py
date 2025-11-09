"""
Instruction and Operand definitions
"""

from common import *
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .instruction_table import InstructionDescriptor


class OperandType(IntEnum2):
    """Operand types for generic instruction representation"""
    NONE        = 0     # No operand
    BYTE        = 1     # C: 8-bit unsigned
    SHORT       = 2     # H: 16-bit unsigned
    INT         = 3     # i: 32-bit signed
    UINT        = 4     # I: 32-bit unsigned
    FLOAT       = 5     # f: 32-bit float
    STRING      = 6     # S: String offset
    OFFSET      = 7     # O: Code offset (branch target)
    FUNC        = 8     # F: Function name/id
    VALUE       = 9     # V: VM-specific value type (e.g., ScpValue)


@dataclass
class Operand:
    """Generic operand representation"""
    type    : OperandType
    value   : Any

    def __str__(self) -> str:
        match self.type:
            case OperandType.OFFSET:
                return f'loc_{self.value:X}'

            case OperandType.FUNC:
                return f'func_{self.value}'

            case OperandType.STRING:
                return f'"{self.value}"'

            case OperandType.FLOAT:
                return f'{self.value}'

            case _:
                return str(self.value)


@dataclass
class Instruction:
    """Generic instruction representation"""
    offset        : int                                             # Offset in bytecode
    opcode        : int                                             # Raw opcode value
    descriptor    : 'InstructionDescriptor'                         # Instruction descriptor from table
    operands      : list[Operand] = field(default_factory = list)
    size          : int = 0                                         # Instruction size in bytes

    @property
    def mnemonic(self) -> str:
        """Instruction mnemonic"""
        return self.descriptor.mnemonic

    def __str__(self) -> str:
        if not self.operands:
            return self.mnemonic

        ops = ', '.join(str(op) for op in self.operands)
        return f'{self.mnemonic}({ops})'

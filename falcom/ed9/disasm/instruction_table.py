"""
Instruction table interface and descriptor
"""

from common import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .basic_block import BranchKind
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    from ml import fileio
    from .instruction import Instruction, Operand


class OperandType(IntEnum2):
    """Operand types"""
    Empty       = 0
    SInt8       = 1
    SInt16      = 2
    SInt32      = 3
    UInt8       = 4
    UInt16      = 5
    UInt32      = 6
    Float32     = 7
    String      = 8
    Offset      = 9
    UserDefined = 100

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class OperandFormat:
    """Operand format descriptor"""
    sizeTable: dict[OperandType, int] = {
        OperandType.SInt8   : 1,
        OperandType.SInt16  : 2,
        OperandType.SInt32  : 4,
        OperandType.UInt8   : 1,
        OperandType.UInt16  : 2,
        OperandType.UInt32  : 4,
        OperandType.Float32 : 4,
        OperandType.String  : None,
        OperandType.Offset  : 4,
    }

    def __init__(self, oprType: OperandType, is_hex: bool = False):
        self.type   = oprType
        self.is_hex = is_hex

    def __str__(self):
        return repr(self.type)

    def __repr__(self):
        return self.__str__()

    @property
    def size(self):
        return self.sizeTable.get(self.type)


class OperandDescriptor:
    """Operand descriptor with read/write/format methods"""
    formatTable: dict[str, 'OperandDescriptor'] = {}

    @classmethod
    def fromFormatString(cls, fmtstr: str, formatTable: dict = None) -> tuple['OperandDescriptor', ...]:
        formatTable = formatTable if formatTable else cls.formatTable
        return tuple(formatTable[f] for f in fmtstr)

    def __init__(self, format: OperandFormat):
        self.format = format

    @property
    def type(self) -> OperandType:
        return self.format.type

    def read_value(self, fs: 'fileio.FileStream') -> Any:
        """Read operand value from file stream"""
        return {
            OperandType.SInt8   : lambda: fs.ReadChar(),
            OperandType.SInt16  : lambda: fs.ReadShort(),
            OperandType.SInt32  : lambda: fs.ReadLong(),
            OperandType.UInt8   : lambda: fs.ReadByte(),
            OperandType.UInt16  : lambda: fs.ReadUShort(),
            OperandType.UInt32  : lambda: fs.ReadULong(),
            OperandType.Float32 : lambda: fs.ReadFloat(),
            OperandType.Offset  : lambda: fs.ReadULong(),
        }[self.format.type]()

    @staticmethod
    def format_operand(operand: 'Operand') -> str:
        """Format operand for display"""
        return {
            OperandType.SInt8   : OperandDescriptor.format_integer,
            OperandType.SInt16  : OperandDescriptor.format_integer,
            OperandType.SInt32  : OperandDescriptor.format_integer,
            OperandType.UInt8   : OperandDescriptor.format_integer,
            OperandType.UInt16  : OperandDescriptor.format_integer,
            OperandType.UInt32  : OperandDescriptor.format_integer,
            OperandType.Float32 : OperandDescriptor.format_float,
            OperandType.Offset  : OperandDescriptor.format_offset,
        }[operand.descriptor.format.type](operand)

    @staticmethod
    def format_integer(operand: 'Operand') -> str:
        fmt = operand.descriptor.format
        if fmt.is_hex:
            if fmt.type in (OperandType.SInt8, OperandType.SInt16, OperandType.SInt32):
                # Signed integer
                if operand.value < 0:
                    return f'-0x{-operand.value:0{fmt.size * 2}X}'
                else:
                    return f'0x{operand.value:0{fmt.size * 2}X}'
            else:
                # Unsigned integer
                return f'0x{operand.value:0{fmt.size * 2}X}'
        return str(operand.value)

    @staticmethod
    def format_float(operand: 'Operand') -> str:
        return str(operand.value)

    @staticmethod
    def format_offset(operand: 'Operand') -> str:
        return f'loc_{operand.value:X}'


# Initialize format table
def _oprdesc(oprType: OperandType, is_hex: bool = False):
    return OperandDescriptor(OperandFormat(oprType, is_hex=is_hex))

OperandDescriptor.formatTable.update({
    'c' : _oprdesc(OperandType.UInt8, is_hex=False),
    'h' : _oprdesc(OperandType.UInt16, is_hex=False),
    'i' : _oprdesc(OperandType.UInt32, is_hex=False),

    'C' : _oprdesc(OperandType.SInt8, is_hex=False),
    'H' : _oprdesc(OperandType.SInt16, is_hex=False),
    'I' : _oprdesc(OperandType.SInt32, is_hex=False),

    'x' : _oprdesc(OperandType.UInt8, is_hex=True),
    'X' : _oprdesc(OperandType.UInt16, is_hex=True),
    'Y' : _oprdesc(OperandType.UInt32, is_hex=True),

    'f' : _oprdesc(OperandType.Float32),
    'O' : _oprdesc(OperandType.Offset),
})


@dataclass
class BranchTarget:
    """Branch target information"""
    kind   : 'BranchKind'
    offset : int

    @classmethod
    def unconditional(cls, offset: int) -> 'BranchTarget':
        """Create unconditional branch target"""
        return cls(BranchKind.UNCONDITIONAL, offset)

    @classmethod
    def true_branch(cls, offset: int) -> 'BranchTarget':
        """Create true branch target"""
        return cls(BranchKind.TRUE, offset)

    @classmethod
    def false_branch(cls, offset: int) -> 'BranchTarget':
        """Create false branch target"""
        return cls(BranchKind.FALSE, offset)


class InstructionFlags(IntFlag2):
    """Instruction flags (can be combined with bitwise OR)"""
    NONE        = 0
    END_BLOCK   = 1 << 0  # Terminates a basic block
    START_BLOCK = 1 << 1  # Starts a new basic block at target


@dataclass
class InstructionDescriptor:
    """Descriptor for an instruction in the instruction table"""
    opcode: int
    mnemonic: str
    flags: InstructionFlags = InstructionFlags.NONE

    def is_end_block(self) -> bool:
        """Check if instruction terminates a basic block"""
        return bool(self.flags & InstructionFlags.END_BLOCK)

    def is_start_block(self) -> bool:
        """Check if instruction starts a new basic block"""
        return bool(self.flags & InstructionFlags.START_BLOCK)

    def get_branch_targets(self, inst: 'Instruction', current_pos: int) -> list[BranchTarget]:
        """
        Get branch targets from this instruction.

        Returns:
            List of branch targets.
        """
        return []

    def format_operands(self, operands: list['Operand']) -> str:
        """
        Format operands for display.

        Returns:
            Formatted operand string
        """
        return ', '.join(OperandDescriptor.format_operand(op) for op in operands)

    def format_instruction(self, inst: 'Instruction') -> str:
        """
        Format instruction for display.

        Returns:
            Formatted instruction string
        """
        if not inst.operands:
            return self.mnemonic

        ops = self.format_operands(inst.operands)
        return f'{self.mnemonic}({ops})'


class InstructionTable(ABC):
    """
    Abstract interface for instruction tables.

    Subclasses must implement VM-specific instruction decoding.
    """

    @abstractmethod
    def read_opcode(self, fs: 'fileio.FileStream') -> int:
        """Read opcode from file stream"""
        pass

    @abstractmethod
    def get_descriptor(self, opcode: int) -> InstructionDescriptor:
        """Get instruction descriptor for opcode"""
        pass

    @abstractmethod
    def read_operands(
        self,
        fs: 'fileio.FileStream',
        inst: 'Instruction',
        offset: int
    ) -> list['Operand']:
        """
        Read operands for instruction.

        Args:
            fs: File stream positioned after opcode
            inst: Instruction being decoded (may be modified)
            offset: Instruction offset (for relative addressing)

        Returns:
            List of operands
        """
        pass

    def decode_instruction(
        self,
        fs: 'fileio.FileStream',
        offset: int
    ) -> 'Instruction':
        """
        Decode a complete instruction.

        Args:
            fs: File stream positioned at instruction start
            offset: Instruction offset

        Returns:
            Decoded instruction
        """
        from .instruction import Instruction

        # Read opcode
        opcode = self.read_opcode(fs)

        # Get descriptor
        descriptor = self.get_descriptor(opcode)

        # Create instruction
        inst = Instruction(
            offset      = offset,
            opcode      = opcode,
            descriptor  = descriptor,
            operands    = [],
            size        = 0
        )

        # Read operands (may modify inst.opcode and inst.descriptor)
        operands = self.read_operands(fs, inst, offset)

        # Update instruction
        inst.operands = operands
        inst.size = fs.Position - offset

        return inst

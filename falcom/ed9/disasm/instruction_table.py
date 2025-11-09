"""
Instruction table interface and descriptor
"""

from common import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .basic_block import BranchKind
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml import fileio
    from .instruction import Instruction, Operand


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
        descriptor: InstructionDescriptor,
        offset: int
    ) -> list['Operand']:
        """
        Read operands for instruction.

        Args:
            fs: File stream positioned after opcode
            descriptor: Instruction descriptor
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

        # Read operands
        operands = self.read_operands(fs, descriptor, offset)

        # Calculate size
        size = fs.Position - offset

        return Instruction(
            offset      = offset,
            opcode      = opcode,
            descriptor  = descriptor,
            operands    = operands,
            size        = size
        )

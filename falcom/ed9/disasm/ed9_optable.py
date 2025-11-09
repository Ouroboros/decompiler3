"""
ED9 (Kuro no Kiseki) Instruction Table

Based on Decompiler2/Falcom/ED9/InstructionTable/scena.py
"""

from common import *
from ml import fileio

from .instruction_table import *
from .instruction import *
from .basic_block import *
from ..parser.types_scp import *


@dataclass
class InstructionEntry:
    """Single instruction table entry"""
    opcode: int
    mnemonic: str
    operand_fmt: str = ''
    flags: InstructionFlags = InstructionFlags.NONE


# ED9 Instruction Table
ED9_OPCODE_TABLE = [
    # Stack operations
    InstructionEntry(0x00, 'PUSH',                    'V'),       # Special: decoded to PUSH_INT/PUSH_FLOAT/PUSH_STR
    InstructionEntry(0x01, 'POP',                     'C'),
    InstructionEntry(0x02, 'LOAD_STACK',              'i'),
    InstructionEntry(0x03, 'LOAD_STACK_DEREF',        'i'),
    InstructionEntry(0x04, 'PUSH_STACK_OFFSET',       'i'),
    InstructionEntry(0x05, 'POP_TO',                  'i'),
    InstructionEntry(0x06, 'POP_TO_DEREF',            'i'),

    # Global variables
    InstructionEntry(0x07, 'LOAD_GLOBAL',             'i'),
    InstructionEntry(0x08, 'SET_GLOBAL',              'i'),

    # Registers
    InstructionEntry(0x09, 'GET_REG',                 'C'),
    InstructionEntry(0x0A, 'SET_REG',                 'C'),

    # Control flow
    InstructionEntry(0x0B, 'JMP',                     'O',   InstructionFlags.END_BLOCK),
    InstructionEntry(0x0C, 'CALL',                    'F',   InstructionFlags.END_BLOCK),
    InstructionEntry(0x0D, 'RETURN',                  '',    InstructionFlags.END_BLOCK),
    InstructionEntry(0x0E, 'POP_JMP_NOT_ZERO',        'O',   InstructionFlags.END_BLOCK),
    InstructionEntry(0x0F, 'POP_JMP_ZERO',            'O',   InstructionFlags.END_BLOCK),

    # Arithmetic
    InstructionEntry(0x10, 'ADD'),
    InstructionEntry(0x11, 'SUB'),
    InstructionEntry(0x12, 'MUL'),
    InstructionEntry(0x13, 'DIV'),
    InstructionEntry(0x14, 'MOD'),

    # Comparison
    InstructionEntry(0x15, 'EQ'),
    InstructionEntry(0x16, 'NE'),
    InstructionEntry(0x17, 'GT'),
    InstructionEntry(0x18, 'GE'),
    InstructionEntry(0x19, 'LT'),
    InstructionEntry(0x1A, 'LE'),

    # Bitwise & Logical
    InstructionEntry(0x1B, 'BITWISE_AND'),
    InstructionEntry(0x1C, 'BITWISE_OR'),
    InstructionEntry(0x1D, 'LOGICAL_AND'),
    InstructionEntry(0x1E, 'LOGICAL_OR'),

    # Unary
    InstructionEntry(0x1F, 'NEG'),
    InstructionEntry(0x20, 'EZ'),
    InstructionEntry(0x21, 'NOT'),

    # Script calls
    InstructionEntry(0x22, 'CALL_SCRIPT',             'VVC', InstructionFlags.END_BLOCK),
    InstructionEntry(0x23, 'CALL_SCRIPT_NO_RETURN',   'VVC', InstructionFlags.END_BLOCK),

    # System
    InstructionEntry(0x24, 'SYSCALL',                 'CBB'),
    InstructionEntry(0x25, 'PUSH_CALLER_FRAME',       'O',   InstructionFlags.START_BLOCK),
    InstructionEntry(0x26, 'DEBUG_SET_LINENO',        'H'),
    InstructionEntry(0x27, 'POPN',                    'C'),
    InstructionEntry(0x28, 'DEBUG_LOG',               'L'),
]

# Helper to lookup opcode from table
def _opcode(mnemonic: str) -> int:
    for entry in ED9_OPCODE_TABLE:
        if entry.mnemonic == mnemonic:
            return entry.opcode
    raise ValueError(f'Unknown mnemonic: {mnemonic}')

class ED9Opcode(IntEnum2):
    """ED9 VM Opcodes"""
    # Stack operations
    PUSH                    = _opcode('PUSH')
    POP                     = _opcode('POP')
    LOAD_STACK              = _opcode('LOAD_STACK')
    LOAD_STACK_DEREF        = _opcode('LOAD_STACK_DEREF')
    PUSH_STACK_OFFSET       = _opcode('PUSH_STACK_OFFSET')
    POP_TO                  = _opcode('POP_TO')
    POP_TO_DEREF            = _opcode('POP_TO_DEREF')

    # Global variables
    LOAD_GLOBAL             = _opcode('LOAD_GLOBAL')
    SET_GLOBAL              = _opcode('SET_GLOBAL')

    # Registers
    GET_REG                 = _opcode('GET_REG')
    SET_REG                 = _opcode('SET_REG')

    # Control flow
    JMP                     = _opcode('JMP')
    CALL                    = _opcode('CALL')
    RETURN                  = _opcode('RETURN')
    POP_JMP_NOT_ZERO        = _opcode('POP_JMP_NOT_ZERO')
    POP_JMP_ZERO            = _opcode('POP_JMP_ZERO')

    # Arithmetic
    ADD                     = _opcode('ADD')
    SUB                     = _opcode('SUB')
    MUL                     = _opcode('MUL')
    DIV                     = _opcode('DIV')
    MOD                     = _opcode('MOD')

    # Comparison
    EQ                      = _opcode('EQ')
    NE                      = _opcode('NE')
    GT                      = _opcode('GT')
    GE                      = _opcode('GE')
    LT                      = _opcode('LT')
    LE                      = _opcode('LE')

    # Bitwise & Logical
    BITWISE_AND             = _opcode('BITWISE_AND')
    BITWISE_OR              = _opcode('BITWISE_OR')
    LOGICAL_AND             = _opcode('LOGICAL_AND')
    LOGICAL_OR              = _opcode('LOGICAL_OR')

    # Unary
    NEG                     = _opcode('NEG')
    EZ                      = _opcode('EZ')
    NOT                     = _opcode('NOT')

    # Script calls
    CALL_SCRIPT             = _opcode('CALL_SCRIPT')
    CALL_SCRIPT_NO_RETURN   = _opcode('CALL_SCRIPT_NO_RETURN')

    # System
    SYSCALL                 = _opcode('SYSCALL')
    PUSH_CALLER_FRAME       = _opcode('PUSH_CALLER_FRAME')
    DEBUG_SET_LINENO        = _opcode('DEBUG_SET_LINENO')
    POPN                    = _opcode('POPN')
    DEBUG_LOG               = _opcode('DEBUG_LOG')


class ED9InstructionDescriptor(InstructionDescriptor):
    """ED9-specific instruction descriptor with operand format"""

    def __init__(
        self,
        opcode: int,
        mnemonic: str,
        operand_fmt: str = '',
        flags: InstructionFlags = InstructionFlags.NONE
    ):
        super().__init__(opcode, mnemonic, flags)
        self.operand_fmt = operand_fmt

    def get_branch_targets(self, inst, current_pos: int) -> list[BranchTarget]:
        """Extract branch targets from instruction"""
        targets = []

        if self.opcode == ED9Opcode.JMP:
            # Unconditional jump
            target = self._get_offset_operand(inst)
            if target is not None:
                targets.append(BranchTarget.unconditional(target))

        elif self.opcode in (ED9Opcode.POP_JMP_ZERO, ED9Opcode.POP_JMP_NOT_ZERO):
            # Conditional jump
            target = self._get_offset_operand(inst)
            if target is not None:
                targets.append(BranchTarget.true_branch(target))
                targets.append(BranchTarget.false_branch(current_pos))

        elif self.opcode == ED9Opcode.PUSH_CALLER_FRAME:
            # Return address
            target = self._get_offset_operand(inst)
            if target is not None:
                targets.append(BranchTarget.unconditional(target))

        return targets

    def _get_offset_operand(self, inst) -> int | None:
        """Extract offset operand from instruction"""
        for op in inst.operands:
            if op.type == OperandType.OFFSET:
                return op.value
        return None


class ED9InstructionTable(InstructionTable):
    """ED9 VM Instruction Table"""

    def __init__(self):
        self.descriptors: dict[int, ED9InstructionDescriptor] = {}
        self._init_table()

    def _init_table(self):
        """Initialize instruction table from ED9_OPCODE_TABLE"""
        for entry in ED9_OPCODE_TABLE:
            desc = ED9InstructionDescriptor(
                opcode       = entry.opcode,
                mnemonic     = entry.mnemonic,
                operand_fmt  = entry.operand_fmt,
                flags        = entry.flags
            )
            self.descriptors[entry.opcode] = desc

    def read_opcode(self, fs: fileio.FileStream) -> int:
        """Read opcode byte"""
        return fs.ReadByte()

    def get_descriptor(self, opcode: int) -> ED9InstructionDescriptor:
        """Get instruction descriptor"""
        if opcode not in self.descriptors:
            raise ValueError(f'Unknown opcode: 0x{opcode:02X}')
        return self.descriptors[opcode]

    def read_operands(
        self,
        fs: fileio.FileStream,
        descriptor: InstructionDescriptor,
        offset: int
    ) -> list[Operand]:
        """Read operands according to format string"""
        if not isinstance(descriptor, ED9InstructionDescriptor):
            raise TypeError('Expected ED9InstructionDescriptor')

        operands = []
        fmt = descriptor.operand_fmt

        for ch in fmt:
            op = self._read_operand(fs, ch)
            operands.append(op)

        # Special handling for PUSH instruction
        if descriptor.opcode == ED9Opcode.PUSH:
            operands = self._decode_push(descriptor, operands)

        return operands

    def _read_operand(self, fs: fileio.FileStream, fmt: str) -> Operand:
        """Read single operand based on format character"""
        match fmt:
            case 'C':  # Byte
                return Operand(OperandType.BYTE, fs.ReadByte())

            case 'H':  # Short
                return Operand(OperandType.SHORT, fs.ReadUShort())

            case 'i':  # Signed int
                return Operand(OperandType.INT, fs.ReadLong())

            case 'I':  # Unsigned int
                return Operand(OperandType.UINT, fs.ReadULong())

            case 'f':  # Float
                return Operand(OperandType.FLOAT, fs.ReadFloat())

            case 'O':  # Offset (code address)
                return Operand(OperandType.OFFSET, fs.ReadULong())

            case 'F':  # Function ID
                return Operand(OperandType.FUNC, fs.ReadULong())

            case 'V':  # ScpValue
                value = ScpValue(fs = fs)
                return Operand(OperandType.VALUE, value)

            case 'S':  # String offset
                return Operand(OperandType.STRING, fs.ReadULong())

            case 'L':  # Log string (same as S)
                return Operand(OperandType.STRING, fs.ReadULong())

            case _:
                raise ValueError(f'Unknown operand format: {fmt}')

    def _decode_push(
        self,
        descriptor: ED9InstructionDescriptor,
        operands: list[Operand]
    ) -> list[Operand]:
        """Decode PUSH instruction to specific variant based on value type"""
        if not operands or operands[0].type != OperandType.VALUE:
            return operands

        value = operands[0].value
        if not isinstance(value, ScpValue):
            return operands

        # Change mnemonic based on value type
        match value.type:
            case ScpValue.Type.Integer:
                descriptor.mnemonic = 'PUSH_INT'
                return [Operand(OperandType.INT, value.value)]

            case ScpValue.Type.Float:
                descriptor.mnemonic = 'PUSH_FLOAT'
                return [Operand(OperandType.FLOAT, value.value)]

            case ScpValue.Type.String:
                descriptor.mnemonic = 'PUSH_STR'
                return [Operand(OperandType.STRING, value.value)]

        return operands


# Global instruction table instance
ED9_INSTRUCTION_TABLE = ED9InstructionTable()

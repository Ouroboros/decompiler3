"""
ED9 (Kuro no Kiseki) Instruction Table

Based on Decompiler2/Falcom/ED9/InstructionTable/scena.py
"""

from common import *
from ml import fileio
from typing import TYPE_CHECKING

from .instruction_table import *
from .instruction import *
from .basic_block import *
from ..parser.types_scp import *

if TYPE_CHECKING:
    from .formatter import FormatterContext

# ED9-specific operand types
class ED9OperandType(IntEnum2):
    """ED9-specific operand types extending base OperandType"""
    Func    = OperandType.UserDefined + 1  # Function ID
    Value   = OperandType.UserDefined + 2  # ScpValue

# ED9 operand descriptor with extended functionality
class ED9OperandDescriptor(OperandDescriptor):
    """ED9-specific operand descriptor"""

    def read_value(self, fs: fileio.FileStream) -> Any:
        """Read ED9-specific operand values"""
        match self.format.type:
            case ED9OperandType.Func:
                return fs.ReadUShort()

            case ED9OperandType.Value:
                return ScpValue(fs=fs)

            case _:
                return super().read_value(fs)

    def format_operand(self, operand: Operand, context: 'FormatterContext') -> str:
        """Format ED9-specific operands"""
        match self.format.type:
            case ED9OperandType.Func:
                # Try to get function name from context
                if context.get_func_name:
                    func_name = context.get_func_name(operand.value)
                    if func_name:
                        return func_name
                return f'func_{operand.value}'

            case ED9OperandType.Value:
                return str(operand.value)

            case OperandType.String:
                return f'"{operand.value}"'

            case _:
                return super().format_operand(operand, context)

# ED9 operand descriptor factory
def _ed9_oprdesc(opr_type: OperandType | ED9OperandType, is_hex: bool = False):
    return ED9OperandDescriptor(OperandFormat(opr_type, is_hex=is_hex))

# Extend format table with ED9-specific types
ED9_FORMAT_TABLE = OperandDescriptor.format_table.copy()
ED9_FORMAT_TABLE.update({
    'F' : _ed9_oprdesc(ED9OperandType.Func),
    'V' : _ed9_oprdesc(ED9OperandType.Value),
    'S' : _ed9_oprdesc(OperandType.String),
    'L' : _ed9_oprdesc(OperandType.String),
})

if TYPE_CHECKING:
    from .disassembler import DisassemblerContext




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
    InstructionEntry(0x00, 'PUSH',                    'CV'),       # Special: decoded to PUSH_INT/PUSH_FLOAT/PUSH_STR
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

    # Pseudo-instructions (optimized from other instructions)
    InstructionEntry(0x1000, 'PUSH_CURRENT_FUNC_ID',  ''),
    InstructionEntry(0x1001, 'PUSH_RET_ADDR',         'O'),
    InstructionEntry(0x1002, 'PUSH_RAW',              'I'),
    InstructionEntry(0x1003, 'PUSH_INT',              'I'),
    InstructionEntry(0x1004, 'PUSH_FLOAT',            'f'),
    InstructionEntry(0x1005, 'PUSH_STR',              'S'),
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

    # Pseudo-instructions
    PUSH_CURRENT_FUNC_ID    = _opcode('PUSH_CURRENT_FUNC_ID')
    PUSH_RET_ADDR           = _opcode('PUSH_RET_ADDR')
    PUSH_RAW                = _opcode('PUSH_RAW')
    PUSH_INT                = _opcode('PUSH_INT')
    PUSH_FLOAT              = _opcode('PUSH_FLOAT')
    PUSH_STR                = _opcode('PUSH_STR')


class ED9InstructionDescriptor(InstructionDescriptor):
    """ED9-specific instruction descriptor with operand format"""
    _allow_creation = True

    def __init__(
        self,
        opcode: int,
        mnemonic: str,
        operand_fmt: str = '',
        flags: InstructionFlags = InstructionFlags.NONE
    ):
        if not ED9InstructionDescriptor._allow_creation:
            raise RuntimeError('Cannot create ED9InstructionDescriptor after ED9InstructionTable initialization')
        super().__init__(opcode, mnemonic, flags)
        self.operand_fmt = operand_fmt

    def format_operands(self, operands: list[Operand], context: 'FormatterContext') -> list[str]:
        """Format operands for display (ED9-specific)"""
        return [op.descriptor.format_operand(op, context) for op in operands]

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
            if op.descriptor.format.type == OperandType.Offset:
                return op.value
        return None



class ED9InstructionTable(InstructionTable):
    """ED9 VM Instruction Table"""

    def __init__(self):
        self.descriptors: dict[int, ED9InstructionDescriptor] = {}
        self._init_table()
        # Disable further descriptor creation
        ED9InstructionDescriptor._allow_creation = False

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
        fs          : fileio.FileStream,
        inst        : Instruction,
        offset      : int
    ) -> list[Operand]:
        """Read operands according to format string"""
        descriptor = inst.descriptor
        if not isinstance(descriptor, ED9InstructionDescriptor):
            raise TypeError('Expected ED9InstructionDescriptor')

        # Get operand descriptors from format string
        operand_descriptors = OperandDescriptor.from_format_string(descriptor.operand_fmt, ED9_FORMAT_TABLE)

        operands = []
        for op_desc in operand_descriptors:
            value = op_desc.read_value(fs)
            operands.append(Operand(descriptor=op_desc, value=value))

        # Special handling for PUSH instruction
        if descriptor.opcode == ED9Opcode.PUSH:
            self._decode_push(inst, operands)
            return inst.operands

        return operands

    def _decode_push(self, inst: Instruction, operands: list[Operand]):
        """Decode PUSH instruction to specific variant based on value type"""

        operand_size = operands[0]
        operand_value = operands[1]

        if operand_size.descriptor.format.type not in (OperandType.SInt8, OperandType.UInt8):
            raise ValueError(f'Expected SInt8 operand for size, got {operand_size.descriptor.format.type}')

        if operand_value.descriptor.format.type != ED9OperandType.Value:
            raise ValueError(f'Expected Value operand, got {operand_value.descriptor.format.type}')

        value = operand_value.value
        if not isinstance(value, ScpValue):
            raise ValueError(f'Expected ScpValue, got {type(value)}')

        # Helper to update instruction with new opcode and operand
        def update_push_inst(opcode: int, operand_value: Any):
            inst.opcode = opcode
            inst.descriptor = self.get_descriptor(opcode)
            # Get operand descriptor from instruction descriptor
            op_desc = OperandDescriptor.from_format_string(inst.descriptor.operand_fmt, ED9_FORMAT_TABLE)[0]
            inst.operands = [Operand(descriptor=op_desc, value=operand_value)]

        # Replace descriptor and operands based on value type
        match value.type:
            case ScpValue.Type.Raw:
                update_push_inst(ED9Opcode.PUSH_RAW, value.value)

            case ScpValue.Type.Integer:
                update_push_inst(ED9Opcode.PUSH_INT, value.value)

            case ScpValue.Type.Float:
                update_push_inst(ED9Opcode.PUSH_FLOAT, value.value)

            case ScpValue.Type.String:
                update_push_inst(ED9Opcode.PUSH_STR, value.value)


def ed9_create_fallthrough_jump(offset: int, target: int, inst_table: 'InstructionTable') -> Instruction:
    """
    Create a synthetic JMP instruction for split blocks.

    Args:
        offset: Instruction offset (split point)
        target: Jump target offset
        inst_table: Instruction table

    Returns:
        Synthetic JMP instruction
    """

    # Get JMP descriptor from instruction table
    jmp_descriptor = inst_table.get_descriptor(ED9Opcode.JMP)

    # Create offset operand from instruction descriptor
    op_desc = OperandDescriptor.from_format_string(jmp_descriptor.operand_fmt, ED9_FORMAT_TABLE)[0]
    operand = Operand(descriptor = op_desc, value = target)

    # Create synthetic instruction
    # Use offset - 1 to avoid collision with tail block's first instruction
    # Use size = 0 to indicate this is synthetic (not in original bytecode)
    synthetic_inst = Instruction(
        offset     = offset - 1,
        opcode     = jmp_descriptor.opcode,
        descriptor = jmp_descriptor,
        operands   = [operand],
        size       = 0  # Synthetic instruction, no actual bytes
    )

    return synthetic_inst


def ed9_optimize_instruction(context: 'DisassemblerContext', current_inst: Instruction, block: BasicBlock) -> list[BranchTarget]:
    """
    ED9 instruction optimization callback

    Args:
        context: Disassembler context with callbacks
        current_inst: Current instruction just decoded
        block: Current basic block

    Called for every instruction to optimize patterns.
    - Optimizes instructions in-place (modifies block.instructions)
    - Returns branch targets only when current_inst is END_BLOCK

    Optimizes instruction patterns:
    - CALL pattern: PUSH(args...), PUSH(func_id), PUSH(ret_addr), CALL(func_id)

    Returns:
        List of branch targets (only when current_inst ends block)
    """
    # Check if we're in a new block, clear stack simulation if so
    # if context.current_block_offset != block.offset:
    #     context.current_block_offset = block.offset
    #     context.stack_simulation.clear()

    # Simulate stack operations
    _simulate_stack_operation(context, current_inst)

    # Only optimize and return targets when we hit CALL
    if current_inst.opcode == ED9Opcode.CALL:
        return _optimize_call_pattern(context, current_inst)

    return []


# Initialize stack simulation instruction groups
PUSH_VARIANTS = (
    ED9Opcode.PUSH_INT,
    ED9Opcode.PUSH_RAW,
    ED9Opcode.PUSH_FLOAT,
    ED9Opcode.PUSH_STR,
)

PUSH_VALUE_OPS = (
    ED9Opcode.GET_REG,
    ED9Opcode.LOAD_GLOBAL,
    ED9Opcode.LOAD_STACK,
    ED9Opcode.LOAD_STACK_DEREF,
)

POP_VALUE_OPS = (
    ED9Opcode.SET_REG,
    ED9Opcode.SET_GLOBAL,
    ED9Opcode.POP_TO,
    ED9Opcode.POP_TO_DEREF,
)

BINARY_OPS = (
    ED9Opcode.ADD,
    ED9Opcode.SUB,
    ED9Opcode.MUL,
    ED9Opcode.DIV,
    ED9Opcode.MOD,
    ED9Opcode.EQ,
    ED9Opcode.NE,
    ED9Opcode.GT,
    ED9Opcode.GE,
    ED9Opcode.LT,
    ED9Opcode.LE,
    ED9Opcode.BITWISE_AND,
    ED9Opcode.BITWISE_OR,
    ED9Opcode.LOGICAL_AND,
    ED9Opcode.LOGICAL_OR,
)

UNARY_OPS = (
    ED9Opcode.NEG,
    ED9Opcode.EZ,
    ED9Opcode.NOT,
)

CONDITIONAL_JUMPS = (
    ED9Opcode.POP_JMP_ZERO,
    ED9Opcode.POP_JMP_NOT_ZERO,
)


def _simulate_stack_operation(context: 'DisassemblerContext', inst: Instruction):
    """
    Simulate stack operations to track which PUSH instruction produced each stack value

    Updates context.stack_simulation with (instruction, stack_depth) pairs
    """
    stack = context.stack_simulation
    opcode = inst.opcode

    # PUSH operations - add to stack
    if opcode in PUSH_VARIANTS:
        stack.append(inst)

    # GET_REG, LOAD_GLOBAL, LOAD_STACK, etc - push value
    elif opcode in PUSH_VALUE_OPS:
        stack.append(inst)

    # POP operations - remove from stack
    elif opcode == ED9Opcode.POP:
        count = inst.operands[0].value if inst.operands else 1
        for _ in range(count):
            stack.pop()

    elif opcode == ED9Opcode.POPN:
        count = inst.operands[0].value if inst.operands else 0
        for _ in range(count):
            stack.pop()

    # SET_REG, SET_GLOBAL, POP_TO, etc - pop value
    elif opcode in POP_VALUE_OPS:
        stack.pop()

    # Binary operations - pop 2, push 1
    elif opcode in BINARY_OPS:
        stack.pop()
        stack.pop()
        stack.append(inst)  # Result of operation

    # Unary operations - pop 1, push 1
    elif opcode in UNARY_OPS:
        stack.pop()
        stack.append(inst)  # Result of operation

    # Conditional jumps - pop 1
    elif opcode in CONDITIONAL_JUMPS:
        stack.pop()


def _optimize_call_pattern(context: 'DisassemblerContext', call_inst: Instruction) -> list[BranchTarget]:
    """
    Optimize CALL pattern: detect PUSH_RET_ADDR and PUSH_CURRENT_FUNC_ID

    Pattern: PUSH(func_id), PUSH(ret_addr), PUSH(arg1), ..., PUSH(argN), CALL(func_id)
    - Get argc from context.get_func_argc(func_id)
    - Pop argc arguments from stack, then ret_addr and func_id should be on top

    Args:
        context: Disassembler context with stack_simulation
        call_inst: CALL instruction

    Returns:
        List of branch targets from PUSH_RET_ADDR
    """
    stack = context.stack_simulation

    if len(stack) < 2:
        return []

    func_id = call_inst.operands[0].value
    argc = context.get_func_argc(func_id)

    # Pattern: PUSH(func_id), PUSH(ret_addr), PUSH(arg1), ..., PUSH(argN), CALL
    # Stack (top to bottom): argN, ..., arg1, ret_addr, func_id
    # We need at least (argc + 2) values on stack
    if len(stack) < argc + 2:
        return []

    targets = []
    PUSH_VARIANTS = (ED9Opcode.PUSH_INT, ED9Opcode.PUSH_RAW)

    # Pop argc arguments from stack simulation to get ret_addr and func_id
    # Stack index from end: -1 is top (last arg), -(argc+1) is ret_addr, -(argc+2) is func_id
    ret_addr_inst = stack[-(argc + 1)]
    func_id_inst = stack[-(argc + 2)]

    # Check if these are PUSH instructions (not results of operations)
    if not isinstance(ret_addr_inst, Instruction) or not isinstance(func_id_inst, Instruction):
        return []

    # Optimize func_id PUSH to PUSH_CURRENT_FUNC_ID
    if func_id_inst.opcode in PUSH_VARIANTS:
        func_id_inst.opcode = ED9Opcode.PUSH_CURRENT_FUNC_ID
        func_id_inst.descriptor = context.instruction_table.get_descriptor(ED9Opcode.PUSH_CURRENT_FUNC_ID)
        func_id_inst.operands.clear()

    # Optimize ret_addr PUSH to PUSH_RET_ADDR (if not already optimized)
    if ret_addr_inst.opcode in PUSH_VARIANTS:
        ret_addr = ret_addr_inst.operands[0].value
        ret_addr_inst.opcode = ED9Opcode.PUSH_RET_ADDR
        ret_addr_inst.descriptor = context.instruction_table.get_descriptor(ED9Opcode.PUSH_RET_ADDR)
        # Change operand to OFFSET from instruction descriptor
        op_desc = OperandDescriptor.from_format_string(ret_addr_inst.descriptor.operand_fmt, ED9_FORMAT_TABLE)[0]
        ret_addr_inst.operands = [Operand(descriptor=op_desc, value = ret_addr)]
        # Create branch target for return address
        targets.append(BranchTarget.unconditional(int(ret_addr)))

    return targets


# Global instruction table instance
ED9_INSTRUCTION_TABLE = ED9InstructionTable()

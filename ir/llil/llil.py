'''
Low Level IL v2 - Optimized Stack-based Design
Following the confirmed VM semantics with layered architecture
'''

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, TYPE_CHECKING
from enum import IntEnum

from ir.core import *

if TYPE_CHECKING:
    # Forward declaration for type hints
    pass

# Constants
WORD_SIZE = 4  # 4 bytes per word


class LowLevelILOperation(IntEnum):
    '''Atomic LLIL operations'''

    # Stack operations (atomic)
    LLIL_STACK_STORE = 0        # STACK[sp + offset] = value
    LLIL_STACK_LOAD = 1         # load STACK[sp + offset]
    LLIL_SP_ADD = 2             # sp = sp + delta

    # Frame operations (relative to frame base, for function parameters/locals)
    LLIL_FRAME_LOAD = 5         # load STACK[frame + offset]
    LLIL_FRAME_STORE = 6        # STACK[frame + offset] = value

    # Register operations
    LLIL_REG_STORE = 10         # R[index] = value
    LLIL_REG_LOAD = 11          # load R[index]

    # Arithmetic operations (stack-based)
    LLIL_ADD = 20               # binary operation on stack
    LLIL_SUB = 21
    LLIL_MUL = 22
    LLIL_DIV = 23
    LLIL_EQ = 24
    LLIL_NE = 25
    LLIL_LT = 26
    LLIL_LE = 27
    LLIL_GT = 28
    LLIL_GE = 29
    LLIL_AND = 30               # bitwise AND
    LLIL_OR = 31                # bitwise OR
    LLIL_LOGICAL_AND = 32       # logical AND (&&)
    LLIL_LOGICAL_OR = 33        # logical OR (||)

    # Unary operations
    LLIL_NEG = 35               # arithmetic negation (-x)
    LLIL_NOT = 36               # logical NOT (!x)
    LLIL_TEST_ZERO = 37         # test if zero (x == 0)

    # Control flow
    LLIL_JMP = 40               # unconditional jump
    LLIL_BRANCH = 41            # conditional branch
    LLIL_CALL = 42              # function call
    LLIL_RET = 43               # return

    # Constants and special
    LLIL_CONST = 50             # constant value
    LLIL_LABEL = 51             # label
    LLIL_NOP = 52               # no operation

    # VM specific
    LLIL_SYSCALL = 60           # system call
    LLIL_DEBUG = 61             # debug info
    LLIL_STACK_ADDR = 62        # address of stack location (sp + offset)

    # Falcom VM specific (user-defined extensions)
    LLIL_PUSH_CALLER_FRAME = 1000  # Falcom VM: push caller frame (4 values)
    LLIL_CALL_SCRIPT = 1001        # Falcom VM: call script function

    # User-defined extensions (reserved range: 1002+)
    LLIL_USER_DEFINED = 1002    # Start of user-defined operations


class LowLevelILInstruction(ILInstruction):
    '''Base class for all LLIL instructions'''

    def __init__(self, operation: LowLevelILOperation):
        super().__init__()
        self.address = 0
        self.inst_index = 0
        self.operation = operation
        self.options = ILOptions()  # Formatting and processing options

    @property
    def operation_name(self) -> str:
        '''Get operation name without LLIL_ prefix (e.g., 'LLIL_ADD' -> 'ADD')'''
        return self.operation.name.replace('LLIL_', '')

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {str(self)}>'


# === Instruction Categories (following BinaryNinja design) ===

class LowLevelILExpr(LowLevelILInstruction):
    '''Base class for value expressions (produces a value)

    Expressions can be used as operands to other instructions.
    Examples: CONST, StackLoad, FrameLoad, BinaryOp, UnaryOp

    vstack only stores LowLevelILExpr instances.
    '''
    pass


class LowLevelILStatement(LowLevelILInstruction):
    '''Base class for statements (side effects, no value)

    Statements have side effects but do not produce values.
    Examples: StackStore, FrameStore, SpAdd, PUSH_CALLER_FRAME

    Basic blocks only store LowLevelILStatement instances.
    '''
    pass


# === Atomic Stack Operations ===

class LowLevelILStackStore(LowLevelILStatement):
    '''STACK[sp + offset] = value (statement)

    Note: offset is in bytes, but displayed as word offset (offset // WORD_SIZE)
    '''

    def __init__(self, value: Union['LowLevelILInstruction', int, str], offset: int = 0, slot_index: int = None):
        super().__init__(LowLevelILOperation.LLIL_STACK_STORE)
        self.value = value
        self.offset = offset  # Byte offset
        self.slot_index = slot_index  # Absolute stack slot index

    def __str__(self) -> str:
        # Convert byte offset to word offset for display
        word_offset = self.offset // WORD_SIZE
        if word_offset == 0:
            return f'STACK[sp] = {self.value}'
        elif word_offset > 0:
            return f'STACK[sp + {word_offset}] = {self.value}'
        else:
            return f'STACK[sp - {-word_offset}] = {self.value}'


class LowLevelILStackLoad(LowLevelILExpr):
    '''load STACK[sp + offset] (expression)

    Note: offset is in bytes, but displayed as word offset (offset // WORD_SIZE)
    '''

    def __init__(self, offset: int, slot_index: int):
        super().__init__(LowLevelILOperation.LLIL_STACK_LOAD)
        self.offset = offset  # Byte offset
        self.slot_index = slot_index  # Absolute stack slot index

    def __str__(self) -> str:
        # Convert byte offset to word offset for display
        word_offset = self.offset // WORD_SIZE
        if word_offset == 0:
            return f'STACK[sp<{self.slot_index}>]'
        elif word_offset > 0:
            return f'STACK[sp + {word_offset}<{self.slot_index}>]'
        else:
            return f'STACK[sp - {-word_offset}<{self.slot_index}>]'


class LowLevelILFrameLoad(LowLevelILExpr):
    '''load STACK[frame + offset] (expression)

    Frame-relative load for function parameters and local variables.
    frame = sp at function entry.

    Function parameters are at negative offsets (pushed by caller):
      STACK[frame - 8] = arg1
      STACK[frame - 7] = arg2
      ...
      STACK[frame - 1] = argN

    Note: offset is in bytes, converted to word offset for display
    '''

    def __init__(self, offset: int = 0):
        super().__init__(LowLevelILOperation.LLIL_FRAME_LOAD)
        self.offset = offset  # Byte offset relative to frame

    def __str__(self) -> str:
        # Convert byte offset to word offset for display
        # Always show explicit offset: fp + 0, fp + 1, etc.
        word_offset = self.offset // WORD_SIZE
        if word_offset >= 0:
            return f'STACK[fp + {word_offset}]'
        else:
            return f'STACK[fp - {-word_offset}]'


class LowLevelILFrameStore(LowLevelILStatement):
    '''STACK[frame + offset] = value (statement)

    Frame-relative store for function parameters and local variables.
    frame = sp at function entry.

    Note: offset is in bytes, converted to word offset for display
    '''

    def __init__(self, value: Union['LowLevelILInstruction', int, str], offset: int = 0):
        super().__init__(LowLevelILOperation.LLIL_FRAME_STORE)
        self.value = value
        self.offset = offset  # Byte offset relative to frame

    def __str__(self) -> str:
        # Convert byte offset to word offset for display
        # Always show explicit offset: fp + 0, fp + 1, etc.
        word_offset = self.offset // WORD_SIZE
        if word_offset >= 0:
            return f'STACK[fp + {word_offset}] = {self.value}'
        else:
            return f'STACK[fp - {-word_offset}] = {self.value}'


class LowLevelILSpAdd(LowLevelILStatement):
    '''sp = sp + delta (statement)'''

    def __init__(self, delta: int):
        super().__init__(LowLevelILOperation.LLIL_SP_ADD)
        self.delta = delta

    def __str__(self) -> str:
        if self.delta == 1:
            return 'sp++'
        elif self.delta == -1:
            return 'sp--'
        elif self.delta > 0:
            return f'sp += {self.delta}'
        else:
            return f'sp -= {-self.delta}'




class LowLevelILStackAddr(LowLevelILExpr):
    '''Address of a specific stack slot (constant, independent of sp changes)

    Used for PUSH_STACK_OFFSET which pushes the address of a stack slot.
    The slot_index is computed at build time and remains constant regardless
    of subsequent sp changes.
    '''

    def __init__(self, slot_index: int):
        super().__init__(LowLevelILOperation.LLIL_STACK_ADDR)
        self.slot_index = slot_index  # Absolute stack slot index

    def __str__(self) -> str:
        return f'&STACK[{self.slot_index}]'


# Alias for compatibility
LowLevelILVspAdd = LowLevelILSpAdd


# === Register Operations ===

class LowLevelILRegStore(LowLevelILStatement):
    '''REG[index] = value (statement)'''

    def __init__(self, reg_index: int, value: Union['LowLevelILInstruction', int]):
        super().__init__(LowLevelILOperation.LLIL_REG_STORE)
        self.reg_index = reg_index
        self.value = value

    def __str__(self) -> str:
        return f'REG[{self.reg_index}] = {self.value}'


class LowLevelILRegLoad(LowLevelILExpr):
    '''load REG[index] (expression)'''

    def __init__(self, reg_index: int):
        super().__init__(LowLevelILOperation.LLIL_REG_LOAD)
        self.reg_index = reg_index

    def __str__(self) -> str:
        return f'REG[{self.reg_index}]'


# === Arithmetic Operations ===

class LowLevelILBinaryOp(LowLevelILExpr, BinaryOperation):
    '''Base for binary operations (expression)'''

    def __init__(
        self,
        operation: LowLevelILOperation,
        lhs: 'LowLevelILExpr' = None,
        rhs: 'LowLevelILExpr' = None
    ):
        super().__init__(operation)
        self.lhs = lhs  # Left operand expression
        self.rhs = rhs  # Right operand expression

    def __str__(self) -> str:
        if self.lhs and self.rhs:
            return f'{self.operation_name}({self.lhs}, {self.rhs})'
        return self.operation_name


class LowLevelILAdd(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_ADD, lhs, rhs)


class LowLevelILMul(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_MUL, lhs, rhs)


class LowLevelILSub(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_SUB, lhs, rhs)


class LowLevelILDiv(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_DIV, lhs, rhs)


class LowLevelILEq(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_EQ, lhs, rhs)


class LowLevelILNe(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_NE, lhs, rhs)


class LowLevelILLt(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_LT, lhs, rhs)


class LowLevelILLe(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_LE, lhs, rhs)


class LowLevelILGt(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_GT, lhs, rhs)


class LowLevelILGe(LowLevelILBinaryOp):
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_GE, lhs, rhs)


class LowLevelILAnd(LowLevelILBinaryOp):
    '''Bitwise AND'''
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_AND, lhs, rhs)


class LowLevelILOr(LowLevelILBinaryOp):
    '''Bitwise OR'''
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_OR, lhs, rhs)


class LowLevelILLogicalAnd(LowLevelILBinaryOp):
    '''Logical AND (&&)'''
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_LOGICAL_AND, lhs, rhs)


class LowLevelILLogicalOr(LowLevelILBinaryOp):
    '''Logical OR (||)'''
    def __init__(self, lhs: 'LowLevelILExpr' = None, rhs: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_LOGICAL_OR, lhs, rhs)


# === Unary Operations ===

class LowLevelILUnaryOp(LowLevelILExpr, UnaryOperation):
    '''Base class for unary operations (expression)'''

    def __init__(self, operation: LowLevelILOperation, operand: 'LowLevelILExpr' = None):
        super().__init__(operation)
        self.operand = operand

    def __str__(self) -> str:
        if self.operand:
            return f'{self.operation_name}({self.operand})'
        return self.operation_name


class LowLevelILNeg(LowLevelILUnaryOp):
    '''Arithmetic negation (-x)'''
    def __init__(self, operand: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_NEG, operand)


class LowLevelILNot(LowLevelILUnaryOp):
    '''Logical NOT (!x)'''
    def __init__(self, operand: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_NOT, operand)


class LowLevelILTestZero(LowLevelILUnaryOp):
    '''Test if zero (x == 0)'''
    def __init__(self, operand: 'LowLevelILExpr' = None):
        super().__init__(LowLevelILOperation.LLIL_TEST_ZERO, operand)


# === Control Flow ===

class LowLevelILGoto(LowLevelILStatement, Terminal):
    '''Unconditional jump - target must be a BasicBlock (terminal)'''

    def __init__(self, target: 'LowLevelILBasicBlock'):
        super().__init__(LowLevelILOperation.LLIL_JMP)
        self.target = target

    def __str__(self) -> str:
        return f'goto {self.target.block_name}'


class LowLevelILJmp(LowLevelILGoto):
    '''Alias for LowLevelILGoto for convenience'''
    pass


class LowLevelILIf(LowLevelILStatement, Terminal):
    '''Conditional branch - targets must be BasicBlocks (terminal)

    condition: LowLevelILInstruction that evaluates to true/false

    Note: This is a terminal instruction - no more instructions can be added
    to the block after an If instruction.
    '''

    def __init__(self, condition: 'LowLevelILBinaryOp', true_target: 'LowLevelILBasicBlock',
                 false_target: Optional['LowLevelILBasicBlock'] = None):
        super().__init__(LowLevelILOperation.LLIL_BRANCH)
        self.condition = condition  # An instruction that produces a boolean value
        self.true_target = true_target
        self.false_target = false_target

    def __str__(self) -> str:
        true_name = self.true_target.block_name
        if self.false_target:
            false_name = self.false_target.block_name
            return f'if {self.condition} goto {true_name} else {false_name}'
        else:
            return f'if {self.condition} goto {true_name}'


class LowLevelILCall(LowLevelILStatement, Terminal):
    '''Function call (terminal)

    In Falcom VM, call is a terminal instruction because control transfers
    to the called function, then returns to an explicit return address
    (not fall-through).
    '''

    def __init__(
        self,
        target: Union[str, 'LowLevelILInstruction'],
        return_target: Union[str, 'LowLevelILBasicBlock'],
    ):
        super().__init__(LowLevelILOperation.LLIL_CALL)
        self.target = target
        self.return_target = return_target  # Where to return after call (label or block)
        self.args = []    # for debug only

    def __str__(self) -> str:
        if self.args:
            return f'call {self.target}({', '.join(str(arg) for arg in self.args)})'
        else:
            return f'call {self.target}'


class LowLevelILRet(LowLevelILStatement, Terminal):
    '''Return from function (terminal)'''

    def __init__(self):
        super().__init__(LowLevelILOperation.LLIL_RET)

    def __str__(self) -> str:
        return 'return'


# === Constants and Special ===

class LowLevelILConst(LowLevelILExpr, Constant):
    '''Constant value (int, float, or string) (expression)'''

    def __init__(self, value: Union[int, float, str], is_hex: bool = False, is_raw: bool = False):
        super().__init__(LowLevelILOperation.LLIL_CONST)
        self.value = value
        self.is_hex = is_hex  # True to display as hex, False for decimal (default)
        self.is_raw = is_raw  # True for raw values (type-less), False for typed constants

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"

        elif isinstance(self.value, float):
            # Format float with up to 6 decimal places, strip trailing zeros
            formatted = f'{self.value:.6f}'
            # Remove trailing zeros and decimal point if not needed
            formatted = formatted.rstrip('0').rstrip('.')
            return formatted

        elif isinstance(self.value, int):
            # Raw values are always displayed in hex
            if self.is_raw:
                if self.value < 0:
                    return f'-0x{-self.value:X}'
                else:
                    return f'0x{self.value:X}'
            elif self.is_hex:
                # Hex display
                if self.value < 0:
                    # Negative hex: -0xAB
                    return f'-0x{-self.value:X}'
                else:
                    return f'0x{self.value:X}'
            else:
                # Decimal display
                return str(self.value)

        else:
            return str(self.value)


class LowLevelILLabelInstr(LowLevelILStatement):
    '''Label instruction (for marking positions in code) (statement)'''

    def __init__(self, name: str):
        super().__init__(LowLevelILOperation.LLIL_LABEL)
        self.name = name

    def __str__(self) -> str:
        return f'{self.name}:'


class LowLevelILDebug(LowLevelILStatement):
    '''Debug information (statement)'''

    def __init__(self, debug_type: str, value: Any):
        super().__init__(LowLevelILOperation.LLIL_DEBUG)
        self.debug_type = debug_type
        self.value = value

    def __str__(self) -> str:
        return f'dbg_{self.debug_type.lower()} {self.value}'


# === VM Specific ===

class LowLevelILSyscall(LowLevelILStatement):
    '''System call (statement)'''

    def __init__(self, subsystem: int, cmd: int, argc: int):
        super().__init__(LowLevelILOperation.LLIL_SYSCALL)
        self.subsystem = subsystem
        self.cmd = cmd
        self.argc = argc

    def __str__(self) -> str:
        return f'SYSCALL({self.subsystem}, 0x{self.cmd:02x}, {self.argc})'


# === Helper Functions ===

def default_label_for_addr(addr: int) -> str:
    '''Generate default label name from address

    Args:
        addr: Block start address

    Returns:
        Label in format 'loc_{addr:X}' (e.g., 'loc_1FFDB6')
    '''
    return f'loc_{addr:X}'


# === Container Classes ===

class LowLevelILBasicBlock:
    '''Basic block containing LLIL instructions (following BN design)'''

    def __init__(self, start: int, index: int = 0, label: Optional[str] = None):
        self.start = start
        self.end = start
        self.index = index  # Block index in function
        self.label = label if label is not None else default_label_for_addr(start)
        self.instructions: List[LowLevelILInstruction] = []
        self.sp_in = 0   # sp state at block entry
        self.sp_out = 0  # sp state at block exit

        # Control flow edges (following BN design)
        self.outgoing_edges: List['LowLevelILBasicBlock'] = []
        self.incoming_edges: List['LowLevelILBasicBlock'] = []

    def add_instruction(self, inst: LowLevelILInstruction):
        '''Add instruction to this block

        Raises:
            RuntimeError: If block already has a terminal instruction
        '''
        if self.has_terminal:
            raise RuntimeError(
                f'Cannot add instruction to {self.block_name}: '
                f'block already has terminal instruction {self.instructions[-1]}'
            )
        inst.inst_index = len(self.instructions)
        self.instructions.append(inst)

    def add_outgoing_edge(self, target: 'LowLevelILBasicBlock'):
        '''Add outgoing edge to another block'''
        if target not in self.outgoing_edges:
            self.outgoing_edges.append(target)
        if self not in target.incoming_edges:
            target.incoming_edges.append(self)

    @property
    def has_terminal(self) -> bool:
        '''Check if block ends with a terminal instruction'''
        if not self.instructions:
            return False
        return isinstance(self.instructions[-1], Terminal)

    @property
    def block_name(self) -> str:
        '''Get canonical block name for jump targets and references

        Always returns "block_N" format for consistency.
        Use label_name property to get the user-defined label if present.
        '''
        return f'block_{self.index}'

    @property
    def label_name(self) -> Optional[str]:
        '''Get label name if this block starts with a label instruction'''
        if self.instructions and isinstance(self.instructions[0], LowLevelILLabelInstr):
            return self.instructions[0].name
        return None

    def __str__(self) -> str:
        result = f'{self.block_name} @ {hex(self.start)}: [sp={self.sp_in}]\n'
        for i, inst in enumerate(self.instructions):
            result += f'  {inst}\n'
        if self.outgoing_edges:
            targets = [b.block_name for b in self.outgoing_edges]
            result += f'  -> {', '.join(targets)}\n'
        return result


class LowLevelILFunction:
    '''Function containing LLIL basic blocks (following BN design)'''

    def __init__(self, name: str, start_addr: int = 0, num_params: int = 0):
        self.name = name
        self.start_addr = start_addr
        self.num_params = num_params  # Number of parameters
        self.basic_blocks: List[LowLevelILBasicBlock] = []
        self._block_map: dict[int, LowLevelILBasicBlock] = {}  # addr -> block
        self._label_map: dict[str, LowLevelILBasicBlock] = {}  # label -> block
        self.frame_base_sp: Optional[int] = None  # Frame pointer (sp at function entry)

    def add_basic_block(self, block: LowLevelILBasicBlock):
        '''Add basic block to function'''
        block.index = len(self.basic_blocks)
        self.basic_blocks.append(block)
        self._block_map[block.start] = block
        self._label_map[block.label] = block

    def get_block_by_addr(self, addr: int) -> Optional[LowLevelILBasicBlock]:
        '''Get basic block by start address'''
        return self._block_map.get(addr)

    def get_block_by_label(self, label: str) -> Optional[LowLevelILBasicBlock]:
        '''Get block by label name (O(1) lookup using label map)'''
        return self._label_map.get(label)

    def build_cfg(self):
        '''Build control flow graph from terminal instructions

        Note: With block-based targets, edges are created when instructions
        are added. This method handles fall-through edges for non-terminal blocks.
        '''
        for block in self.basic_blocks:
            if not block.instructions:
                continue

            last_inst = block.instructions[-1]

            # Terminal instructions already have their edges set via block references
            if isinstance(last_inst, LowLevelILGoto):
                # Edge already created: block -> target
                block.add_outgoing_edge(last_inst.target)

            elif isinstance(last_inst, LowLevelILIf):
                # Both targets must be explicitly specified
                if any([
                    last_inst.true_target is None,
                    last_inst.false_target is None,
                ]):
                    raise RuntimeError(
                        f'Block {block.index} has If instruction with no true_target or false_target. '
                        f'Both true_target and false_target must be explicitly specified.'
                    )

                block.add_outgoing_edge(last_inst.true_target)
                block.add_outgoing_edge(last_inst.false_target)

            elif isinstance(last_inst, LowLevelILCall):
                # Call returns to explicit return target
                if last_inst.return_target is not None:
                    return_block = last_inst.return_target
                    # Resolve label if needed
                    if isinstance(return_block, str):
                        return_block = self.get_block_by_label(return_block)
                        if return_block is None:
                            raise RuntimeError(f'Undefined return label: {last_inst.return_target}')

                    block.add_outgoing_edge(return_block)

                else:
                    raise NotImplementedError('Fall through is not supported')
                    # next_idx = block.index + 1
                    # if next_idx < len(self.basic_blocks):
                    #     block.add_outgoing_edge(self.basic_blocks[next_idx])

            elif not isinstance(last_inst, Terminal):
                # Should not happen - all blocks must end with terminal
                raise RuntimeError(
                    f'Block {block.index} ends with non-terminal instruction: {last_inst}'
                )

    def __str__(self) -> str:
        result = f'; ---------- {self.name} ----------\n'
        for block in self.basic_blocks:
            result += str(block)
        return result


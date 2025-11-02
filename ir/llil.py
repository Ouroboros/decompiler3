"""
Low Level IL v2 - Optimized Stack-based Design
Following the confirmed VM semantics with layered architecture
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, TYPE_CHECKING
from enum import IntEnum
import uuid

if TYPE_CHECKING:
    from typing import NewType
    InstructionIndex = NewType('InstructionIndex', int)
else:
    InstructionIndex = int


class LowLevelILOperation(IntEnum):
    """Atomic LLIL operations"""

    # Stack operations (atomic)
    LLIL_STACK_STORE = 0        # S[vsp + offset] = value
    LLIL_STACK_LOAD = 1         # load S[vsp + offset]
    LLIL_VSP_ADD = 2            # vsp = vsp + delta

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


class LowLevelILInstruction(ABC):
    """Base class for all LLIL instructions"""

    def __init__(self, operation: LowLevelILOperation, size: int = 4):
        self.operation = operation
        self.size = size
        self.address = 0
        self.instr_index = 0

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {str(self)}>"


# === Instruction Categories (following BinaryNinja design) ===

class ControlFlow(LowLevelILInstruction):
    """Base class for control flow instructions"""
    pass


class Terminal(ControlFlow):
    """Base class for terminal control flow instructions (goto, ret, etc)"""
    pass


# === Atomic Stack Operations ===

class LowLevelILStackStore(LowLevelILInstruction):
    """S[vsp + offset] = value"""

    def __init__(self, value: Union['LowLevelILInstruction', int, str], offset: int = 0, size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_STACK_STORE, size)
        self.value = value
        self.offset = offset

    def __str__(self) -> str:
        if self.offset == 0:
            return f"S[vsp] = {self.value}"
        elif self.offset > 0:
            return f"S[vsp + {self.offset}] = {self.value}"
        else:
            return f"S[vsp - {-self.offset}] = {self.value}"


class LowLevelILStackLoad(LowLevelILInstruction):
    """load S[vsp + offset]"""

    def __init__(self, offset: int = 0, size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_STACK_LOAD, size)
        self.offset = offset

    def __str__(self) -> str:
        if self.offset == 0:
            return "S[vsp]"
        elif self.offset > 0:
            return f"S[vsp + {self.offset}]"
        else:
            return f"S[vsp - {-self.offset}]"


class LowLevelILVspAdd(LowLevelILInstruction):
    """vsp = vsp + delta"""

    def __init__(self, delta: int):
        super().__init__(LowLevelILOperation.LLIL_VSP_ADD, 0)
        self.delta = delta

    def __str__(self) -> str:
        if self.delta == 1:
            return "vsp++"
        elif self.delta == -1:
            return "vsp--"
        elif self.delta > 0:
            return f"vsp += {self.delta}"
        else:
            return f"vsp -= {-self.delta}"


# === Register Operations ===

class LowLevelILRegStore(LowLevelILInstruction):
    """R[index] = value"""

    def __init__(self, reg_index: int, value: Union['LowLevelILInstruction', int], size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_REG_STORE, size)
        self.reg_index = reg_index
        self.value = value

    def __str__(self) -> str:
        return f"R[{self.reg_index}] = {self.value}"


class LowLevelILRegLoad(LowLevelILInstruction):
    """load R[index]"""

    def __init__(self, reg_index: int, size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_REG_LOAD, size)
        self.reg_index = reg_index

    def __str__(self) -> str:
        return f"R[{self.reg_index}]"


# === Arithmetic Operations ===

class LowLevelILBinaryOp(LowLevelILInstruction):
    """Base for binary operations"""

    def __init__(self, operation: LowLevelILOperation, size: int = 4):
        super().__init__(operation, size)


class LowLevelILAdd(LowLevelILBinaryOp):
    def __init__(self, size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_ADD, size)

    def __str__(self) -> str:
        return "ADD"


class LowLevelILEq(LowLevelILBinaryOp):
    def __init__(self, size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_EQ, size)

    def __str__(self) -> str:
        return "EQ"


# === Label for control flow ===

class LowLevelILLabel:
    """Label for control flow targets (similar to BinaryNinja's design)"""

    def __init__(self):
        self.resolved = False
        self.ref = False
        self.operand: Optional[InstructionIndex] = None

    def __str__(self) -> str:
        if self.operand is not None:
            return f"@{self.operand}"
        return "@unresolved"


# === Control Flow ===

class LowLevelILGoto(Terminal):
    """Unconditional jump (following BN naming)"""

    def __init__(self, target: Union['LowLevelILLabel', InstructionIndex]):
        super().__init__(LowLevelILOperation.LLIL_JMP)
        self.target = target

    def __str__(self) -> str:
        return f"goto {self.target}"


class LowLevelILJmp(LowLevelILGoto):
    """Alias for LowLevelILGoto for convenience"""
    pass


class LowLevelILIf(ControlFlow):
    """Conditional branch based on stack top"""

    def __init__(self, condition: str, true_target: Union['LowLevelILLabel', InstructionIndex],
                 false_target: Union['LowLevelILLabel', InstructionIndex]):
        super().__init__(LowLevelILOperation.LLIL_BRANCH)
        self.condition = condition  # "zero", "nonzero", etc.
        self.true_target = true_target
        self.false_target = false_target

    def __str__(self) -> str:
        return f"if {self.condition} then {self.true_target} else {self.false_target}"


class LowLevelILBranch(LowLevelILIf):
    """Simplified branch with only one target (falls through otherwise)"""

    def __init__(self, condition: str, target: Union['LowLevelILLabel', InstructionIndex]):
        # For single target branch, we don't set false_target
        super().__init__(condition, target, None)  # type: ignore

    def __str__(self) -> str:
        return f"if {self.condition} goto {self.true_target}"


class LowLevelILCall(ControlFlow):
    """Function call"""

    def __init__(self, target: Union[str, 'LowLevelILInstruction']):
        super().__init__(LowLevelILOperation.LLIL_CALL)
        self.target = target

    def __str__(self) -> str:
        return f"call {self.target}"


class LowLevelILRet(Terminal):
    """Return from function"""

    def __init__(self):
        super().__init__(LowLevelILOperation.LLIL_RET)

    def __str__(self) -> str:
        return "return"


# === Constants and Special ===

class LowLevelILConst(LowLevelILInstruction):
    """Constant value"""

    def __init__(self, value: Union[int, str], size: int = 4):
        super().__init__(LowLevelILOperation.LLIL_CONST, size)
        self.value = value

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        elif isinstance(self.value, int):
            if self.value < 0:
                return str(self.value)
            else:
                return f"0x{self.value:08x}" if self.value > 255 else str(self.value)
        else:
            return str(self.value)


class LowLevelILLabelInstr(LowLevelILInstruction):
    """Label instruction (for marking positions in code)"""

    def __init__(self, name: str):
        super().__init__(LowLevelILOperation.LLIL_LABEL)
        self.name = name

    def __str__(self) -> str:
        return f"{self.name}:"


class LowLevelILDebug(LowLevelILInstruction):
    """Debug information"""

    def __init__(self, debug_type: str, value: Any):
        super().__init__(LowLevelILOperation.LLIL_DEBUG)
        self.debug_type = debug_type
        self.value = value

    def __str__(self) -> str:
        return f"DBG_{self.debug_type.upper()} {self.value}"


# === VM Specific ===

class LowLevelILSyscall(LowLevelILInstruction):
    """System call"""

    def __init__(self, catalog: int, cmd: int, arg_count: int):
        super().__init__(LowLevelILOperation.LLIL_SYSCALL)
        self.catalog = catalog
        self.cmd = cmd
        self.arg_count = arg_count

    def __str__(self) -> str:
        return f"SYSCALL({self.catalog}, 0x{self.cmd:02x}, {self.arg_count})"


# === Container Classes ===

class LowLevelILBasicBlock:
    """Basic block containing LLIL instructions (following BN design)"""

    def __init__(self, start: int, index: int = 0):
        self.start = start
        self.end = start
        self.index = index  # Block index in function
        self.instructions: List[LowLevelILInstruction] = []
        self.vsp_in = 0   # vsp state at block entry
        self.vsp_out = 0  # vsp state at block exit

        # Control flow edges (following BN design)
        self.outgoing_edges: List['LowLevelILBasicBlock'] = []
        self.incoming_edges: List['LowLevelILBasicBlock'] = []

    def add_instruction(self, instr: LowLevelILInstruction):
        """Add instruction to this block"""
        instr.instr_index = len(self.instructions)
        self.instructions.append(instr)

    def add_outgoing_edge(self, target: 'LowLevelILBasicBlock'):
        """Add outgoing edge to another block"""
        if target not in self.outgoing_edges:
            self.outgoing_edges.append(target)
        if self not in target.incoming_edges:
            target.incoming_edges.append(self)

    @property
    def has_terminal(self) -> bool:
        """Check if block ends with a terminal instruction"""
        if not self.instructions:
            return False
        return isinstance(self.instructions[-1], Terminal)

    def __str__(self) -> str:
        result = f"block_{self.index} @ {hex(self.start)}: [vsp={self.vsp_in}]\n"
        for i, instr in enumerate(self.instructions):
            result += f"  {instr}\n"
        if self.outgoing_edges:
            targets = [f"block_{b.index}" for b in self.outgoing_edges]
            result += f"  -> {', '.join(targets)}\n"
        return result


class LowLevelILFunction:
    """Function containing LLIL basic blocks (following BN design)"""

    def __init__(self, name: str, start_addr: int = 0):
        self.name = name
        self.start_addr = start_addr
        self.basic_blocks: List[LowLevelILBasicBlock] = []
        self._block_map: dict[int, LowLevelILBasicBlock] = {}  # addr -> block

    def add_basic_block(self, block: LowLevelILBasicBlock):
        """Add basic block to function"""
        block.index = len(self.basic_blocks)
        self.basic_blocks.append(block)
        self._block_map[block.start] = block

    def get_basic_block_at(self, addr: int) -> Optional[LowLevelILBasicBlock]:
        """Get basic block at address"""
        return self._block_map.get(addr)

    def build_cfg(self):
        """Build control flow graph from terminal instructions"""
        for block in self.basic_blocks:
            if not block.instructions:
                continue

            last_instr = block.instructions[-1]

            # Handle different terminal types
            if isinstance(last_instr, LowLevelILGoto):
                # Unconditional jump
                if isinstance(last_instr.target, int):
                    target_block = self.get_basic_block_at(last_instr.target)
                    if target_block:
                        block.add_outgoing_edge(target_block)

            elif isinstance(last_instr, LowLevelILIf):
                # Conditional branch
                if isinstance(last_instr.true_target, int):
                    true_block = self.get_basic_block_at(last_instr.true_target)
                    if true_block:
                        block.add_outgoing_edge(true_block)

                if last_instr.false_target and isinstance(last_instr.false_target, int):
                    false_block = self.get_basic_block_at(last_instr.false_target)
                    if false_block:
                        block.add_outgoing_edge(false_block)

            elif not isinstance(last_instr, Terminal):
                # Falls through to next block
                next_idx = block.index + 1
                if next_idx < len(self.basic_blocks):
                    block.add_outgoing_edge(self.basic_blocks[next_idx])

    def __str__(self) -> str:
        result = f"; ---------- {self.name} ----------\n"
        for block in self.basic_blocks:
            result += str(block)
        return result


# === Special Constants ===

class FalcomConstants:
    """Falcom VM specific constants"""

    @staticmethod
    def current_func_id():
        return LowLevelILConst("func_id", 4)

    @staticmethod
    def ret_addr(label: str):
        return LowLevelILConst(f"&{label}", 8)
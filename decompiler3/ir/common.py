"""
Common IL Infrastructure - Based on BinaryNinja Design

This module provides the foundational classes and interfaces for the
three-layer IR system (LLIL, MLIL, HLIL), heavily inspired by BinaryNinja's architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Dict, Set, Tuple, Generator, NewType
from dataclasses import dataclass
from enum import Enum, IntEnum
import uuid


# Type definitions following BinaryNinja style
ExpressionIndex = NewType('ExpressionIndex', int)
InstructionIndex = NewType('InstructionIndex', int)
Index = Union[ExpressionIndex, InstructionIndex]


class ILOperation(IntEnum):
    """Base enum for IL operations"""
    pass


class OperandType(IntEnum):
    """Types of operands for IL instructions"""
    INT = 0      # Integer literal
    STR = 1      # String literal
    FLOAT = 2    # Float literal
    FUNC_ID = 3  # Current function ID
    RET_ADDR = 4 # Return address label
    EXPR = 5     # Expression/instruction result


@dataclass
class TypedOperand:
    """Typed operand for IL instructions"""
    operand_type: OperandType
    value: Any

    def __str__(self) -> str:
        if self.operand_type == OperandType.INT:
            return str(self.value)
        elif self.operand_type == OperandType.STR:
            return f"'{self.value}'"
        elif self.operand_type == OperandType.FLOAT:
            return f"{self.value}f"
        elif self.operand_type == OperandType.FUNC_ID:
            return "current_func_id()"
        elif self.operand_type == OperandType.RET_ADDR:
            return f"ret_addr('{self.value}')"
        else:
            return str(self.value)


class LowLevelILOperation(ILOperation):
    """Low Level IL Operations - based on BinaryNinja LLIL"""
    # Arithmetic operations
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4
    NEG = 5

    # Bitwise operations
    AND = 10
    OR = 11
    XOR = 12
    NOT = 13
    LSL = 14  # Logical shift left
    LSR = 15  # Logical shift right
    ASR = 16  # Arithmetic shift right

    # Comparison operations
    CMP_E = 20    # Equal
    CMP_NE = 21   # Not equal
    CMP_SLT = 22  # Signed less than
    CMP_ULT = 23  # Unsigned less than
    CMP_SLE = 24  # Signed less than or equal
    CMP_ULE = 25  # Unsigned less than or equal

    # Memory operations
    LOAD = 30
    STORE = 31

    # Register operations
    REG = 40
    SET_REG = 41

    # Stack operations
    PUSH = 50
    POP = 51

    # Control flow
    JUMP = 60
    JUMP_TO = 61
    CALL = 62
    RET = 63
    IF = 64
    GOTO = 65
    LABEL = 66

    # Constants and variables
    CONST = 70
    CONST_PTR = 71

    # Special
    NOP = 80
    UNDEF = 81
    UNIMPL = 82
    NORET = 83
    BP = 84
    SYSCALL = 85


class MediumLevelILOperation(ILOperation):
    """Medium Level IL Operations - based on BinaryNinja MLIL"""
    # All LLIL operations plus medium-level specific ones
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3

    # Comparison operations
    CMP_SLE = 20  # Signed less than or equal

    # Variable operations
    VAR = 100
    SET_VAR = 101
    VAR_FIELD = 102
    SET_VAR_FIELD = 103

    # Address operations
    ADDRESS_OF = 110

    # Array operations
    VAR_SPLIT = 120

    # Control flow (medium level)
    JUMP = 130
    GOTO = 131
    IF = 132
    CALL = 133
    TAILCALL = 134
    RET = 135

    # Constants
    CONST = 140
    CONST_PTR = 141

    # Special
    NOP = 150
    UNDEF = 151
    UNIMPL = 152


class HighLevelILOperation(ILOperation):
    """High Level IL Operations - based on BinaryNinja HLIL"""
    # Basic arithmetic
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3

    # Comparison operations
    CMP_E = 20    # Equal
    CMP_NE = 21   # Not equal
    CMP_SLT = 22  # Signed less than
    CMP_ULT = 23  # Unsigned less than
    CMP_SLE = 24  # Signed less than or equal
    CMP_ULE = 25  # Unsigned less than or equal

    # Variable operations
    VAR = 200
    ASSIGN = 201
    VAR_INIT = 202

    # Control structures
    IF = 210
    WHILE = 211
    FOR = 212
    DO_WHILE = 213
    SWITCH = 214
    GOTO = 215
    LABEL = 216

    # Function operations
    CALL = 220
    TAILCALL = 221
    RET = 222

    # Block structures
    BLOCK = 230

    # Constants
    CONST = 240


@dataclass(frozen=True)
class ILOperationAndSize:
    """Operation with size information"""
    operation: ILOperation
    size: int

    def __repr__(self):
        if self.size == 0:
            return f"<{self.__class__.__name__}: {self.operation.name}>"
        return f"<{self.__class__.__name__}: {self.operation.name} {self.size}>"


class ILInstructionAttribute(Enum):
    """Instruction attributes for analysis"""
    IL_INTRINSIC = "intrinsic"
    IL_FLAG_WRITE = "flag_write"
    IL_FLAG_READ = "flag_read"


# Mixins for instruction types (following BinaryNinja pattern)
class Terminal:
    """Instructions that terminate a basic block"""
    pass


class ControlFlow:
    """Instructions that affect control flow"""
    pass


class Memory:
    """Instructions that access memory"""
    pass


class Arithmetic:
    """Arithmetic instructions"""
    pass


class Comparison:
    """Comparison instructions"""
    pass


class Constant:
    """Constant value instructions"""
    pass


class BinaryOperation:
    """Binary operation instructions"""
    pass


class UnaryOperation:
    """Unary operation instructions"""
    pass


class Call:
    """Function call instructions"""
    pass


class Return:
    """Return instructions"""
    pass


class SSA:
    """SSA form instructions"""
    pass


class Phi:
    """Phi node instructions"""
    pass


class Loop:
    """Loop structure instructions"""
    pass


class BaseILInstruction(ABC):
    """Base class for all IL instructions - inspired by BinaryNinja"""

    def __init__(self, operation: ILOperation, size: int = 0):
        object.__setattr__(self, 'operation', operation)
        object.__setattr__(self, 'size', size)
        object.__setattr__(self, 'operands', [])
        object.__setattr__(self, 'attributes', set())
        object.__setattr__(self, 'source_location', None)

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the instruction"""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {str(self)}>"

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        """Get detailed operand information"""
        return []


@dataclass(frozen=True)
class ILSourceLocation:
    """Source location information for IL instructions"""
    address: Optional[int] = None
    file_name: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None


class ILRegister:
    """Register representation in IL"""

    def __init__(self, name: str, index: int, size: int = 4):
        self.name = name
        self.index = index
        self.size = size
        self.temp = (index & 0x80000000) != 0

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<ILRegister: {self.name}>"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, ILRegister):
            return self.name == other.name and self.index == other.index
        return False


class ILFlag:
    """Flag representation in IL"""

    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index
        self.temp = (index & 0x80000000) != 0

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<ILFlag: {self.name}>"


class ILBasicBlock:
    """Basic block containing IL instructions"""

    def __init__(self, start_address: int):
        self.start_address = start_address
        self.id = str(uuid.uuid4())
        self.instructions: List[BaseILInstruction] = []
        self.predecessors: List['ILBasicBlock'] = []
        self.successors: List['ILBasicBlock'] = []
        self.dominance_frontier: List['ILBasicBlock'] = []

    def add_instruction(self, instruction: BaseILInstruction):
        """Add instruction to this basic block"""
        self.instructions.append(instruction)

    def __str__(self) -> str:
        result = f"block_{self.id[:8]} @ 0x{self.start_address:x}:\n"
        for i, inst in enumerate(self.instructions):
            result += f"  {i}: {inst}\n"
        return result


class ILFunction:
    """Function containing basic blocks and IL instructions"""

    def __init__(self, name: str, address: Optional[int] = None):
        self.name = name
        self.address = address
        self.basic_blocks: List[ILBasicBlock] = []
        self.source_function = None  # Reference to original function
        self.ssa_form = False

    def get_basic_block(self, address: int) -> Optional[ILBasicBlock]:
        """Get basic block by address"""
        for bb in self.basic_blocks:
            if bb.start_address == address:
                return bb
        return None

    def add_basic_block(self, bb: ILBasicBlock):
        """Add basic block to function"""
        self.basic_blocks.append(bb)

    def __str__(self) -> str:
        result = f"function {self.name}() {{\n"
        for block in self.basic_blocks:
            for line in str(block).split('\n'):
                if line.strip():
                    result += f"  {line}\n"
        result += "}"
        return result


# Type aliases following BinaryNinja pattern
InstructionOrExpression = Union[BaseILInstruction, Index]
ILInstructionsType = Generator[BaseILInstruction, None, None]
ILBasicBlocksType = Generator[ILBasicBlock, None, None]
"""
Low Level Intermediate Language (LLIL) - Based on BinaryNinja Design

This module implements LLIL instructions following BinaryNinja's architecture.
LLIL is the lowest level IR, closest to assembly language.
"""

from typing import Any, List, Optional, Union, Dict, Tuple
from dataclasses import dataclass

from .common import (
    BaseILInstruction, ILBasicBlock, ILFunction, ILRegister, ILFlag,
    LowLevelILOperation, ILOperationAndSize, InstructionIndex, ExpressionIndex,
    Terminal, ControlFlow, Memory, Arithmetic, Comparison, Constant,
    BinaryOperation, UnaryOperation, Call, Return
)


@dataclass(frozen=True)
class LowLevelILOperationAndSize(ILOperationAndSize):
    """LLIL operation with size"""
    operation: LowLevelILOperation


class LowLevelILInstruction(BaseILInstruction):
    """Base class for all LLIL instructions"""

    def __init__(self, operation: LowLevelILOperation, size: int = 0):
        super().__init__(operation, size)
        self.operation = operation

    def _get_expr(self, index: int) -> 'LowLevelILInstruction':
        """Get expression operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")

    def _get_int(self, index: int) -> int:
        """Get integer operand at index"""
        if index < len(self.operands):
            return int(self.operands[index])
        raise IndexError(f"Operand index {index} out of range")

    def _get_reg(self, index: int) -> ILRegister:
        """Get register operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")

    def _get_flag(self, index: int) -> ILFlag:
        """Get flag operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")


# ============================================================================
# Arithmetic Instructions
# ============================================================================

class LowLevelILAdd(LowLevelILInstruction, Arithmetic, BinaryOperation):
    """Add two operands"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.ADD, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} + {self.right}"



class LowLevelILSub(LowLevelILInstruction, Arithmetic, BinaryOperation):
    """Subtract two operands"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.SUB, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} - {self.right}"



class LowLevelILMul(LowLevelILInstruction, Arithmetic, BinaryOperation):
    """Multiply two operands"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.MUL, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} * {self.right}"



class LowLevelILDiv(LowLevelILInstruction, Arithmetic, BinaryOperation):
    """Divide two operands"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.DIV, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} / {self.right}"


# ============================================================================
# Comparison Instructions
# ============================================================================


class LowLevelILCmpE(LowLevelILInstruction, Comparison, BinaryOperation):
    """Compare equal"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.CMP_E, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} == {self.right}"



class LowLevelILCmpNe(LowLevelILInstruction, Comparison, BinaryOperation):
    """Compare not equal"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.CMP_NE, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} != {self.right}"



class LowLevelILCmpSlt(LowLevelILInstruction, Comparison, BinaryOperation):
    """Compare signed less than"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.CMP_SLT, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} < {self.right}"


class LowLevelILCmpSle(LowLevelILInstruction, Comparison, BinaryOperation):
    """Compare signed less than or equal"""

    def __init__(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.CMP_SLE, size)
        self.operands = [left, right]

    @property
    def left(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "LowLevelILInstruction"),
            ("right", self.right, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.left} <= {self.right}"


# ============================================================================
# Memory Instructions
# ============================================================================


class LowLevelILLoad(LowLevelILInstruction, Memory):
    """Load from memory"""

    def __init__(self, address: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.LOAD, size)
        self.operands = [address]

    @property
    def src(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("src", self.src, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"[{self.src}]"



class LowLevelILStore(LowLevelILInstruction, Memory):
    """Store to memory"""

    def __init__(self, address: LowLevelILInstruction, value: LowLevelILInstruction, size: int = 4):
        super().__init__(LowLevelILOperation.STORE, size)
        self.operands = [address, value]

    @property
    def dest(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def src(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "LowLevelILInstruction"),
            ("src", self.src, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"[{self.dest}] = {self.src}"


# ============================================================================
# Register Instructions
# ============================================================================


class LowLevelILReg(LowLevelILInstruction):
    """Register access"""

    def __init__(self, reg: ILRegister):
        super().__init__(LowLevelILOperation.REG, reg.size)
        self.operands = [reg]

    @property
    def src(self) -> ILRegister:
        return self._get_reg(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("src", self.src, "ILRegister"),
        ]

    def __str__(self) -> str:
        return str(self.src)



class LowLevelILSetReg(LowLevelILInstruction):
    """Set register value"""

    def __init__(self, reg: ILRegister, value: LowLevelILInstruction):
        super().__init__(LowLevelILOperation.SET_REG, reg.size)
        self.operands = [reg, value]

    @property
    def dest(self) -> ILRegister:
        return self._get_reg(0)

    @property
    def src(self) -> LowLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "ILRegister"),
            ("src", self.src, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.dest} = {self.src}"


# ============================================================================
# Control Flow Instructions
# ============================================================================


class LowLevelILJump(LowLevelILInstruction, Terminal):
    """Unconditional jump"""

    def __init__(self, dest: LowLevelILInstruction):
        super().__init__(LowLevelILOperation.JUMP, 0)
        self.operands = [dest]

    @property
    def dest(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "LowLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"jump {self.dest}"



class LowLevelILGoto(LowLevelILInstruction, Terminal):
    """Goto instruction index"""

    def __init__(self, dest: InstructionIndex):
        super().__init__(LowLevelILOperation.GOTO, 0)
        self.operands = [dest]

    @property
    def dest(self) -> InstructionIndex:
        return InstructionIndex(self._get_int(0))

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "InstructionIndex"),
        ]

    def __str__(self) -> str:
        return f"goto {self.dest}"



class LowLevelILIf(LowLevelILInstruction, ControlFlow):
    """Conditional branch"""

    def __init__(self, condition: LowLevelILInstruction, true_target: InstructionIndex, false_target: InstructionIndex):
        super().__init__(LowLevelILOperation.IF, 0)
        self.operands = [condition, true_target, false_target]

    @property
    def condition(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def true(self) -> InstructionIndex:
        return InstructionIndex(self._get_int(1))

    @property
    def false(self) -> InstructionIndex:
        return InstructionIndex(self._get_int(2))

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("condition", self.condition, "LowLevelILInstruction"),
            ("true", self.true, "InstructionIndex"),
            ("false", self.false, "InstructionIndex"),
        ]

    def __str__(self) -> str:
        return f"if ({self.condition}) goto {self.true} else goto {self.false}"



class LowLevelILCall(LowLevelILInstruction, Call):
    """Function call"""

    def __init__(self, dest: LowLevelILInstruction, arguments: Optional[List[LowLevelILInstruction]] = None):
        super().__init__(LowLevelILOperation.CALL, 0)
        if arguments is None:
            arguments = []
        self.operands = [dest] + arguments

    @property
    def dest(self) -> LowLevelILInstruction:
        return self._get_expr(0)

    @property
    def arguments(self) -> List[LowLevelILInstruction]:
        return self.operands[1:]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [("dest", self.dest, "LowLevelILInstruction")]
        if self.arguments:
            operands.append(("arguments", self.arguments, "List[LowLevelILInstruction]"))
        return operands

    def __str__(self) -> str:
        if self.arguments:
            args = ", ".join(str(arg) for arg in self.arguments)
            return f"call {self.dest}({args})"
        return f"call {self.dest}"



class LowLevelILRet(LowLevelILInstruction, Return, Terminal):
    """Return from function"""

    def __init__(self, value: Optional[LowLevelILInstruction] = None):
        super().__init__(LowLevelILOperation.RET, 0)
        if value is not None:
            self.operands = [value]
        else:
            self.operands = []

    @property
    def dest(self) -> Optional[LowLevelILInstruction]:
        if self.operands:
            return self._get_expr(0)
        return None

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        if self.dest is not None:
            return [("dest", self.dest, "LowLevelILInstruction")]
        return []

    def __str__(self) -> str:
        if self.dest is not None:
            return f"return {self.dest}"
        return "return"


# ============================================================================
# Constant Instructions
# ============================================================================


class LowLevelILConst(LowLevelILInstruction, Constant):
    """Constant value"""

    def __init__(self, value: Union[int, float], size: int = 4):
        super().__init__(LowLevelILOperation.CONST, size)
        self.operands = [value]

    @property
    def constant(self) -> Union[int, float]:
        return self.operands[0]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("constant", self.constant, "int"),
        ]

    def __str__(self) -> str:
        return str(self.constant)



class LowLevelILConstPtr(LowLevelILInstruction, Constant):
    """Constant pointer"""

    def __init__(self, value: int, size: int = 8):
        super().__init__(LowLevelILOperation.CONST_PTR, size)
        self.operands = [value]

    @property
    def constant(self) -> int:
        return self.operands[0]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("constant", self.constant, "int"),
        ]

    def __str__(self) -> str:
        return f"0x{self.constant:x}"


# ============================================================================
# Special Instructions
# ============================================================================


class LowLevelILNop(LowLevelILInstruction):
    """No operation"""

    def __init__(self):
        super().__init__(LowLevelILOperation.NOP, 0)

    def __str__(self) -> str:
        return "nop"



class LowLevelILUndef(LowLevelILInstruction, Terminal):
    """Undefined instruction"""

    def __init__(self):
        super().__init__(LowLevelILOperation.UNDEF, 0)

    def __str__(self) -> str:
        return "undef"



class LowLevelILUnimpl(LowLevelILInstruction):
    """Unimplemented instruction"""

    def __init__(self):
        super().__init__(LowLevelILOperation.UNIMPL, 0)

    def __str__(self) -> str:
        return "unimpl"


# ============================================================================
# LLIL Function and Basic Block Classes
# ============================================================================

class LowLevelILBasicBlock(ILBasicBlock):
    """LLIL Basic Block"""

    def __init__(self, start_address: int):
        super().__init__(start_address)
        self.instructions: List[LowLevelILInstruction] = []


class LowLevelILFunction(ILFunction):
    """LLIL Function"""

    def __init__(self, name: str, address: Optional[int] = None):
        super().__init__(name, address)
        self.basic_blocks: List[LowLevelILBasicBlock] = []


# ============================================================================
# LLIL Builder (for constructing LLIL)
# ============================================================================

class LowLevelILBuilder:
    """Builder for constructing LLIL instructions"""

    def __init__(self, function: LowLevelILFunction):
        self.function = function
        self.current_block: Optional[LowLevelILBasicBlock] = None

    def set_current_block(self, block: LowLevelILBasicBlock):
        """Set the current basic block for instruction insertion"""
        self.current_block = block

    def add_instruction(self, instruction: LowLevelILInstruction):
        """Add instruction to current block"""
        if self.current_block is None:
            raise RuntimeError("No current block set")
        self.current_block.add_instruction(instruction)

    # Arithmetic operations
    def add(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILAdd:
        return LowLevelILAdd(left, right, size)

    def sub(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILSub:
        return LowLevelILSub(left, right, size)

    def mul(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILMul:
        return LowLevelILMul(left, right, size)

    def div(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILDiv:
        return LowLevelILDiv(left, right, size)

    # Comparison operations
    def cmp_e(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILCmpE:
        return LowLevelILCmpE(left, right, size)

    def cmp_ne(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILCmpNe:
        return LowLevelILCmpNe(left, right, size)

    def cmp_slt(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILCmpSlt:
        return LowLevelILCmpSlt(left, right, size)

    def cmp_sle(self, left: LowLevelILInstruction, right: LowLevelILInstruction, size: int = 4) -> LowLevelILCmpSle:
        return LowLevelILCmpSle(left, right, size)

    # Memory operations
    def load(self, address: LowLevelILInstruction, size: int = 4) -> LowLevelILLoad:
        return LowLevelILLoad(address, size)

    def store(self, address: LowLevelILInstruction, value: LowLevelILInstruction, size: int = 4) -> LowLevelILStore:
        return LowLevelILStore(address, value, size)

    # Register operations
    def reg(self, reg: ILRegister) -> LowLevelILReg:
        return LowLevelILReg(reg)

    def set_reg(self, reg: ILRegister, value: LowLevelILInstruction) -> LowLevelILSetReg:
        return LowLevelILSetReg(reg, value)

    # Control flow
    def jump(self, dest: LowLevelILInstruction) -> LowLevelILJump:
        return LowLevelILJump(dest)

    def goto(self, dest: InstructionIndex) -> LowLevelILGoto:
        return LowLevelILGoto(dest)

    def if_stmt(self, condition: LowLevelILInstruction, true_target: InstructionIndex, false_target: InstructionIndex) -> LowLevelILIf:
        return LowLevelILIf(condition, true_target, false_target)

    def call(self, dest: LowLevelILInstruction, arguments: Optional[List[LowLevelILInstruction]] = None) -> LowLevelILCall:
        return LowLevelILCall(dest, arguments)

    def ret(self, value: Optional[LowLevelILInstruction] = None) -> LowLevelILRet:
        return LowLevelILRet(value)

    # Constants
    def const(self, value: Union[int, float], size: int = 4) -> LowLevelILConst:
        return LowLevelILConst(value, size)

    def const_ptr(self, value: int, size: int = 8) -> LowLevelILConstPtr:
        return LowLevelILConstPtr(value, size)

    # Special
    def nop(self) -> LowLevelILNop:
        return LowLevelILNop()

    def undef(self) -> LowLevelILUndef:
        return LowLevelILUndef()

    def unimpl(self) -> LowLevelILUnimpl:
        return LowLevelILUnimpl()
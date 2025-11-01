"""
Medium Level Intermediate Language (MLIL) - Based on BinaryNinja Design

This module implements MLIL instructions following BinaryNinja's architecture.
MLIL is a higher-level IR with variables instead of registers and memory locations.
"""

from typing import Any, List, Optional, Union, Dict, Tuple
from dataclasses import dataclass

from .common import (
    BaseILInstruction, ILBasicBlock, ILFunction, ILRegister, ILFlag,
    MediumLevelILOperation, ILOperationAndSize, InstructionIndex, ExpressionIndex,
    Terminal, ControlFlow, Memory, Arithmetic, Comparison, Constant,
    BinaryOperation, UnaryOperation, Call, Return
)


@dataclass(frozen=True)
class MediumLevelILOperationAndSize(ILOperationAndSize):
    """MLIL operation with size"""
    operation: MediumLevelILOperation


@dataclass(frozen=True)
class Variable:
    """Variable representation in MLIL"""
    name: str
    var_type: Optional[str] = None
    size: int = 4
    source_type: str = "auto"  # auto, user, etc.

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<Variable: {self.name}>"


@dataclass(frozen=True, order=True)
class SSAVariable:
    """SSA form variable with version number"""
    var: Variable
    version: int

    def __repr__(self):
        return f"<SSAVariable: {self.var} version {self.version}>"

    @property
    def name(self) -> str:
        return f"{self.var.name}#{self.version}"


class MediumLevelILInstruction(BaseILInstruction):
    """Base class for all MLIL instructions"""

    def __init__(self, operation: MediumLevelILOperation, size: int = 0):
        super().__init__(operation, size)
        self.operation = operation

    def _get_expr(self, index: int) -> 'MediumLevelILInstruction':
        """Get expression operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")

    def _get_int(self, index: int) -> int:
        """Get integer operand at index"""
        if index < len(self.operands):
            return int(self.operands[index])
        raise IndexError(f"Operand index {index} out of range")

    def _get_var(self, index: int) -> Variable:
        """Get variable operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")

    def _get_var_ssa(self, index: int) -> SSAVariable:
        """Get SSA variable operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")


# ============================================================================
# Arithmetic Instructions
# ============================================================================


class MediumLevelILAdd(MediumLevelILInstruction, Arithmetic, BinaryOperation):
    """Add two operands"""

    def __init__(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4):
        super().__init__(MediumLevelILOperation.ADD, size)
        self.operands = [left, right]

    @property
    def left(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> MediumLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "MediumLevelILInstruction"),
            ("right", self.right, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"



class MediumLevelILSub(MediumLevelILInstruction, Arithmetic, BinaryOperation):
    """Subtract two operands"""

    def __init__(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4):
        super().__init__(MediumLevelILOperation.SUB, size)
        self.operands = [left, right]

    @property
    def left(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> MediumLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "MediumLevelILInstruction"),
            ("right", self.right, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"



class MediumLevelILMul(MediumLevelILInstruction, Arithmetic, BinaryOperation):
    """Multiply two operands"""

    def __init__(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4):
        super().__init__(MediumLevelILOperation.MUL, size)
        self.operands = [left, right]

    @property
    def left(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> MediumLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "MediumLevelILInstruction"),
            ("right", self.right, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"



class MediumLevelILDiv(MediumLevelILInstruction, Arithmetic, BinaryOperation):
    """Divide two operands"""

    def __init__(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4):
        super().__init__(MediumLevelILOperation.DIV, size)
        self.operands = [left, right]

    @property
    def left(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def right(self) -> MediumLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "MediumLevelILInstruction"),
            ("right", self.right, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


# ============================================================================
# Variable Instructions
# ============================================================================


class MediumLevelILVar(MediumLevelILInstruction):
    """Variable reference"""

    def __init__(self, var: Variable):
        super().__init__(MediumLevelILOperation.VAR, var.size)
        self.operands = [var]

    @property
    def src(self) -> Variable:
        return self._get_var(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("src", self.src, "Variable"),
        ]

    def __str__(self) -> str:
        return str(self.src)



class MediumLevelILSetVar(MediumLevelILInstruction):
    """Set variable value"""

    def __init__(self, var: Variable, value: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.SET_VAR, var.size)
        self.operands = [var, value]

    @property
    def dest(self) -> Variable:
        return self._get_var(0)

    @property
    def src(self) -> MediumLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "Variable"),
            ("src", self.src, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.dest} = {self.src}"



class MediumLevelILVarSsa(MediumLevelILInstruction):
    """SSA variable reference"""

    def __init__(self, var: SSAVariable):
        super().__init__(MediumLevelILOperation.VAR, var.var.size)
        self.operands = [var]

    @property
    def src(self) -> SSAVariable:
        return self._get_var_ssa(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("src", self.src, "SSAVariable"),
        ]

    def __str__(self) -> str:
        return str(self.src.name)



class MediumLevelILSetVarSsa(MediumLevelILInstruction):
    """Set SSA variable value"""

    def __init__(self, var: SSAVariable, value: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.SET_VAR, var.var.size)
        self.operands = [var, value]

    @property
    def dest(self) -> SSAVariable:
        return self._get_var_ssa(0)

    @property
    def src(self) -> MediumLevelILInstruction:
        return self._get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "SSAVariable"),
            ("src", self.src, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.dest.name} = {self.src}"


# ============================================================================
# Control Flow Instructions
# ============================================================================


class MediumLevelILJump(MediumLevelILInstruction, Terminal):
    """Unconditional jump"""

    def __init__(self, dest: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.JUMP, 0)
        self.operands = [dest]

    @property
    def dest(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "MediumLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"jump {self.dest}"



class MediumLevelILGoto(MediumLevelILInstruction, Terminal):
    """Goto instruction index"""

    def __init__(self, dest: InstructionIndex):
        super().__init__(MediumLevelILOperation.GOTO, 0)
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



class MediumLevelILIf(MediumLevelILInstruction, Terminal):
    """Conditional branch"""

    def __init__(self, condition: MediumLevelILInstruction, true_target: InstructionIndex, false_target: InstructionIndex):
        super().__init__(MediumLevelILOperation.IF, 0)
        self.operands = [condition, true_target, false_target]

    @property
    def condition(self) -> MediumLevelILInstruction:
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
            ("condition", self.condition, "MediumLevelILInstruction"),
            ("true", self.true, "InstructionIndex"),
            ("false", self.false, "InstructionIndex"),
        ]

    def __str__(self) -> str:
        return f"if ({self.condition}) goto {self.true} else goto {self.false}"



class MediumLevelILCall(MediumLevelILInstruction, Call):
    """Function call"""

    def __init__(self, dest: MediumLevelILInstruction, arguments: Optional[List[MediumLevelILInstruction]] = None):
        super().__init__(MediumLevelILOperation.CALL, 0)
        if arguments is None:
            arguments = []
        self.operands = [dest] + arguments

    @property
    def dest(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def arguments(self) -> List[MediumLevelILInstruction]:
        return self.operands[1:]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [("dest", self.dest, "MediumLevelILInstruction")]
        if self.arguments:
            operands.append(("arguments", self.arguments, "List[MediumLevelILInstruction]"))
        return operands

    def __str__(self) -> str:
        if self.arguments:
            args = ", ".join(str(arg) for arg in self.arguments)
            return f"call {self.dest}({args})"
        return f"call {self.dest}"



class MediumLevelILTailcall(MediumLevelILInstruction, Call, Terminal):
    """Tail call"""

    def __init__(self, dest: MediumLevelILInstruction, arguments: Optional[List[MediumLevelILInstruction]] = None):
        super().__init__(MediumLevelILOperation.TAILCALL, 0)
        if arguments is None:
            arguments = []
        self.operands = [dest] + arguments

    @property
    def dest(self) -> MediumLevelILInstruction:
        return self._get_expr(0)

    @property
    def arguments(self) -> List[MediumLevelILInstruction]:
        return self.operands[1:]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [("dest", self.dest, "MediumLevelILInstruction")]
        if self.arguments:
            operands.append(("arguments", self.arguments, "List[MediumLevelILInstruction]"))
        return operands

    def __str__(self) -> str:
        if self.arguments:
            args = ", ".join(str(arg) for arg in self.arguments)
            return f"tailcall {self.dest}({args})"
        return f"tailcall {self.dest}"



class MediumLevelILRet(MediumLevelILInstruction, Return, Terminal):
    """Return from function"""

    def __init__(self, sources: Optional[List[MediumLevelILInstruction]] = None):
        super().__init__(MediumLevelILOperation.RET, 0)
        if sources is None:
            sources = []
        self.operands = sources

    @property
    def src(self) -> List[MediumLevelILInstruction]:
        return self.operands

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        if self.src:
            return [("src", self.src, "List[MediumLevelILInstruction]")]
        return []

    def __str__(self) -> str:
        if self.src:
            if len(self.src) == 1:
                return f"return {self.src[0]}"
            else:
                src_str = ", ".join(str(s) for s in self.src)
                return f"return ({src_str})"
        return "return"


# ============================================================================
# Constant Instructions
# ============================================================================


class MediumLevelILConst(MediumLevelILInstruction, Constant):
    """Constant value"""

    def __init__(self, value: Union[int, float], size: int = 4):
        super().__init__(MediumLevelILOperation.CONST, size)
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



class MediumLevelILConstPtr(MediumLevelILInstruction, Constant):
    """Constant pointer"""

    def __init__(self, value: int, size: int = 8):
        super().__init__(MediumLevelILOperation.CONST_PTR, size)
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


class MediumLevelILNop(MediumLevelILInstruction):
    """No operation"""

    def __init__(self):
        super().__init__(MediumLevelILOperation.NOP, 0)

    def __str__(self) -> str:
        return "nop"



class MediumLevelILUndef(MediumLevelILInstruction):
    """Undefined value"""

    def __init__(self):
        super().__init__(MediumLevelILOperation.UNDEF, 0)

    def __str__(self) -> str:
        return "undef"



class MediumLevelILUnimpl(MediumLevelILInstruction):
    """Unimplemented instruction"""

    def __init__(self):
        super().__init__(MediumLevelILOperation.UNIMPL, 0)

    def __str__(self) -> str:
        return "unimpl"


# ============================================================================
# MLIL Function and Basic Block Classes
# ============================================================================

class MediumLevelILBasicBlock(ILBasicBlock):
    """MLIL Basic Block"""

    def __init__(self, start_address: int):
        super().__init__(start_address)
        self.instructions: List[MediumLevelILInstruction] = []


class MediumLevelILFunction(ILFunction):
    """MLIL Function"""

    def __init__(self, name: str, address: Optional[int] = None):
        super().__init__(name, address)
        self.basic_blocks: List[MediumLevelILBasicBlock] = []
        self.variables: Dict[str, Variable] = {}

    def create_variable(self, name: str, var_type: Optional[str] = None, size: int = 4) -> Variable:
        """Create a new variable in this function"""
        var = Variable(name, var_type, size)
        self.variables[name] = var
        return var

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get variable by name"""
        return self.variables.get(name)


# ============================================================================
# MLIL Builder (for constructing MLIL)
# ============================================================================

class MediumLevelILBuilder:
    """Builder for constructing MLIL instructions"""

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.current_block: Optional[MediumLevelILBasicBlock] = None

    def set_current_block(self, block: MediumLevelILBasicBlock):
        """Set the current basic block for instruction insertion"""
        self.current_block = block

    def add_instruction(self, instruction: MediumLevelILInstruction):
        """Add instruction to current block"""
        if self.current_block is None:
            raise RuntimeError("No current block set")
        self.current_block.add_instruction(instruction)

    # Arithmetic operations
    def add(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4) -> MediumLevelILAdd:
        return MediumLevelILAdd(left, right, size)

    def sub(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4) -> MediumLevelILSub:
        return MediumLevelILSub(left, right, size)

    def mul(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4) -> MediumLevelILMul:
        return MediumLevelILMul(left, right, size)

    def div(self, left: MediumLevelILInstruction, right: MediumLevelILInstruction, size: int = 4) -> MediumLevelILDiv:
        return MediumLevelILDiv(left, right, size)

    # Variable operations
    def var(self, var: Variable) -> MediumLevelILVar:
        return MediumLevelILVar(var)

    def set_var(self, var: Variable, value: MediumLevelILInstruction) -> MediumLevelILSetVar:
        return MediumLevelILSetVar(var, value)

    def var_ssa(self, var: SSAVariable) -> MediumLevelILVarSsa:
        return MediumLevelILVarSsa(var)

    def set_var_ssa(self, var: SSAVariable, value: MediumLevelILInstruction) -> MediumLevelILSetVarSsa:
        return MediumLevelILSetVarSsa(var, value)

    # Control flow
    def jump(self, dest: MediumLevelILInstruction) -> MediumLevelILJump:
        return MediumLevelILJump(dest)

    def goto(self, dest: InstructionIndex) -> MediumLevelILGoto:
        return MediumLevelILGoto(dest)

    def if_stmt(self, condition: MediumLevelILInstruction, true_target: InstructionIndex, false_target: InstructionIndex) -> MediumLevelILIf:
        return MediumLevelILIf(condition, true_target, false_target)

    def call(self, dest: MediumLevelILInstruction, arguments: Optional[List[MediumLevelILInstruction]] = None) -> MediumLevelILCall:
        return MediumLevelILCall(dest, arguments)

    def tailcall(self, dest: MediumLevelILInstruction, arguments: Optional[List[MediumLevelILInstruction]] = None) -> MediumLevelILTailcall:
        return MediumLevelILTailcall(dest, arguments)

    def ret(self, sources: Optional[List[MediumLevelILInstruction]] = None) -> MediumLevelILRet:
        return MediumLevelILRet(sources)

    # Constants
    def const(self, value: Union[int, float], size: int = 4) -> MediumLevelILConst:
        return MediumLevelILConst(value, size)

    def const_ptr(self, value: int, size: int = 8) -> MediumLevelILConstPtr:
        return MediumLevelILConstPtr(value, size)

    # Special
    def nop(self) -> MediumLevelILNop:
        return MediumLevelILNop()

    def undef(self) -> MediumLevelILUndef:
        return MediumLevelILUndef()

    def unimpl(self) -> MediumLevelILUnimpl:
        return MediumLevelILUnimpl()
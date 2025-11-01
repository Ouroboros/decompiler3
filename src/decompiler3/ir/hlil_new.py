"""
High Level Intermediate Language (HLIL) - Based on BinaryNinja Design

This module implements HLIL instructions following BinaryNinja's architecture.
HLIL is the highest-level IR with structured control flow and high-level constructs.
"""

from typing import Any, List, Optional, Union, Dict, Tuple
from dataclasses import dataclass

from .common import (
    BaseILInstruction, ILBasicBlock, ILFunction, ILRegister, ILFlag,
    HighLevelILOperation, ILOperationAndSize, InstructionIndex, ExpressionIndex,
    Terminal, ControlFlow, Memory, Arithmetic, Comparison, Constant,
    BinaryOperation, UnaryOperation, Call, Return, Loop
)
from .mlil_new import Variable, SSAVariable


@dataclass(frozen=True)
class HighLevelILOperationAndSize(ILOperationAndSize):
    """HLIL operation with size"""
    operation: HighLevelILOperation


@dataclass
class GotoLabel:
    """Label for goto statements"""
    function: 'HighLevelILFunction'
    id: int

    def __repr__(self):
        return f"<GotoLabel: {self.name}>"

    def __str__(self):
        return self.name

    @property
    def name(self) -> str:
        return f"label_{self.id}"


class HighLevelILInstruction(BaseILInstruction):
    """Base class for all HLIL instructions"""

    def __init__(self, operation: HighLevelILOperation, size: int = 0):
        super().__init__(operation, size)
        self.operation = operation

    def get_expr(self, index: int) -> 'HighLevelILInstruction':
        """Get expression operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")

    def get_int(self, index: int) -> Optional[int]:
        """Get integer operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        return None

    def get_var(self, index: int) -> Variable:
        """Get variable operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")

    def get_var_ssa(self, index: int, version_index: int) -> SSAVariable:
        """Get SSA variable operand at index"""
        if index < len(self.operands) and version_index < len(self.operands):
            var = self.operands[index]
            version = self.operands[version_index]
            return SSAVariable(var, version)
        raise IndexError(f"Operand index out of range")

    def get_label(self, index: int) -> GotoLabel:
        """Get label operand at index"""
        if index < len(self.operands):
            return self.operands[index]
        raise IndexError(f"Operand index {index} out of range")


# ============================================================================
# Arithmetic Instructions
# ============================================================================

@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILAdd(HighLevelILInstruction, Arithmetic, BinaryOperation):
    """Add two operands"""

    def __init__(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4):
        super().__init__(HighLevelILOperation.ADD, size)
        self.operands = [left, right]

    @property
    def left(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def right(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "HighLevelILInstruction"),
            ("right", self.right, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILSub(HighLevelILInstruction, Arithmetic, BinaryOperation):
    """Subtract two operands"""

    def __init__(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4):
        super().__init__(HighLevelILOperation.SUB, size)
        self.operands = [left, right]

    @property
    def left(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def right(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "HighLevelILInstruction"),
            ("right", self.right, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILMul(HighLevelILInstruction, Arithmetic, BinaryOperation):
    """Multiply two operands"""

    def __init__(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4):
        super().__init__(HighLevelILOperation.MUL, size)
        self.operands = [left, right]

    @property
    def left(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def right(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "HighLevelILInstruction"),
            ("right", self.right, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILDiv(HighLevelILInstruction, Arithmetic, BinaryOperation):
    """Divide two operands"""

    def __init__(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4):
        super().__init__(HighLevelILOperation.DIV, size)
        self.operands = [left, right]

    @property
    def left(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def right(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("left", self.left, "HighLevelILInstruction"),
            ("right", self.right, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


# ============================================================================
# Variable Instructions
# ============================================================================

@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILVar(HighLevelILInstruction):
    """Variable reference"""

    def __init__(self, var: Variable):
        super().__init__(HighLevelILOperation.VAR, var.size)
        self.operands = [var]

    @property
    def src(self) -> Variable:
        return self.get_var(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("src", self.src, "Variable"),
        ]

    def __str__(self) -> str:
        return str(self.src)


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILAssign(HighLevelILInstruction):
    """Assignment statement"""

    def __init__(self, dest: HighLevelILInstruction, src: HighLevelILInstruction):
        super().__init__(HighLevelILOperation.ASSIGN, 0)
        self.operands = [dest, src]

    @property
    def dest(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def src(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "HighLevelILInstruction"),
            ("src", self.src, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.dest} = {self.src}"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILVarInit(HighLevelILInstruction):
    """Variable initialization"""

    def __init__(self, dest: Variable, src: HighLevelILInstruction):
        super().__init__(HighLevelILOperation.VAR_INIT, dest.size)
        self.operands = [dest, src]

    @property
    def dest(self) -> Variable:
        return self.get_var(0)

    @property
    def src(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "Variable"),
            ("src", self.src, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"{self.dest.var_type or 'auto'} {self.dest} = {self.src}"


# ============================================================================
# Control Flow Instructions
# ============================================================================

@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILIf(HighLevelILInstruction, ControlFlow):
    """If statement"""

    def __init__(self, condition: HighLevelILInstruction, true_body: HighLevelILInstruction, false_body: Optional[HighLevelILInstruction] = None):
        super().__init__(HighLevelILOperation.IF, 0)
        if false_body is not None:
            self.operands = [condition, true_body, false_body]
        else:
            self.operands = [condition, true_body]

    @property
    def condition(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def true(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def false(self) -> Optional[HighLevelILInstruction]:
        if len(self.operands) > 2:
            return self.get_expr(2)
        return None

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [
            ("condition", self.condition, "HighLevelILInstruction"),
            ("true", self.true, "HighLevelILInstruction"),
        ]
        if self.false is not None:
            operands.append(("false", self.false, "HighLevelILInstruction"))
        return operands

    def __str__(self) -> str:
        if self.false is not None:
            return f"if ({self.condition}) {{\n  {self.true}\n}} else {{\n  {self.false}\n}}"
        return f"if ({self.condition}) {{\n  {self.true}\n}}"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILWhile(HighLevelILInstruction, Loop):
    """While loop"""

    def __init__(self, condition: HighLevelILInstruction, body: HighLevelILInstruction):
        super().__init__(HighLevelILOperation.WHILE, 0)
        self.operands = [condition, body]

    @property
    def condition(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def body(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("condition", self.condition, "HighLevelILInstruction"),
            ("body", self.body, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"while ({self.condition}) {{\n  {self.body}\n}}"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILFor(HighLevelILInstruction, Loop):
    """For loop"""

    def __init__(self, init: HighLevelILInstruction, condition: HighLevelILInstruction, update: HighLevelILInstruction, body: HighLevelILInstruction):
        super().__init__(HighLevelILOperation.FOR, 0)
        self.operands = [init, condition, update, body]

    @property
    def init(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def condition(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def update(self) -> HighLevelILInstruction:
        return self.get_expr(2)

    @property
    def body(self) -> HighLevelILInstruction:
        return self.get_expr(3)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("init", self.init, "HighLevelILInstruction"),
            ("condition", self.condition, "HighLevelILInstruction"),
            ("update", self.update, "HighLevelILInstruction"),
            ("body", self.body, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"for ({self.init}; {self.condition}; {self.update}) {{\n  {self.body}\n}}"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILDoWhile(HighLevelILInstruction, Loop):
    """Do-while loop"""

    def __init__(self, body: HighLevelILInstruction, condition: HighLevelILInstruction):
        super().__init__(HighLevelILOperation.DO_WHILE, 0)
        self.operands = [body, condition]

    @property
    def body(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def condition(self) -> HighLevelILInstruction:
        return self.get_expr(1)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("body", self.body, "HighLevelILInstruction"),
            ("condition", self.condition, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"do {{\n  {self.body}\n}} while ({self.condition})"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILSwitch(HighLevelILInstruction, ControlFlow):
    """Switch statement"""

    def __init__(self, condition: HighLevelILInstruction, default_case: Optional[HighLevelILInstruction], cases: List[Tuple[List[int], HighLevelILInstruction]]):
        super().__init__(HighLevelILOperation.SWITCH, 0)
        self.operands = [condition, default_case, cases]

    @property
    def condition(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def default(self) -> Optional[HighLevelILInstruction]:
        return self.operands[1] if self.operands[1] is not None else None

    @property
    def cases(self) -> List[Tuple[List[int], HighLevelILInstruction]]:
        return self.operands[2]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [
            ("condition", self.condition, "HighLevelILInstruction"),
            ("cases", self.cases, "List[Tuple[List[int], HighLevelILInstruction]]")
        ]
        if self.default is not None:
            operands.append(("default", self.default, "HighLevelILInstruction"))
        return operands

    def __str__(self) -> str:
        result = f"switch ({self.condition}) {{\n"
        for values, body in self.cases:
            for value in values:
                result += f"  case {value}:\n"
            result += f"    {body}\n    break;\n"
        if self.default is not None:
            result += f"  default:\n    {self.default}\n"
        result += "}"
        return result


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILJump(HighLevelILInstruction, Terminal):
    """Unconditional jump"""

    def __init__(self, dest: HighLevelILInstruction):
        super().__init__(HighLevelILOperation.GOTO, 0)
        self.operands = [dest]

    @property
    def dest(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("dest", self.dest, "HighLevelILInstruction"),
        ]

    def __str__(self) -> str:
        return f"jump {self.dest}"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILGoto(HighLevelILInstruction, Terminal):
    """Goto label"""

    def __init__(self, target: GotoLabel):
        super().__init__(HighLevelILOperation.GOTO, 0)
        self.operands = [target]

    @property
    def target(self) -> GotoLabel:
        return self.get_label(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("target", self.target, "GotoLabel"),
        ]

    def __str__(self) -> str:
        return f"goto {self.target}"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILLabel(HighLevelILInstruction):
    """Label definition"""

    def __init__(self, target: GotoLabel):
        super().__init__(HighLevelILOperation.LABEL, 0)
        self.operands = [target]

    @property
    def target(self) -> GotoLabel:
        return self.get_label(0)

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("target", self.target, "GotoLabel"),
        ]

    def __str__(self) -> str:
        return f"{self.target}:"


# ============================================================================
# Function Call Instructions
# ============================================================================

@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILCall(HighLevelILInstruction, Call):
    """Function call"""

    def __init__(self, dest: HighLevelILInstruction, arguments: Optional[List[HighLevelILInstruction]] = None):
        super().__init__(HighLevelILOperation.CALL, 0)
        if arguments is None:
            arguments = []
        self.operands = [dest] + arguments

    @property
    def dest(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def params(self) -> List[HighLevelILInstruction]:
        return self.operands[1:]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [("dest", self.dest, "HighLevelILInstruction")]
        if self.params:
            operands.append(("params", self.params, "List[HighLevelILInstruction]"))
        return operands

    def __str__(self) -> str:
        if self.params:
            args = ", ".join(str(arg) for arg in self.params)
            return f"{self.dest}({args})"
        return f"{self.dest}()"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILTailcall(HighLevelILInstruction, Call, Terminal):
    """Tail call"""

    def __init__(self, dest: HighLevelILInstruction, arguments: Optional[List[HighLevelILInstruction]] = None):
        super().__init__(HighLevelILOperation.TAILCALL, 0)
        if arguments is None:
            arguments = []
        self.operands = [dest] + arguments

    @property
    def dest(self) -> HighLevelILInstruction:
        return self.get_expr(0)

    @property
    def params(self) -> List[HighLevelILInstruction]:
        return self.operands[1:]

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        operands = [("dest", self.dest, "HighLevelILInstruction")]
        if self.params:
            operands.append(("params", self.params, "List[HighLevelILInstruction]"))
        return operands

    def __str__(self) -> str:
        if self.params:
            args = ", ".join(str(arg) for arg in self.params)
            return f"tailcall {self.dest}({args})"
        return f"tailcall {self.dest}()"


@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILRet(HighLevelILInstruction, Return, Terminal):
    """Return from function"""

    def __init__(self, sources: Optional[List[HighLevelILInstruction]] = None):
        super().__init__(HighLevelILOperation.RET, 0)
        if sources is None:
            sources = []
        self.operands = sources

    @property
    def src(self) -> List[HighLevelILInstruction]:
        return self.operands

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        if self.src:
            return [("src", self.src, "List[HighLevelILInstruction]")]
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
# Block Instructions
# ============================================================================

@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILBlock(HighLevelILInstruction):
    """Block of statements"""

    def __init__(self, body: List[HighLevelILInstruction]):
        super().__init__(HighLevelILOperation.BLOCK, 0)
        self.operands = body

    @property
    def body(self) -> List[HighLevelILInstruction]:
        return self.operands

    @property
    def detailed_operands(self) -> List[Tuple[str, Any, str]]:
        return [
            ("body", self.body, "List[HighLevelILInstruction]"),
        ]

    def __str__(self) -> str:
        if not self.body:
            return "{}"
        if len(self.body) == 1:
            return str(self.body[0])

        result = "{\n"
        for stmt in self.body:
            for line in str(stmt).split('\n'):
                result += f"  {line}\n"
        result += "}"
        return result


# ============================================================================
# Constant Instructions
# ============================================================================

@dataclass(frozen=True, repr=False, eq=False)
class HighLevelILConst(HighLevelILInstruction, Constant):
    """Constant value"""

    def __init__(self, value: Union[int, float], size: int = 4):
        super().__init__(HighLevelILOperation.CONST, size)
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


# ============================================================================
# HLIL Function and Basic Block Classes
# ============================================================================

class HighLevelILBasicBlock(ILBasicBlock):
    """HLIL Basic Block"""

    def __init__(self, start_address: int):
        super().__init__(start_address)
        self.instructions: List[HighLevelILInstruction] = []


class HighLevelILFunction(ILFunction):
    """HLIL Function"""

    def __init__(self, name: str, address: Optional[int] = None):
        super().__init__(name, address)
        self.basic_blocks: List[HighLevelILBasicBlock] = []
        self.variables: Dict[str, Variable] = {}
        self.labels: Dict[int, GotoLabel] = {}
        self._next_label_id = 0

    def create_variable(self, name: str, var_type: Optional[str] = None, size: int = 4) -> Variable:
        """Create a new variable in this function"""
        var = Variable(name, var_type, size)
        self.variables[name] = var
        return var

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get variable by name"""
        return self.variables.get(name)

    def create_label(self) -> GotoLabel:
        """Create a new label"""
        label = GotoLabel(self, self._next_label_id)
        self.labels[self._next_label_id] = label
        self._next_label_id += 1
        return label

    def get_label(self, label_id: int) -> Optional[GotoLabel]:
        """Get label by ID"""
        return self.labels.get(label_id)


# ============================================================================
# HLIL Builder (for constructing HLIL)
# ============================================================================

class HighLevelILBuilder:
    """Builder for constructing HLIL instructions"""

    def __init__(self, function: HighLevelILFunction):
        self.function = function
        self.current_block: Optional[HighLevelILBasicBlock] = None

    def set_current_block(self, block: HighLevelILBasicBlock):
        """Set the current basic block for instruction insertion"""
        self.current_block = block

    def add_instruction(self, instruction: HighLevelILInstruction):
        """Add instruction to current block"""
        if self.current_block is None:
            raise RuntimeError("No current block set")
        self.current_block.add_instruction(instruction)

    # Arithmetic operations
    def add(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4) -> HighLevelILAdd:
        return HighLevelILAdd(left, right, size)

    def sub(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4) -> HighLevelILSub:
        return HighLevelILSub(left, right, size)

    def mul(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4) -> HighLevelILMul:
        return HighLevelILMul(left, right, size)

    def div(self, left: HighLevelILInstruction, right: HighLevelILInstruction, size: int = 4) -> HighLevelILDiv:
        return HighLevelILDiv(left, right, size)

    # Variable operations
    def var(self, var: Variable) -> HighLevelILVar:
        return HighLevelILVar(var)

    def assign(self, dest: HighLevelILInstruction, src: HighLevelILInstruction) -> HighLevelILAssign:
        return HighLevelILAssign(dest, src)

    def var_init(self, dest: Variable, src: HighLevelILInstruction) -> HighLevelILVarInit:
        return HighLevelILVarInit(dest, src)

    # Control flow
    def if_stmt(self, condition: HighLevelILInstruction, true_body: HighLevelILInstruction, false_body: Optional[HighLevelILInstruction] = None) -> HighLevelILIf:
        return HighLevelILIf(condition, true_body, false_body)

    def while_loop(self, condition: HighLevelILInstruction, body: HighLevelILInstruction) -> HighLevelILWhile:
        return HighLevelILWhile(condition, body)

    def for_loop(self, init: HighLevelILInstruction, condition: HighLevelILInstruction, update: HighLevelILInstruction, body: HighLevelILInstruction) -> HighLevelILFor:
        return HighLevelILFor(init, condition, update, body)

    def do_while_loop(self, body: HighLevelILInstruction, condition: HighLevelILInstruction) -> HighLevelILDoWhile:
        return HighLevelILDoWhile(body, condition)

    def switch_stmt(self, condition: HighLevelILInstruction, default_case: Optional[HighLevelILInstruction], cases: List[Tuple[List[int], HighLevelILInstruction]]) -> HighLevelILSwitch:
        return HighLevelILSwitch(condition, default_case, cases)

    def jump(self, dest: HighLevelILInstruction) -> HighLevelILJump:
        return HighLevelILJump(dest)

    def goto(self, target: GotoLabel) -> HighLevelILGoto:
        return HighLevelILGoto(target)

    def label(self, target: GotoLabel) -> HighLevelILLabel:
        return HighLevelILLabel(target)

    # Function calls
    def call(self, dest: HighLevelILInstruction, arguments: Optional[List[HighLevelILInstruction]] = None) -> HighLevelILCall:
        return HighLevelILCall(dest, arguments)

    def tailcall(self, dest: HighLevelILInstruction, arguments: Optional[List[HighLevelILInstruction]] = None) -> HighLevelILTailcall:
        return HighLevelILTailcall(dest, arguments)

    def ret(self, sources: Optional[List[HighLevelILInstruction]] = None) -> HighLevelILRet:
        return HighLevelILRet(sources)

    # Block
    def block(self, body: List[HighLevelILInstruction]) -> HighLevelILBlock:
        return HighLevelILBlock(body)

    # Constants
    def const(self, value: Union[int, float], size: int = 4) -> HighLevelILConst:
        return HighLevelILConst(value, size)
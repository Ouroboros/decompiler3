'''
Medium Level IL - Stack-free intermediate representation

MLIL eliminates stack semantics from LLIL, converting stack operations to variables.
This makes data flow analysis and optimization much easier.

Design principles:
- No stack operations (no sp, no STACK[sp+offset])
- Variables instead of stack slots
- Same CFG structure as LLIL
- Each MLIL instruction maps to one or more LLIL instructions
'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

from common import *
from ir.core import *


FLOAT_ROUND_REL_TOL = 1e-6
FLOAT_ROUND_ABS_TOL = 1e-9

if TYPE_CHECKING:
    from ir.llil import LowLevelILBasicBlock, LowLevelILFunction


# === Naming Utilities ===

def mlil_stack_var_name(slot_index: int) -> str:
    '''Generate variable name for stack slot

    Args:
        slot_index: Stack slot index

    Returns:
        Variable name (e.g., 'var_s0', 'var_s1', ...)
    '''
    return f'var_s{slot_index}'


def mlil_arg_var_name(arg_index: int) -> str:
    '''Generate variable name for function argument

    Args:
        arg_index: Argument index

    Returns:
        Variable name (e.g., 'arg_0', 'arg_1', ...)
    '''
    return f'arg{arg_index}'


class MediumLevelILOperation(IntEnum2):
    '''MLIL operations - stack-free version of LLIL'''

    # Constants
    MLIL_CONST              = 0     # Constant value (int/float/str)

    # Variables
    MLIL_VAR                = 10    # Load variable
    MLIL_SET_VAR            = 11    # Store to variable

    # Arithmetic operations
    MLIL_ADD                = 20
    MLIL_SUB                = 21
    MLIL_MUL                = 22
    MLIL_DIV                = 23
    MLIL_MOD                = 24

    # Bitwise operations
    MLIL_AND                = 30
    MLIL_OR                 = 31
    MLIL_XOR                = 32
    MLIL_SHL                = 33
    MLIL_SHR                = 34

    # Logical operations
    MLIL_LOGICAL_AND        = 40
    MLIL_LOGICAL_OR         = 41
    MLIL_LOGICAL_NOT        = 42

    # Comparison operations
    MLIL_EQ                 = 50
    MLIL_NE                 = 51
    MLIL_LT                 = 52
    MLIL_LE                 = 53
    MLIL_GT                 = 54
    MLIL_GE                 = 55

    # Unary operations
    MLIL_NEG                = 60
    MLIL_TEST_ZERO          = 61
    MLIL_ADDRESS_OF         = 62    # Address of variable (&var)

    # Control flow
    MLIL_GOTO               = 70    # Unconditional jump
    MLIL_IF                 = 71    # Conditional branch
    MLIL_RET                = 72    # Return (with optional value)

    # Function calls
    MLIL_CALL               = 80    # Function call
    MLIL_SYSCALL            = 81    # System call

    # Globals
    MLIL_LOAD_GLOBAL        = 90
    MLIL_STORE_GLOBAL       = 91

    # Registers
    MLIL_LOAD_REG           = 100
    MLIL_STORE_REG          = 101

    # Debug
    MLIL_NOP                = 110
    MLIL_DEBUG              = 111

    # Falcom VM specific
    MLIL_CALL_SCRIPT        = 1000

    # SSA (future)
    MLIL_PHI                = 2000
    MLIL_VAR_SSA            = 2001
    MLIL_SET_VAR_SSA        = 2002


class MediumLevelILInstruction(ILInstruction):
    '''Base class for all MLIL instructions

    Unlike LLIL, MLIL instructions don't have stack semantics.
    All values are stored in variables.
    '''

    def __init__(self, operation: MediumLevelILOperation):
        super().__init__()
        self.operation = operation
        self.address = 0
        self.inst_index = -1  # Inherited from LLIL instruction index
        self.llil_index = -1  # Source LLIL instruction index (for debugging/mapping)
        self.options = ILOptions()

    @property
    def operation_name(self) -> str:
        return self.operation.name.replace('MLIL_', '')

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.operation_name}>'


# === Instruction Categories ===

class MediumLevelILExpr(MediumLevelILInstruction):
    '''Base class for value expressions (produces a value)

    Expressions can be used as operands to other instructions.
    Examples: Const, Var, BinaryOp, UnaryOp, LoadGlobal, LoadReg
    '''
    pass


class MediumLevelILStatement(MediumLevelILInstruction):
    '''Base class for statements (side effects, no value)

    Statements have side effects but do not produce values.
    Examples: SetVar, StoreGlobal, StoreReg, Goto, If, Ret, Call
    '''
    pass


# === Variables ===

class MLILVariable:
    '''A variable in MLIL (non-SSA form)

    Variables represent abstract storage locations that replace stack slots.
    Each variable has a name and optionally tracks its source stack slot.
    '''

    def __init__(self, name: str, slot_index: int = -1):
        self.name = name
        self.slot_index = slot_index  # Original stack slot (for debugging)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.slot_index >= 0:
            return f'MLILVariable({self.name}, slot={self.slot_index})'
        return f'MLILVariable({self.name})'

    def __eq__(self, other) -> bool:
        return isinstance(other, MLILVariable) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


# === Constants ===

class MLILConst(MediumLevelILExpr, Constant):
    '''Constant value (int, float, or string)'''

    def __init__(self, value: Any, is_hex: bool = False):
        super().__init__(MediumLevelILOperation.MLIL_CONST)
        self.value = value
        self.is_hex = is_hex

    def __str__(self) -> str:
        if isinstance(self.value, int):
            if self.is_hex:
                return f'0x{self.value:X}'

            return str(self.value)

        elif isinstance(self.value, float):
            precision = default_float_precision_decimals()
            value = self.value
            round_value = round(value, precision)

            rel_tol = FLOAT_ROUND_REL_TOL
            abs_tol = FLOAT_ROUND_ABS_TOL

            if abs(round_value - value) <= max(abs(value) * rel_tol, abs_tol):
                if value != round_value:
                    print(f'float: {precision} {value} -> {round_value}')
                value = round_value

            return f'{value}'

        elif isinstance(self.value, str):

            return f'"{self.value}"'
        return str(self.value)


# === Variable Operations ===

class MLILVar(MediumLevelILExpr):
    '''Load variable value'''

    def __init__(self, var: MLILVariable):
        super().__init__(MediumLevelILOperation.MLIL_VAR)
        self.var = var

    def __str__(self) -> str:
        return str(self.var)


class MLILSetVar(MediumLevelILStatement):
    '''Store value to variable'''

    def __init__(self, var: MLILVariable, value: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_SET_VAR)
        self.var = var
        self.value = value

    def __str__(self) -> str:
        return f'{self.var} = {self.value}'


# === Binary Operations ===

class MLILBinaryOp(MediumLevelILExpr, BinaryOperation):
    '''Base class for binary operations'''

    def __init__(self, operation: MediumLevelILOperation, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(operation)
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        op_map = {
            MediumLevelILOperation.MLIL_ADD: '+',
            MediumLevelILOperation.MLIL_SUB: '-',
            MediumLevelILOperation.MLIL_MUL: '*',
            MediumLevelILOperation.MLIL_DIV: '/',
            MediumLevelILOperation.MLIL_MOD: '%',
            MediumLevelILOperation.MLIL_AND: '&',
            MediumLevelILOperation.MLIL_OR: '|',
            MediumLevelILOperation.MLIL_XOR: '^',
            MediumLevelILOperation.MLIL_SHL: '<<',
            MediumLevelILOperation.MLIL_SHR: '>>',
            MediumLevelILOperation.MLIL_LOGICAL_AND: '&&',
            MediumLevelILOperation.MLIL_LOGICAL_OR: '||',
            MediumLevelILOperation.MLIL_EQ: '==',
            MediumLevelILOperation.MLIL_NE: '!=',
            MediumLevelILOperation.MLIL_LT: '<',
            MediumLevelILOperation.MLIL_LE: '<=',
            MediumLevelILOperation.MLIL_GT: '>',
            MediumLevelILOperation.MLIL_GE: '>=',
        }
        op_str = op_map.get(self.operation, self.operation_name)
        return f'({self.lhs} {op_str} {self.rhs})'


class MLILAdd(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_ADD, lhs, rhs)


class MLILSub(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_SUB, lhs, rhs)


class MLILMul(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_MUL, lhs, rhs)


class MLILDiv(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_DIV, lhs, rhs)


class MLILMod(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_MOD, lhs, rhs)


class MLILAnd(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_AND, lhs, rhs)


class MLILOr(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_OR, lhs, rhs)


class MLILXor(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_XOR, lhs, rhs)


class MLILShl(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_SHL, lhs, rhs)


class MLILShr(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_SHR, lhs, rhs)


class MLILLogicalAnd(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_LOGICAL_AND, lhs, rhs)


class MLILLogicalOr(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_LOGICAL_OR, lhs, rhs)


# === Comparison Operations ===

class MLILEq(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_EQ, lhs, rhs)


class MLILNe(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_NE, lhs, rhs)


class MLILLt(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_LT, lhs, rhs)


class MLILLe(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_LE, lhs, rhs)


class MLILGt(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_GT, lhs, rhs)


class MLILGe(MLILBinaryOp):
    def __init__(self, lhs: MediumLevelILInstruction, rhs: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_GE, lhs, rhs)


# === Unary Operations ===

class MLILUnaryOp(MediumLevelILExpr, UnaryOperation):
    '''Base class for unary operations'''

    def __init__(self, operation: MediumLevelILOperation, operand: MediumLevelILInstruction):
        super().__init__(operation)
        self.operand = operand

    def __str__(self) -> str:
        op_map = {
            MediumLevelILOperation.MLIL_NEG: '-',
            MediumLevelILOperation.MLIL_LOGICAL_NOT: '!',
        }
        op_str = op_map.get(self.operation, self.operation_name)
        return f'{op_str}{self.operand}'


class MLILNeg(MLILUnaryOp):
    def __init__(self, operand: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_NEG, operand)


class MLILLogicalNot(MLILUnaryOp):
    def __init__(self, operand: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_LOGICAL_NOT, operand)


class MLILTestZero(MLILUnaryOp):
    def __init__(self, operand: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_TEST_ZERO, operand)

    def __str__(self) -> str:
        return f'({self.operand} == 0)'


class MLILAddressOf(MLILUnaryOp):
    '''Address-of operation (&var)'''

    def __init__(self, operand: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_ADDRESS_OF, operand)

    def __str__(self) -> str:
        return f'&{self.operand}'


# === Control Flow ===

class MLILGoto(MediumLevelILStatement, Terminal):
    '''Unconditional jump'''

    def __init__(self, target: 'MediumLevelILBasicBlock'):
        super().__init__(MediumLevelILOperation.MLIL_GOTO)
        self.target = target

    def __str__(self) -> str:
        return f'goto {self.target.label}'


class MLILIf(MediumLevelILStatement, Terminal):
    '''Conditional branch'''

    def __init__(self, condition: MediumLevelILInstruction, true_target: 'MediumLevelILBasicBlock', false_target: 'MediumLevelILBasicBlock'):
        super().__init__(MediumLevelILOperation.MLIL_IF)
        self.condition = condition
        self.true_target = true_target
        self.false_target = false_target

    def __str__(self) -> str:
        return f'if ({self.condition}) goto {self.true_target.label} else {self.false_target.label}'


class MLILRet(MediumLevelILStatement, Terminal):
    '''Return from function

    Args:
        value: Return value expression, or None for void return
    '''

    def __init__(self, value: Optional[MediumLevelILInstruction] = None):
        super().__init__(MediumLevelILOperation.MLIL_RET)
        self.value = value

    def __str__(self) -> str:
        if self.value is not None:
            return f'return {self.value}'
        else:
            return 'return'


# === Function Calls ===

class MLILCall(MediumLevelILStatement):
    '''Function call'''

    def __init__(self, target: str, args: List[MediumLevelILInstruction]):
        super().__init__(MediumLevelILOperation.MLIL_CALL)
        self.target = target
        self.args = args

    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.args)
        return f'{self.target}({args_str})'


class MLILSyscall(MediumLevelILStatement):
    '''System call'''

    def __init__(self, subsystem: int, cmd: int, args: List[MediumLevelILInstruction]):
        super().__init__(MediumLevelILOperation.MLIL_SYSCALL)
        self.subsystem = subsystem
        self.cmd = cmd
        self.args = args

    def __str__(self) -> str:
            args = [
                f'{self.subsystem}',
                f'{self.cmd}',
                *[str(arg) for arg in self.args],
            ]

            return f'syscall({', '.join(args)})'


class MLILCallScript(MediumLevelILStatement):
    '''Falcom script call'''

    def __init__(self, module: str, func: str, args: List[MediumLevelILInstruction]):
        super().__init__(MediumLevelILOperation.MLIL_CALL_SCRIPT)
        self.module = module
        self.func = func
        self.args = args

    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.args)
        return f'{self.module}.{self.func}({args_str})'


# === Global Variables ===

class MLILLoadGlobal(MediumLevelILExpr):
    '''Load global variable'''

    def __init__(self, index: int):
        super().__init__(MediumLevelILOperation.MLIL_LOAD_GLOBAL)
        self.index = index

    def __str__(self) -> str:
        return f'GLOBAL[{self.index}]'


class MLILStoreGlobal(MediumLevelILStatement):
    '''Store to global variable'''

    def __init__(self, index: int, value: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_STORE_GLOBAL)
        self.index = index
        self.value = value

    def __str__(self) -> str:
        return f'GLOBAL[{self.index}] = {self.value}'


# === Registers ===

class MLILLoadReg(MediumLevelILExpr):
    '''Load register'''

    def __init__(self, index: int):
        super().__init__(MediumLevelILOperation.MLIL_LOAD_REG)
        self.index = index

    def __str__(self) -> str:
        return f'REG[{self.index}]'


class MLILStoreReg(MediumLevelILStatement):
    '''Store to register'''

    def __init__(self, index: int, value: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_STORE_REG)
        self.index = index
        self.value = value

    def __str__(self) -> str:
        return f'REG[{self.index}] = {self.value}'


# === Debug ===

class MLILNop(MediumLevelILStatement):
    '''No operation'''

    def __init__(self):
        super().__init__(MediumLevelILOperation.MLIL_NOP)

    def __str__(self) -> str:
        return 'nop'


class MLILDebug(MediumLevelILStatement):
    '''Debug information'''

    def __init__(self, debug_type: str, value: Any):
        super().__init__(MediumLevelILOperation.MLIL_DEBUG)
        self.debug_type = debug_type
        self.value = value

    def __str__(self) -> str:
        return f'debug.{self.debug_type}({self.value})'


# === Basic Block ===

class MediumLevelILBasicBlock:
    '''MLIL basic block'''

    def __init__(self, index: int, start: int = 0, label: str = None):
        self.index = index
        self.start = start
        self.label = label or f'mlil_{index}'
        self.instructions: List[MediumLevelILInstruction] = []
        self.incoming_edges: List['MediumLevelILBasicBlock'] = []
        self.outgoing_edges: List['MediumLevelILBasicBlock'] = []
        self.llil_block: Optional[LowLevelILBasicBlock] = None  # Source LLIL block

    def add_instruction(self, inst: MediumLevelILInstruction):
        self.instructions.append(inst)

    def add_outgoing_edge(self, block: 'MediumLevelILBasicBlock'):
        if block not in self.outgoing_edges:
            self.outgoing_edges.append(block)
        if self not in block.incoming_edges:
            block.incoming_edges.append(self)

    @property
    def block_name(self) -> str:
        return f'mlil_block_{self.index}'

    @property
    def has_terminal(self) -> bool:
        return (self.instructions and
                isinstance(self.instructions[-1], Terminal))

    def __str__(self) -> str:
        body = '\n'.join(f'  {inst}' for inst in self.instructions)
        return f'{self.label}:\n{body}'

    def __repr__(self) -> str:
        return f'<MediumLevelILBasicBlock {self.index} "{self.label}">'


# === Function ===

class MediumLevelILFunction:
    '''MLIL function container'''

    def __init__(self, name: str, start_addr: int = 0):
        self.name = name
        self.start_addr = start_addr
        self.basic_blocks: List[MediumLevelILBasicBlock] = []
        self.variables: Dict[str, MLILVariable] = {}
        self.llil_function: Optional[LowLevelILFunction] = None  # Source LLIL function
        self._inst_block_map: Dict[int, MediumLevelILBasicBlock] = {}

    def add_basic_block(self, block: MediumLevelILBasicBlock):
        self.basic_blocks.append(block)

    def create_block(self, start: int = 0, label: str = None) -> MediumLevelILBasicBlock:
        block = MediumLevelILBasicBlock(len(self.basic_blocks), start, label)
        self.add_basic_block(block)
        return block

    def get_or_create_variable(self, name: str, slot_index: int = -1) -> MLILVariable:
        '''Get existing variable or create new one'''
        if name not in self.variables:
            self.variables[name] = MLILVariable(name, slot_index)
        return self.variables[name]

    def register_instruction(self, block: MediumLevelILBasicBlock, inst: MediumLevelILInstruction):
        if inst.inst_index == -1:
            raise RuntimeError('MLIL instruction must have inst_index set')
        self._inst_block_map[inst.inst_index] = block

    def get_block_for_instruction(self, inst_index: int) -> Optional[MediumLevelILBasicBlock]:
        return self._inst_block_map.get(inst_index)

    def iter_blocks(self) -> Iterator[MediumLevelILBasicBlock]:
        return iter(self.basic_blocks)

    def iter_instructions(self) -> Iterator[MediumLevelILInstruction]:
        for block in self.basic_blocks:
            yield from block.instructions

    def __str__(self) -> str:
        result = [f'; ===== MLIL Function {self.name} =====']
        for block in self.basic_blocks:
            result.append('')
            result.append(str(block))
        return '\n'.join(result)

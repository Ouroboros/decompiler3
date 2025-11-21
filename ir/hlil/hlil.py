'''
High Level Intermediate Language (HLIL)

Structured control flow representation for decompiled code.
Converts unstructured MLIL (goto/label) to structured constructs (if/while/for).

HLIL is completely independent from MLIL - it has its own expression and statement types.
'''

from typing import List, Optional, Union
from enum import Enum, auto


class HLILOperation(Enum):
    '''HLIL operation types'''
    # Control flow statements
    HLIL_IF = auto()
    HLIL_WHILE = auto()
    HLIL_DO_WHILE = auto()
    HLIL_FOR = auto()
    HLIL_SWITCH = auto()
    HLIL_BREAK = auto()
    HLIL_CONTINUE = auto()
    HLIL_RETURN = auto()

    # Other statements
    HLIL_BLOCK = auto()
    HLIL_ASSIGN = auto()
    HLIL_EXPR_STMT = auto()
    HLIL_COMMENT = auto()

    # Expressions
    HLIL_VAR = auto()
    HLIL_CONST = auto()
    HLIL_BINARY_OP = auto()
    HLIL_UNARY_OP = auto()
    HLIL_CALL = auto()
    HLIL_SYSCALL = auto()


class HLILInstruction:
    '''Base class for all HLIL instructions'''

    def __init__(self, operation: HLILOperation):
        self.operation = operation

    def __str__(self) -> str:
        return f'{self.operation.name}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class HLILStatement(HLILInstruction):
    '''Base class for HLIL statements'''
    pass


class HLILExpression(HLILInstruction):
    '''Base class for HLIL expressions'''
    pass


# ============================================================================
# Variables and Types
# ============================================================================

class HLILVariable:
    '''HLIL variable with optional type information'''

    def __init__(self, name: str, type_hint: Optional[str] = None):
        self.name = name
        self.type_hint = type_hint  # 'int', 'bool', 'string', 'float', etc.

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if self.type_hint:
            return f'HLILVariable({self.name}: {self.type_hint})'
        return f'HLILVariable({self.name})'

    def __eq__(self, other) -> bool:
        if not isinstance(other, HLILVariable):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


# ============================================================================
# Expressions
# ============================================================================

class HLILVar(HLILExpression):
    '''Variable reference'''

    def __init__(self, var: HLILVariable):
        super().__init__(HLILOperation.HLIL_VAR)
        self.var = var

    def __str__(self) -> str:
        return str(self.var)

    def __repr__(self) -> str:
        return f'HLILVar({self.var.name})'


class HLILConst(HLILExpression):
    '''Constant value'''

    def __init__(self, value: Union[int, float, str, bool]):
        super().__init__(HLILOperation.HLIL_CONST)
        self.value = value

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        elif isinstance(self.value, bool):
            return 'true' if self.value else 'false'
        return str(self.value)

    def __repr__(self) -> str:
        return f'HLILConst({self.value})'


class HLILBinaryOp(HLILExpression):
    '''Binary operation: lhs op rhs'''

    def __init__(self, op: str, lhs: HLILExpression, rhs: HLILExpression):
        super().__init__(HLILOperation.HLIL_BINARY_OP)
        self.op = op  # '+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', '&&', '||', etc.
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return f'{self.lhs} {self.op} {self.rhs}'

    def __repr__(self) -> str:
        return f'HLILBinaryOp({self.op})'


class HLILUnaryOp(HLILExpression):
    '''Unary operation: op operand'''

    def __init__(self, op: str, operand: HLILExpression):
        super().__init__(HLILOperation.HLIL_UNARY_OP)
        self.op = op  # '-', '!', '~', etc.
        self.operand = operand

    def __str__(self) -> str:
        return f'{self.op}{self.operand}'

    def __repr__(self) -> str:
        return f'HLILUnaryOp({self.op})'


class HLILCall(HLILExpression):
    '''Function call'''

    def __init__(self, func_name: str, args: List[HLILExpression]):
        super().__init__(HLILOperation.HLIL_CALL)
        self.func_name = func_name
        self.args = args

    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.args)
        return f'{self.func_name}({args_str})'

    def __repr__(self) -> str:
        return f'HLILCall({self.func_name}, {len(self.args)} args)'


class HLILSyscall(HLILExpression):
    '''System call'''

    def __init__(self, subsystem: str, cmd: str, args: List[HLILExpression]):
        super().__init__(HLILOperation.HLIL_SYSCALL)
        self.subsystem = subsystem
        self.cmd = cmd
        self.args = args

    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.args)
        return f'{self.subsystem}.{self.cmd}({args_str})'

    def __repr__(self) -> str:
        return f'HLILSyscall({self.subsystem}.{self.cmd})'


# ============================================================================
# Control Flow Statements
# ============================================================================

class HLILBlock(HLILStatement):
    '''Block of statements

    Represents a sequence of statements executed in order.
    '''

    def __init__(self, statements: Optional[List[HLILStatement]] = None):
        super().__init__(HLILOperation.HLIL_BLOCK)
        self.statements = statements or []

    def add_statement(self, stmt: HLILStatement):
        '''Add a statement to the block'''
        self.statements.append(stmt)

    def __str__(self) -> str:
        if not self.statements:
            return '{ }'
        return f'{{ {len(self.statements)} statements }}'

    def __repr__(self) -> str:
        return f'HLILBlock({len(self.statements)} stmts)'


class HLILIf(HLILStatement):
    '''Conditional statement

    if (condition) {
        true_block
    } else {
        false_block
    }
    '''

    def __init__(self, condition: HLILExpression, true_block: 'HLILBlock', false_block: Optional['HLILBlock'] = None):
        super().__init__(HLILOperation.HLIL_IF)
        self.condition = condition
        self.true_block = true_block
        self.false_block = false_block

    def __str__(self) -> str:
        if self.false_block:
            return f'if ({self.condition}) {{ ... }} else {{ ... }}'
        return f'if ({self.condition}) {{ ... }}'

    def __repr__(self) -> str:
        return f'HLILIf(cond={self.condition})'


class HLILWhile(HLILStatement):
    '''While loop

    while (condition) {
        body
    }
    '''

    def __init__(self, condition: HLILExpression, body: 'HLILBlock'):
        super().__init__(HLILOperation.HLIL_WHILE)
        self.condition = condition
        self.body = body

    def __str__(self) -> str:
        return f'while ({self.condition}) {{ ... }}'

    def __repr__(self) -> str:
        return f'HLILWhile(cond={self.condition})'


class HLILDoWhile(HLILStatement):
    '''Do-while loop

    do {
        body
    } while (condition);
    '''

    def __init__(self, condition: HLILExpression, body: 'HLILBlock'):
        super().__init__(HLILOperation.HLIL_DO_WHILE)
        self.condition = condition
        self.body = body

    def __str__(self) -> str:
        return f'do {{ ... }} while ({self.condition})'

    def __repr__(self) -> str:
        return f'HLILDoWhile(cond={self.condition})'


class HLILFor(HLILStatement):
    '''For loop

    for (init; condition; update) {
        body
    }
    '''

    def __init__(self, init: Optional[HLILStatement], condition: Optional[HLILExpression],
                 update: Optional[HLILStatement], body: 'HLILBlock'):
        super().__init__(HLILOperation.HLIL_FOR)
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __str__(self) -> str:
        return f'for ({self.init}; {self.condition}; {self.update}) {{ ... }}'

    def __repr__(self) -> str:
        return f'HLILFor()'


class HLILSwitchCase:
    '''Switch case clause

    case value:
        body
    '''

    def __init__(self, value: Optional[HLILExpression], body: 'HLILBlock'):
        self.value = value  # None for default case
        self.body = body

    def is_default(self) -> bool:
        '''Check if this is the default case'''
        return self.value is None

    def __str__(self) -> str:
        if self.is_default():
            return 'default: { ... }'
        return f'case {self.value}: {{ ... }}'

    def __repr__(self) -> str:
        if self.is_default():
            return 'HLILSwitchCase(default)'
        return f'HLILSwitchCase({self.value})'


class HLILSwitch(HLILStatement):
    '''Switch statement

    switch (scrutinee) {
        case val1: body1
        case val2: body2
        default: default_body
    }
    '''

    def __init__(self, scrutinee: HLILExpression, cases: List[HLILSwitchCase]):
        super().__init__(HLILOperation.HLIL_SWITCH)
        self.scrutinee = scrutinee
        self.cases = cases

    def __str__(self) -> str:
        return f'switch ({self.scrutinee}) {{ {len(self.cases)} cases }}'

    def __repr__(self) -> str:
        return f'HLILSwitch({len(self.cases)} cases)'


class HLILBreak(HLILStatement):
    '''Break statement

    break;
    '''

    def __init__(self):
        super().__init__(HLILOperation.HLIL_BREAK)

    def __str__(self) -> str:
        return 'break'

    def __repr__(self) -> str:
        return 'HLILBreak()'


class HLILContinue(HLILStatement):
    '''Continue statement

    continue;
    '''

    def __init__(self):
        super().__init__(HLILOperation.HLIL_CONTINUE)

    def __str__(self) -> str:
        return 'continue'

    def __repr__(self) -> str:
        return 'HLILContinue()'


class HLILReturn(HLILStatement):
    '''Return statement

    return [value];
    '''

    def __init__(self, value: Optional[HLILExpression] = None):
        super().__init__(HLILOperation.HLIL_RETURN)
        self.value = value

    def __str__(self) -> str:
        if self.value is not None:
            return f'return {self.value}'
        return 'return'

    def __repr__(self) -> str:
        return f'HLILReturn({self.value})'


# ============================================================================
# Other Statements
# ============================================================================

class HLILAssign(HLILStatement):
    '''Assignment statement

    dest = src;
    '''

    def __init__(self, dest: HLILExpression, src: HLILExpression):
        super().__init__(HLILOperation.HLIL_ASSIGN)
        self.dest = dest
        self.src = src

    def __str__(self) -> str:
        return f'{self.dest} = {self.src}'

    def __repr__(self) -> str:
        return f'HLILAssign({self.dest} = {self.src})'


class HLILExprStmt(HLILStatement):
    '''Expression statement

    expression;

    Used for function calls and other expressions executed for side effects.
    '''

    def __init__(self, expr: HLILExpression):
        super().__init__(HLILOperation.HLIL_EXPR_STMT)
        self.expr = expr

    def __str__(self) -> str:
        return f'{self.expr}'

    def __repr__(self) -> str:
        return f'HLILExprStmt({self.expr})'


class HLILComment(HLILStatement):
    '''Comment statement

    // comment

    Used for debug information and annotations.
    '''

    def __init__(self, text: str):
        super().__init__(HLILOperation.HLIL_COMMENT)
        self.text = text

    def __str__(self) -> str:
        return f'// {self.text}'

    def __repr__(self) -> str:
        return f'HLILComment({self.text})'


# ============================================================================
# Function Container
# ============================================================================

class HighLevelILFunction:
    '''HLIL function container

    Represents a function in structured high-level form.
    '''

    def __init__(self, name: str, start_addr: int = 0):
        self.name = name
        self.start_addr = start_addr
        self.body = HLILBlock()
        self.variables: List[HLILVariable] = []  # Local variables
        self.parameters: List[HLILVariable] = []  # Parameters

    def add_statement(self, stmt: HLILStatement):
        '''Add a statement to the function body'''
        self.body.add_statement(stmt)

    def __str__(self) -> str:
        return f'HighLevelILFunction({self.name}, {len(self.body.statements)} stmts)'

    def __repr__(self) -> str:
        return f'HighLevelILFunction({self.name})'

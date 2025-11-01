"""
LLIL (Low Level Intermediate Language)

Similar to BinaryNinja's LLIL, this represents code close to the assembly level:
- Stack-based operations
- Register references
- Memory operations with explicit addressing
- Direct jumps and calls
- No variable abstraction
"""

from typing import List, Optional, Union, Any
from dataclasses import dataclass
from .base import (
    IRExpression, IRVariable, IRBasicBlock, IRFunction, IRVisitor,
    OperationType, SourceLocation
)


class LLILExpression(IRExpression):
    """Base LLIL expression"""

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_llil_expression(self)


class LLILRegister(LLILExpression):
    """Register reference"""

    def __init__(self, register: str, size: int = 4):
        super().__init__(OperationType.VAR, size)
        self.register = register

    def __str__(self) -> str:
        return f"reg_{self.register}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_register(self)


class LLILStack(LLILExpression):
    """Stack pointer relative access"""

    def __init__(self, offset: int, size: int = 4):
        super().__init__(OperationType.VAR, size)
        self.offset = offset

    def __str__(self) -> str:
        if self.offset >= 0:
            return f"stack+{self.offset}"
        return f"stack{self.offset}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_stack(self)


class LLILConstant(LLILExpression):
    """Constant value"""

    def __init__(self, value: Union[int, float, str], size: int = 4):
        super().__init__(OperationType.CONST, size)
        self.value = value

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_constant(self)


class LLILBinaryOp(LLILExpression):
    """Binary operation"""

    def __init__(self, operation: OperationType, left: LLILExpression, right: LLILExpression, size: int = 4):
        super().__init__(operation, size)
        self.left = left
        self.right = right
        self.operands = [left, right]

    def __str__(self) -> str:
        op_symbols = {
            OperationType.ADD: "+",
            OperationType.SUB: "-",
            OperationType.MUL: "*",
            OperationType.DIV: "/",
            OperationType.MOD: "%",
            OperationType.AND: "&",
            OperationType.OR: "|",
            OperationType.XOR: "^",
            OperationType.LSL: "<<",
            OperationType.LSR: ">>",
            OperationType.ASR: ">>",
            OperationType.CMP_E: "==",
            OperationType.CMP_NE: "!=",
            OperationType.CMP_SLT: "<",
            OperationType.CMP_ULT: "<u",
            OperationType.CMP_SLE: "<=",
            OperationType.CMP_ULE: "<=u",
        }
        symbol = op_symbols.get(self.operation, "?")
        return f"({self.left} {symbol} {self.right})"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_binary_op(self)


class LLILUnaryOp(LLILExpression):
    """Unary operation"""

    def __init__(self, operation: OperationType, operand: LLILExpression, size: int = 4):
        super().__init__(operation, size)
        self.operand = operand
        self.operands = [operand]

    def __str__(self) -> str:
        op_symbols = {
            OperationType.NEG: "-",
            OperationType.NOT: "~",
        }
        symbol = op_symbols.get(self.operation, "?")
        return f"{symbol}{self.operand}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_unary_op(self)


class LLILLoad(LLILExpression):
    """Memory load operation"""

    def __init__(self, address: LLILExpression, size: int = 4):
        super().__init__(OperationType.LOAD, size)
        self.address = address
        self.operands = [address]

    def __str__(self) -> str:
        return f"*({self.address})"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_load(self)


class LLILStore(LLILExpression):
    """Memory store operation"""

    def __init__(self, address: LLILExpression, value: LLILExpression, size: int = 4):
        super().__init__(OperationType.STORE, size)
        self.address = address
        self.value = value
        self.operands = [address, value]

    def __str__(self) -> str:
        return f"*({self.address}) = {self.value}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_store(self)


class LLILCall(LLILExpression):
    """Function call"""

    def __init__(self, target: LLILExpression, arguments: List[LLILExpression], size: int = 4):
        super().__init__(OperationType.CALL, size)
        self.target = target
        self.arguments = arguments
        self.operands = [target] + arguments

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"call {self.target}({args_str})"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_call(self)


class LLILJump(LLILExpression):
    """Unconditional jump"""

    def __init__(self, target: Union[LLILExpression, int]):
        super().__init__(OperationType.JUMP, 0)
        self.target = target
        if isinstance(target, LLILExpression):
            self.operands = [target]

    def __str__(self) -> str:
        return f"goto {self.target}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_jump(self)


class LLILIf(LLILExpression):
    """Conditional branch"""

    def __init__(self, condition: LLILExpression, true_target: Union[LLILExpression, int],
                 false_target: Optional[Union[LLILExpression, int]] = None):
        super().__init__(OperationType.IF, 0)
        self.condition = condition
        self.true_target = true_target
        self.false_target = false_target
        self.operands = [condition]
        if isinstance(true_target, LLILExpression):
            self.operands.append(true_target)
        if isinstance(false_target, LLILExpression):
            self.operands.append(false_target)

    def __str__(self) -> str:
        if self.false_target:
            return f"if ({self.condition}) goto {self.true_target} else goto {self.false_target}"
        return f"if ({self.condition}) goto {self.true_target}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_if(self)


class LLILReturn(LLILExpression):
    """Return from function"""

    def __init__(self, value: Optional[LLILExpression] = None):
        super().__init__(OperationType.RET, 0)
        self.value = value
        if value:
            self.operands = [value]

    def __str__(self) -> str:
        if self.value:
            return f"return {self.value}"
        return "return"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_return(self)


class LLILPush(LLILExpression):
    """Push to stack (stack machine specific)"""

    def __init__(self, value: LLILExpression, size: int = 4):
        super().__init__(OperationType.STORE, size)
        self.value = value
        self.operands = [value]

    def __str__(self) -> str:
        return f"push {self.value}"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_push(self)


class LLILPop(LLILExpression):
    """Pop from stack (stack machine specific)"""

    def __init__(self, size: int = 4):
        super().__init__(OperationType.LOAD, size)

    def __str__(self) -> str:
        return "pop"

    def accept(self, visitor: 'LLILVisitor') -> Any:
        return visitor.visit_pop(self)


class LLILVisitor(IRVisitor):
    """Visitor for LLIL expressions"""

    def visit_expression(self, expr: IRExpression) -> Any:
        if isinstance(expr, LLILExpression):
            return self.visit_llil_expression(expr)
        raise NotImplementedError(f"Unknown expression type: {type(expr)}")

    def visit_llil_expression(self, expr: LLILExpression) -> Any:
        """Generic LLIL expression visitor"""
        return expr.accept(self)

    def visit_register(self, expr: LLILRegister) -> Any:
        pass

    def visit_stack(self, expr: LLILStack) -> Any:
        pass

    def visit_constant(self, expr: LLILConstant) -> Any:
        pass

    def visit_binary_op(self, expr: LLILBinaryOp) -> Any:
        pass

    def visit_unary_op(self, expr: LLILUnaryOp) -> Any:
        pass

    def visit_load(self, expr: LLILLoad) -> Any:
        pass

    def visit_store(self, expr: LLILStore) -> Any:
        pass

    def visit_call(self, expr: LLILCall) -> Any:
        pass

    def visit_jump(self, expr: LLILJump) -> Any:
        pass

    def visit_if(self, expr: LLILIf) -> Any:
        pass

    def visit_return(self, expr: LLILReturn) -> Any:
        pass

    def visit_push(self, expr: LLILPush) -> Any:
        pass

    def visit_pop(self, expr: LLILPop) -> Any:
        pass


class LLILBuilder:
    """Builder for constructing LLIL expressions"""

    def __init__(self, function: IRFunction):
        self.function = function
        self.current_block: Optional[IRBasicBlock] = None

    def set_current_block(self, block: IRBasicBlock):
        """Set current basic block for instruction insertion"""
        self.current_block = block

    def add_instruction(self, expr: LLILExpression):
        """Add instruction to current block"""
        if not self.current_block:
            raise ValueError("No current block set")
        self.current_block.add_instruction(expr)

    # Convenience methods for building expressions
    def const(self, value: Union[int, float, str], size: int = 4) -> LLILConstant:
        return LLILConstant(value, size)

    def reg(self, register: str, size: int = 4) -> LLILRegister:
        return LLILRegister(register, size)

    def stack(self, offset: int, size: int = 4) -> LLILStack:
        return LLILStack(offset, size)

    def load(self, address: LLILExpression, size: int = 4) -> LLILLoad:
        return LLILLoad(address, size)

    def store(self, address: LLILExpression, value: LLILExpression, size: int = 4) -> LLILStore:
        return LLILStore(address, value, size)

    def add(self, left: LLILExpression, right: LLILExpression, size: int = 4) -> LLILBinaryOp:
        return LLILBinaryOp(OperationType.ADD, left, right, size)

    def sub(self, left: LLILExpression, right: LLILExpression, size: int = 4) -> LLILBinaryOp:
        return LLILBinaryOp(OperationType.SUB, left, right, size)

    def call(self, target: LLILExpression, arguments: List[LLILExpression]) -> LLILCall:
        return LLILCall(target, arguments)

    def jump(self, target: Union[LLILExpression, int]) -> LLILJump:
        return LLILJump(target)

    def if_then(self, condition: LLILExpression, true_target: Union[LLILExpression, int],
               false_target: Optional[Union[LLILExpression, int]] = None) -> LLILIf:
        return LLILIf(condition, true_target, false_target)

    def ret(self, value: Optional[LLILExpression] = None) -> LLILReturn:
        return LLILReturn(value)


# Convenience functions for creating LLIL
def llil_const(value: Union[int, float, str], size: int = 4) -> LLILConstant:
    return LLILConstant(value, size)

def llil_reg(register: str, size: int = 4) -> LLILRegister:
    return LLILRegister(register, size)

def llil_add(left: LLILExpression, right: LLILExpression, size: int = 4) -> LLILBinaryOp:
    return LLILBinaryOp(OperationType.ADD, left, right, size)
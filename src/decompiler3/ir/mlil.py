"""
MLIL (Medium Level Intermediate Language)

Similar to BinaryNinja's MLIL, this represents code with:
- Variables instead of registers/stack locations
- Structured control flow
- No stack manipulation
- Function calls with proper argument passing
- Basic type information
"""

from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
from .base import (
    IRExpression, IRVariable, IRBasicBlock, IRFunction, IRVisitor,
    OperationType, SourceLocation
)


class MLILExpression(IRExpression):
    """Base MLIL expression"""

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_mlil_expression(self)


class MLILVariable(MLILExpression):
    """Variable reference"""

    def __init__(self, variable: IRVariable):
        super().__init__(OperationType.VAR, variable.size)
        self.variable = variable

    def __str__(self) -> str:
        return str(self.variable)

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_variable(self)


class MLILConstant(MLILExpression):
    """Constant value"""

    def __init__(self, value: Union[int, float, str, bool], size: int = 4, const_type: Optional[str] = None):
        super().__init__(OperationType.CONST, size)
        self.value = value
        self.const_type = const_type

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        elif isinstance(self.value, bool):
            return "true" if self.value else "false"
        return str(self.value)

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_constant(self)


class MLILBinaryOp(MLILExpression):
    """Binary operation"""

    def __init__(self, operation: OperationType, left: MLILExpression, right: MLILExpression,
                 size: int = 4, result_type: Optional[str] = None):
        super().__init__(operation, size)
        self.left = left
        self.right = right
        self.result_type = result_type
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

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_binary_op(self)


class MLILUnaryOp(MLILExpression):
    """Unary operation"""

    def __init__(self, operation: OperationType, operand: MLILExpression,
                 size: int = 4, result_type: Optional[str] = None):
        super().__init__(operation, size)
        self.operand = operand
        self.result_type = result_type
        self.operands = [operand]

    def __str__(self) -> str:
        op_symbols = {
            OperationType.NEG: "-",
            OperationType.NOT: "~",
        }
        symbol = op_symbols.get(self.operation, "?")
        if self.operation == OperationType.NOT:
            return f"!({self.operand})"
        return f"{symbol}{self.operand}"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_unary_op(self)


class MLILAssignment(MLILExpression):
    """Variable assignment"""

    def __init__(self, dest: MLILVariable, source: MLILExpression):
        super().__init__(OperationType.STORE, dest.size)
        self.dest = dest
        self.source = source
        self.operands = [dest, source]

    def __str__(self) -> str:
        return f"{self.dest} = {self.source}"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_assignment(self)


class MLILLoad(MLILExpression):
    """Memory load (pointer dereference)"""

    def __init__(self, address: MLILExpression, size: int = 4, load_type: Optional[str] = None):
        super().__init__(OperationType.LOAD, size)
        self.address = address
        self.load_type = load_type
        self.operands = [address]

    def __str__(self) -> str:
        return f"*({self.address})"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_load(self)


class MLILStore(MLILExpression):
    """Memory store (pointer assignment)"""

    def __init__(self, address: MLILExpression, value: MLILExpression, size: int = 4):
        super().__init__(OperationType.STORE, size)
        self.address = address
        self.value = value
        self.operands = [address, value]

    def __str__(self) -> str:
        return f"*({self.address}) = {self.value}"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_store(self)


class MLILFieldAccess(MLILExpression):
    """Structure/object field access"""

    def __init__(self, base: MLILExpression, field: str, size: int = 4, field_type: Optional[str] = None):
        super().__init__(OperationType.VAR_FIELD, size)
        self.base = base
        self.field = field
        self.field_type = field_type
        self.operands = [base]

    def __str__(self) -> str:
        return f"{self.base}.{self.field}"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_field_access(self)


class MLILCall(MLILExpression):
    """Function call"""

    def __init__(self, target: MLILExpression, arguments: List[MLILExpression],
                 size: int = 4, return_type: Optional[str] = None):
        super().__init__(OperationType.CALL, size)
        self.target = target
        self.arguments = arguments
        self.return_type = return_type
        self.operands = [target] + arguments

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.target}({args_str})"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_call(self)


class MLILBuiltinCall(MLILExpression):
    """Built-in function call"""

    def __init__(self, builtin_name: str, arguments: List[MLILExpression],
                 size: int = 4, return_type: Optional[str] = None):
        super().__init__(OperationType.BUILTIN_CALL, size)
        self.builtin_name = builtin_name
        self.arguments = arguments
        self.return_type = return_type
        self.operands = arguments

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"__builtin_{self.builtin_name}({args_str})"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_builtin_call(self)


class MLILJump(MLILExpression):
    """Unconditional jump"""

    def __init__(self, target: Union[MLILExpression, int, str]):
        super().__init__(OperationType.JUMP, 0)
        self.target = target
        if isinstance(target, MLILExpression):
            self.operands = [target]

    def __str__(self) -> str:
        return f"goto {self.target}"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_jump(self)


class MLILIf(MLILExpression):
    """Conditional branch"""

    def __init__(self, condition: MLILExpression, true_target: Union[MLILExpression, int, str],
                 false_target: Optional[Union[MLILExpression, int, str]] = None):
        super().__init__(OperationType.IF, 0)
        self.condition = condition
        self.true_target = true_target
        self.false_target = false_target
        self.operands = [condition]
        if isinstance(true_target, MLILExpression):
            self.operands.append(true_target)
        if isinstance(false_target, MLILExpression):
            self.operands.append(false_target)

    def __str__(self) -> str:
        if self.false_target:
            return f"if ({self.condition}) goto {self.true_target} else goto {self.false_target}"
        return f"if ({self.condition}) goto {self.true_target}"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_if(self)


class MLILReturn(MLILExpression):
    """Return from function"""

    def __init__(self, value: Optional[MLILExpression] = None):
        super().__init__(OperationType.RET, 0)
        self.value = value
        if value:
            self.operands = [value]

    def __str__(self) -> str:
        if self.value:
            return f"return {self.value}"
        return "return"

    def accept(self, visitor: 'MLILVisitor') -> Any:
        return visitor.visit_return(self)


class MLILVisitor(IRVisitor):
    """Visitor for MLIL expressions"""

    def visit_expression(self, expr: IRExpression) -> Any:
        if isinstance(expr, MLILExpression):
            return self.visit_mlil_expression(expr)
        raise NotImplementedError(f"Unknown expression type: {type(expr)}")

    def visit_mlil_expression(self, expr: MLILExpression) -> Any:
        """Generic MLIL expression visitor"""
        return expr.accept(self)

    def visit_variable(self, expr: MLILVariable) -> Any:
        pass

    def visit_constant(self, expr: MLILConstant) -> Any:
        pass

    def visit_binary_op(self, expr: MLILBinaryOp) -> Any:
        pass

    def visit_unary_op(self, expr: MLILUnaryOp) -> Any:
        pass

    def visit_assignment(self, expr: MLILAssignment) -> Any:
        pass

    def visit_load(self, expr: MLILLoad) -> Any:
        pass

    def visit_store(self, expr: MLILStore) -> Any:
        pass

    def visit_field_access(self, expr: MLILFieldAccess) -> Any:
        pass

    def visit_call(self, expr: MLILCall) -> Any:
        pass

    def visit_builtin_call(self, expr: MLILBuiltinCall) -> Any:
        pass

    def visit_jump(self, expr: MLILJump) -> Any:
        pass

    def visit_if(self, expr: MLILIf) -> Any:
        pass

    def visit_return(self, expr: MLILReturn) -> Any:
        pass


class MLILBuilder:
    """Builder for constructing MLIL expressions"""

    def __init__(self, function: IRFunction):
        self.function = function
        self.current_block: Optional[IRBasicBlock] = None

    def set_current_block(self, block: IRBasicBlock):
        """Set current basic block for instruction insertion"""
        self.current_block = block

    def add_instruction(self, expr: MLILExpression):
        """Add instruction to current block"""
        if not self.current_block:
            raise ValueError("No current block set")
        self.current_block.add_instruction(expr)

    # Convenience methods for building expressions
    def const(self, value: Union[int, float, str, bool], size: int = 4, const_type: Optional[str] = None) -> MLILConstant:
        return MLILConstant(value, size, const_type)

    def var(self, variable: IRVariable) -> MLILVariable:
        return MLILVariable(variable)

    def assign(self, dest: MLILVariable, source: MLILExpression) -> MLILAssignment:
        return MLILAssignment(dest, source)

    def load(self, address: MLILExpression, size: int = 4, load_type: Optional[str] = None) -> MLILLoad:
        return MLILLoad(address, size, load_type)

    def store(self, address: MLILExpression, value: MLILExpression, size: int = 4) -> MLILStore:
        return MLILStore(address, value, size)

    def field_access(self, base: MLILExpression, field: str, size: int = 4, field_type: Optional[str] = None) -> MLILFieldAccess:
        return MLILFieldAccess(base, field, size, field_type)

    def add(self, left: MLILExpression, right: MLILExpression, size: int = 4, result_type: Optional[str] = None) -> MLILBinaryOp:
        return MLILBinaryOp(OperationType.ADD, left, right, size, result_type)

    def sub(self, left: MLILExpression, right: MLILExpression, size: int = 4, result_type: Optional[str] = None) -> MLILBinaryOp:
        return MLILBinaryOp(OperationType.SUB, left, right, size, result_type)

    def mul(self, left: MLILExpression, right: MLILExpression, size: int = 4, result_type: Optional[str] = None) -> MLILBinaryOp:
        return MLILBinaryOp(OperationType.MUL, left, right, size, result_type)

    def call(self, target: MLILExpression, arguments: List[MLILExpression],
             size: int = 4, return_type: Optional[str] = None) -> MLILCall:
        return MLILCall(target, arguments, size, return_type)

    def builtin_call(self, builtin_name: str, arguments: List[MLILExpression],
                    size: int = 4, return_type: Optional[str] = None) -> MLILBuiltinCall:
        return MLILBuiltinCall(builtin_name, arguments, size, return_type)

    def jump(self, target: Union[MLILExpression, int, str]) -> MLILJump:
        return MLILJump(target)

    def if_then(self, condition: MLILExpression, true_target: Union[MLILExpression, int, str],
               false_target: Optional[Union[MLILExpression, int, str]] = None) -> MLILIf:
        return MLILIf(condition, true_target, false_target)

    def ret(self, value: Optional[MLILExpression] = None) -> MLILReturn:
        return MLILReturn(value)

    def not_op(self, operand: MLILExpression, size: int = 4) -> MLILUnaryOp:
        return MLILUnaryOp(OperationType.NOT, operand, size)

    def negate(self, operand: MLILExpression, size: int = 4) -> MLILUnaryOp:
        return MLILUnaryOp(OperationType.NEG, operand, size)


# Convenience functions for creating MLIL
def mlil_const(value: Union[int, float, str, bool], size: int = 4) -> MLILConstant:
    return MLILConstant(value, size)

def mlil_var(variable: IRVariable) -> MLILVariable:
    return MLILVariable(variable)

def mlil_add(left: MLILExpression, right: MLILExpression, size: int = 4) -> MLILBinaryOp:
    return MLILBinaryOp(OperationType.ADD, left, right, size)
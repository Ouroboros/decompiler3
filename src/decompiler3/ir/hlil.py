"""
HLIL (High Level Intermediate Language)

Similar to BinaryNinja's HLIL, this represents high-level structured code:
- High-level control structures (for, while, switch)
- Proper type information
- Object-oriented constructs
- Ready for decompilation to TypeScript
- Semantic equivalence to high-level languages
"""

from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
from .base import (
    IRExpression, IRVariable, IRBasicBlock, IRFunction, IRVisitor,
    OperationType, SourceLocation, IRType
)


class HLILExpression(IRExpression):
    """Base HLIL expression"""

    def __init__(self, operation: OperationType, size: int = 4, expr_type: Optional[IRType] = None):
        super().__init__(operation, size)
        self.expr_type = expr_type or IRType.ANY

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_hlil_expression(self)


class HLILVariable(HLILExpression):
    """Variable reference with type information"""

    def __init__(self, variable: IRVariable, var_type: Optional[IRType] = None):
        super().__init__(OperationType.VAR, variable.size, var_type)
        self.variable = variable
        if var_type:
            self.variable.var_type = var_type

    def __str__(self) -> str:
        type_str = self.expr_type.to_string() if isinstance(self.expr_type, IRType) else str(self.expr_type)
        return f"{self.variable}: {type_str}"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_variable(self)


class HLILConstant(HLILExpression):
    """Typed constant value"""

    def __init__(self, value: Union[int, float, str, bool], size: int = 4, const_type: Optional[IRType] = None):
        # Infer type from value if not provided
        inferred_type = const_type
        if not inferred_type:
            if isinstance(value, bool):
                inferred_type = IRType.BOOLEAN
            elif isinstance(value, (int, float)):
                inferred_type = IRType.NUMBER
            elif isinstance(value, str):
                inferred_type = IRType.STRING
            else:
                inferred_type = IRType.ANY

        super().__init__(OperationType.CONST, size, inferred_type)
        self.value = value

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        elif isinstance(self.value, bool):
            return "true" if self.value else "false"
        elif self.expr_type == IRType.NUMBER and isinstance(self.value, (int, float)):
            return str(self.value)
        return str(self.value)

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_constant(self)


class HLILBinaryOp(HLILExpression):
    """Binary operation with type inference"""

    def __init__(self, operation: OperationType, left: HLILExpression, right: HLILExpression,
                 size: int = 4, result_type: Optional[IRType] = None):
        super().__init__(operation, size, result_type)
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
            OperationType.CMP_E: "===",  # TypeScript strict equality
            OperationType.CMP_NE: "!==",
            OperationType.CMP_SLT: "<",
            OperationType.CMP_ULT: "<",
            OperationType.CMP_SLE: "<=",
            OperationType.CMP_ULE: "<=",
        }
        symbol = op_symbols.get(self.operation, "?")
        return f"({self.left} {symbol} {self.right})"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_binary_op(self)


class HLILUnaryOp(HLILExpression):
    """Unary operation"""

    def __init__(self, operation: OperationType, operand: HLILExpression,
                 size: int = 4, result_type: Optional[IRType] = None):
        super().__init__(operation, size, result_type)
        self.operand = operand
        self.operands = [operand]

    def __str__(self) -> str:
        op_symbols = {
            OperationType.NEG: "-",
            OperationType.NOT: "!",  # Logical NOT for TypeScript
        }
        symbol = op_symbols.get(self.operation, "?")
        return f"{symbol}{self.operand}"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_unary_op(self)


class HLILAssignment(HLILExpression):
    """Variable assignment with type checking"""

    def __init__(self, dest: HLILVariable, source: HLILExpression):
        super().__init__(OperationType.STORE, dest.size)
        self.dest = dest
        self.source = source
        self.operands = [dest, source]

    def __str__(self) -> str:
        return f"{self.dest} = {self.source}"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_assignment(self)


class HLILFieldAccess(HLILExpression):
    """Object property/field access"""

    def __init__(self, base: HLILExpression, field: str, size: int = 4, field_type: Optional[IRType] = None):
        super().__init__(OperationType.VAR_FIELD, size, field_type)
        self.base = base
        self.field = field
        self.operands = [base]

    def __str__(self) -> str:
        return f"{self.base}.{self.field}"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_field_access(self)


class HLILArrayAccess(HLILExpression):
    """Array/indexer access"""

    def __init__(self, base: HLILExpression, index: HLILExpression,
                 size: int = 4, element_type: Optional[IRType] = None):
        super().__init__(OperationType.LOAD, size, element_type)
        self.base = base
        self.index = index
        self.operands = [base, index]

    def __str__(self) -> str:
        return f"{self.base}[{self.index}]"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_array_access(self)


class HLILCall(HLILExpression):
    """Function call with type signature"""

    def __init__(self, target: HLILExpression, arguments: List[HLILExpression],
                 size: int = 4, return_type: Optional[IRType] = None, is_method: bool = False):
        super().__init__(OperationType.CALL, size, return_type)
        self.target = target
        self.arguments = arguments
        self.is_method = is_method
        self.operands = [target] + arguments

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.target}({args_str})"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_call(self)


class HLILBuiltinCall(HLILExpression):
    """Built-in function call"""

    def __init__(self, builtin_name: str, arguments: List[HLILExpression],
                 size: int = 4, return_type: Optional[IRType] = None):
        super().__init__(OperationType.BUILTIN_CALL, size, return_type)
        self.builtin_name = builtin_name
        self.arguments = arguments
        self.operands = arguments

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"__builtin_{self.builtin_name}({args_str})"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_builtin_call(self)


class HLILIf(HLILExpression):
    """If statement with optional else"""

    def __init__(self, condition: HLILExpression, true_body: List[HLILExpression],
                 false_body: Optional[List[HLILExpression]] = None):
        super().__init__(OperationType.IF, 0)
        self.condition = condition
        self.true_body = true_body
        self.false_body = false_body or []
        self.operands = [condition] + true_body + self.false_body

    def __str__(self) -> str:
        result = f"if ({self.condition}) {{\n"
        for stmt in self.true_body:
            result += f"  {stmt}\n"
        result += "}"
        if self.false_body:
            result += " else {\n"
            for stmt in self.false_body:
                result += f"  {stmt}\n"
            result += "}"
        return result

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_if(self)


class HLILWhile(HLILExpression):
    """While loop"""

    def __init__(self, condition: HLILExpression, body: List[HLILExpression]):
        super().__init__(OperationType.WHILE, 0)
        self.condition = condition
        self.body = body
        self.operands = [condition] + body

    def __str__(self) -> str:
        result = f"while ({self.condition}) {{\n"
        for stmt in self.body:
            result += f"  {stmt}\n"
        result += "}"
        return result

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_while(self)


class HLILFor(HLILExpression):
    """For loop"""

    def __init__(self, init: Optional[HLILExpression], condition: Optional[HLILExpression],
                 update: Optional[HLILExpression], body: List[HLILExpression]):
        super().__init__(OperationType.FOR, 0)
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body
        self.operands = []
        if init:
            self.operands.append(init)
        if condition:
            self.operands.append(condition)
        if update:
            self.operands.append(update)
        self.operands.extend(body)

    def __str__(self) -> str:
        init_str = str(self.init) if self.init else ""
        condition_str = str(self.condition) if self.condition else ""
        update_str = str(self.update) if self.update else ""

        result = f"for ({init_str}; {condition_str}; {update_str}) {{\n"
        for stmt in self.body:
            result += f"  {stmt}\n"
        result += "}"
        return result

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_for(self)


@dataclass
class SwitchCase:
    """Switch case with optional values"""
    values: List[HLILExpression]  # Empty for default case
    body: List[HLILExpression]
    is_default: bool = False


class HLILSwitch(HLILExpression):
    """Switch statement"""

    def __init__(self, expression: HLILExpression, cases: List[SwitchCase]):
        super().__init__(OperationType.SWITCH, 0)
        self.expression = expression
        self.cases = cases
        self.operands = [expression]
        for case in cases:
            self.operands.extend(case.values)
            self.operands.extend(case.body)

    def __str__(self) -> str:
        result = f"switch ({self.expression}) {{\n"
        for case in self.cases:
            if case.is_default:
                result += "  default:\n"
            else:
                for value in case.values:
                    result += f"  case {value}:\n"
            for stmt in case.body:
                result += f"    {stmt}\n"
        result += "}"
        return result

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_switch(self)


class HLILBreak(HLILExpression):
    """Break statement"""

    def __init__(self):
        super().__init__(OperationType.BREAK, 0)

    def __str__(self) -> str:
        return "break"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_break(self)


class HLILContinue(HLILExpression):
    """Continue statement"""

    def __init__(self):
        super().__init__(OperationType.CONTINUE, 0)

    def __str__(self) -> str:
        return "continue"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_continue(self)


class HLILReturn(HLILExpression):
    """Return statement"""

    def __init__(self, value: Optional[HLILExpression] = None):
        super().__init__(OperationType.RET, 0)
        self.value = value
        if value:
            self.operands = [value]

    def __str__(self) -> str:
        if self.value:
            return f"return {self.value}"
        return "return"

    def accept(self, visitor: 'HLILVisitor') -> Any:
        return visitor.visit_return(self)


class HLILVisitor(IRVisitor):
    """Visitor for HLIL expressions"""

    def visit_expression(self, expr: IRExpression) -> Any:
        if isinstance(expr, HLILExpression):
            return self.visit_hlil_expression(expr)
        raise NotImplementedError(f"Unknown expression type: {type(expr)}")

    def visit_hlil_expression(self, expr: HLILExpression) -> Any:
        """Generic HLIL expression visitor"""
        return expr.accept(self)

    def visit_variable(self, expr: HLILVariable) -> Any:
        pass

    def visit_constant(self, expr: HLILConstant) -> Any:
        pass

    def visit_binary_op(self, expr: HLILBinaryOp) -> Any:
        pass

    def visit_unary_op(self, expr: HLILUnaryOp) -> Any:
        pass

    def visit_assignment(self, expr: HLILAssignment) -> Any:
        pass

    def visit_field_access(self, expr: HLILFieldAccess) -> Any:
        pass

    def visit_array_access(self, expr: HLILArrayAccess) -> Any:
        pass

    def visit_call(self, expr: HLILCall) -> Any:
        pass

    def visit_builtin_call(self, expr: HLILBuiltinCall) -> Any:
        pass

    def visit_if(self, expr: HLILIf) -> Any:
        pass

    def visit_while(self, expr: HLILWhile) -> Any:
        pass

    def visit_for(self, expr: HLILFor) -> Any:
        pass

    def visit_switch(self, expr: HLILSwitch) -> Any:
        pass

    def visit_break(self, expr: HLILBreak) -> Any:
        pass

    def visit_continue(self, expr: HLILContinue) -> Any:
        pass

    def visit_return(self, expr: HLILReturn) -> Any:
        pass


class HLILBuilder:
    """Builder for constructing HLIL expressions"""

    def __init__(self, function: IRFunction):
        self.function = function

    # Convenience methods for building expressions
    def const(self, value: Union[int, float, str, bool], const_type: Optional[IRType] = None) -> HLILConstant:
        size = 4
        if const_type == "string":
            size = len(str(value))
        return HLILConstant(value, size, const_type)

    def var(self, variable: IRVariable, var_type: Optional[IRType] = None) -> HLILVariable:
        return HLILVariable(variable, var_type)

    def assign(self, dest: HLILVariable, source: HLILExpression) -> HLILAssignment:
        return HLILAssignment(dest, source)

    def field_access(self, base: HLILExpression, field: str, field_type: Optional[IRType] = None) -> HLILFieldAccess:
        return HLILFieldAccess(base, field, 4, field_type)

    def array_access(self, base: HLILExpression, index: HLILExpression,
                    element_type: Optional[IRType] = None) -> HLILArrayAccess:
        return HLILArrayAccess(base, index, 4, element_type)

    def add(self, left: HLILExpression, right: HLILExpression, result_type: Optional[IRType] = None) -> HLILBinaryOp:
        return HLILBinaryOp(OperationType.ADD, left, right, 4, result_type)

    def sub(self, left: HLILExpression, right: HLILExpression, result_type: Optional[IRType] = None) -> HLILBinaryOp:
        return HLILBinaryOp(OperationType.SUB, left, right, 4, result_type)

    def mul(self, left: HLILExpression, right: HLILExpression, result_type: Optional[IRType] = None) -> HLILBinaryOp:
        return HLILBinaryOp(OperationType.MUL, left, right, 4, result_type)

    def call(self, target: HLILExpression, arguments: List[HLILExpression],
             return_type: Optional[IRType] = None, is_method: bool = False) -> HLILCall:
        return HLILCall(target, arguments, 4, return_type, is_method)

    def builtin_call(self, builtin_name: str, arguments: List[HLILExpression],
                    return_type: Optional[str] = None) -> HLILBuiltinCall:
        return HLILBuiltinCall(builtin_name, arguments, 4, return_type)

    def if_stmt(self, condition: HLILExpression, true_body: List[HLILExpression],
               false_body: Optional[List[HLILExpression]] = None) -> HLILIf:
        return HLILIf(condition, true_body, false_body)

    def while_loop(self, condition: HLILExpression, body: List[HLILExpression]) -> HLILWhile:
        return HLILWhile(condition, body)

    def for_loop(self, init: Optional[HLILExpression], condition: Optional[HLILExpression],
                update: Optional[HLILExpression], body: List[HLILExpression]) -> HLILFor:
        return HLILFor(init, condition, update, body)

    def switch(self, expression: HLILExpression, cases: List[SwitchCase]) -> HLILSwitch:
        return HLILSwitch(expression, cases)

    def break_stmt(self) -> HLILBreak:
        return HLILBreak()

    def continue_stmt(self) -> HLILContinue:
        return HLILContinue()

    def ret(self, value: Optional[HLILExpression] = None) -> HLILReturn:
        return HLILReturn(value)

    def not_op(self, operand: HLILExpression) -> HLILUnaryOp:
        return HLILUnaryOp(OperationType.NOT, operand, 4, "boolean")

    def negate(self, operand: HLILExpression, result_type: Optional[str] = None) -> HLILUnaryOp:
        return HLILUnaryOp(OperationType.NEG, operand, 4, result_type)


# Convenience functions for creating HLIL
def hlil_const(value: Union[int, float, str, bool], const_type: Optional[str] = None) -> HLILConstant:
    return HLILConstant(value, 4, const_type)

def hlil_var(variable: IRVariable, var_type: Optional[str] = None) -> HLILVariable:
    return HLILVariable(variable, var_type)

def hlil_add(left: HLILExpression, right: HLILExpression) -> HLILBinaryOp:
    return HLILBinaryOp(OperationType.ADD, left, right)
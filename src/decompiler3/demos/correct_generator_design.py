#!/usr/bin/env python3
"""
正确的代码生成器设计 - 使用真实系统类型

演示如何正确地将IR与特定语言的代码生成分离
使用项目内的真实类型，而不是重新定义
"""

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

# ===== 使用真实的项目类型 =====

from decompiler3.ir.base import IRExpression, OperationType, IRVariable as BaseIRVariable, IRType
from decompiler3.ir.hlil import HLILExpression, HLILConstant, HLILVariable, HLILBinaryOp, HLILUnaryOp, HLILCall, HLILReturn, HLILIf

print("✅ 使用真实的项目类型")

# ===== 访问者模式基类 =====

class IRVisitor(ABC):
    """IR访问者基类 - 代码生成的正确方式"""

    @abstractmethod
    def visit_constant(self, expr) -> Any:
        pass

    @abstractmethod
    def visit_variable(self, expr) -> Any:
        pass

    @abstractmethod
    def visit_binary_op(self, expr) -> Any:
        pass

    @abstractmethod
    def visit_unary_op(self, expr) -> Any:
        pass

    @abstractmethod
    def visit_call(self, expr) -> Any:
        pass

    @abstractmethod
    def visit_return(self, expr) -> Any:
        pass

    @abstractmethod
    def visit_if(self, expr) -> Any:
        pass

# ===== TypeScript代码生成器 =====

class TypeScriptGenerator(IRVisitor):
    """TypeScript特定的代码生成器 - 使用真实IR类型"""

    def __init__(self, indent_size: int = 2):
        self.indent_size = indent_size
        self.indent_level = 0

    def visit_constant(self, expr) -> str:
        """生成TypeScript常量"""
        if isinstance(expr.value, str):
            return f'"{expr.value}"'
        elif isinstance(expr.value, bool):
            return "true" if expr.value else "false"
        elif expr.value is None:
            return "null"
        else:
            return str(expr.value)

    def visit_variable(self, expr) -> str:
        """生成TypeScript变量引用"""
        return expr.variable.name

    def visit_binary_op(self, expr) -> str:
        """生成TypeScript二元操作"""
        left = self._visit_expression(expr.left)
        right = self._visit_expression(expr.right)

        # TypeScript特定的操作符映射
        ts_operators = {
            OperationType.ADD: "+",
            OperationType.SUB: "-",
            OperationType.MUL: "*",
            OperationType.DIV: "/",
            OperationType.MOD: "%",
            OperationType.AND: "&",
            OperationType.OR: "|",
            OperationType.XOR: "^",
            OperationType.CMP_E: "===",  # TypeScript严格相等
            OperationType.CMP_NE: "!==",
            OperationType.CMP_SLT: "<",
        }

        operator = ts_operators.get(expr.operation, "?")
        return f"({left} {operator} {right})"

    def visit_unary_op(self, expr) -> str:
        """生成TypeScript一元操作"""
        operand = self._visit_expression(expr.operand)
        ts_operators = {
            OperationType.NOT: "!",
            OperationType.NEG: "-",
        }
        operator = ts_operators.get(expr.operation, "?")
        return f"{operator}{operand}"

    def visit_call(self, expr) -> str:
        """生成TypeScript函数调用"""
        target = self._visit_expression(expr.target)
        args = [self._visit_expression(arg) for arg in expr.arguments]
        args_str = ", ".join(args)
        return f"{target}({args_str})"

    def visit_return(self, expr) -> str:
        """生成TypeScript返回语句"""
        if expr.value:
            value = self._visit_expression(expr.value)
            return f"return {value};"
        return "return;"

    def visit_if(self, expr) -> str:
        """生成TypeScript条件语句"""
        condition = self._visit_expression(expr.condition)
        result = f"if ({condition}) {{\n"
        self.indent_level += 1
        for stmt in expr.true_body:
            stmt_code = self._visit_expression(stmt)
            result += self._indent(stmt_code) + "\n"
        self.indent_level -= 1
        result += self._indent("}")
        if expr.false_body:
            result += " else {\n"
            self.indent_level += 1
            for stmt in expr.false_body:
                stmt_code = self._visit_expression(stmt)
                result += self._indent(stmt_code) + "\n"
            self.indent_level -= 1
            result += self._indent("}")
        return result

    def _visit_expression(self, expr) -> str:
        """分发到正确的visit方法"""
        if isinstance(expr, HLILConstant):
            return self.visit_constant(expr)
        elif isinstance(expr, HLILVariable):
            return self.visit_variable(expr)
        elif isinstance(expr, HLILBinaryOp):
            return self.visit_binary_op(expr)
        elif isinstance(expr, HLILUnaryOp):
            return self.visit_unary_op(expr)
        elif isinstance(expr, HLILCall):
            return self.visit_call(expr)
        elif isinstance(expr, HLILReturn):
            return self.visit_return(expr)
        elif isinstance(expr, HLILIf):
            return self.visit_if(expr)
        else:
            return f"/* Unknown: {type(expr).__name__} */"

    def _indent(self, code: str) -> str:
        """添加缩进"""
        return " " * (self.indent_level * self.indent_size) + code

# ===== C++代码生成器 =====

class CppGenerator(IRVisitor):
    """C++特定的代码生成器 - 使用真实IR类型"""

    def visit_constant(self, expr) -> str:
        if isinstance(expr.value, str):
            return f'"{expr.value}"'
        elif isinstance(expr.value, bool):
            return "true" if expr.value else "false"
        else:
            return str(expr.value)

    def visit_variable(self, expr) -> str:
        return expr.variable.name

    def visit_binary_op(self, expr) -> str:
        left = self._visit_expression(expr.left)
        right = self._visit_expression(expr.right)
        cpp_operators = {
            OperationType.ADD: "+",
            OperationType.SUB: "-",
            OperationType.MUL: "*",
            OperationType.DIV: "/",
            OperationType.MOD: "%",
            OperationType.AND: "&",
            OperationType.OR: "|",
            OperationType.XOR: "^",
            OperationType.CMP_E: "==",  # C++相等比较
            OperationType.CMP_NE: "!=",
            OperationType.CMP_SLT: "<",
        }
        operator = cpp_operators.get(expr.operation, "?")
        return f"({left} {operator} {right})"

    def visit_unary_op(self, expr) -> str:
        operand = self._visit_expression(expr.operand)
        return f"!{operand}"

    def visit_call(self, expr) -> str:
        target = self._visit_expression(expr.target)
        args = [self._visit_expression(arg) for arg in expr.arguments]
        return f"{target}({', '.join(args)})"

    def visit_return(self, expr) -> str:
        if expr.value:
            value = self._visit_expression(expr.value)
            return f"return {value};"
        return "return;"

    def visit_if(self, expr) -> str:
        condition = self._visit_expression(expr.condition)
        return f"if ({condition}) {{ /* body */ }}"

    def _visit_expression(self, expr) -> str:
        if isinstance(expr, HLILConstant):
            return self.visit_constant(expr)
        elif isinstance(expr, HLILVariable):
            return self.visit_variable(expr)
        elif isinstance(expr, HLILBinaryOp):
            return self.visit_binary_op(expr)
        elif isinstance(expr, HLILUnaryOp):
            return self.visit_unary_op(expr)
        elif isinstance(expr, HLILCall):
            return self.visit_call(expr)
        elif isinstance(expr, HLILReturn):
            return self.visit_return(expr)
        elif isinstance(expr, HLILIf):
            return self.visit_if(expr)
        else:
            return f"/* Unknown: {type(expr).__name__} */"

# ===== Python代码生成器 =====

class PythonGenerator(IRVisitor):
    """Python特定的代码生成器 - 使用真实IR类型"""

    def visit_constant(self, expr) -> str:
        if isinstance(expr.value, str):
            return f'"{expr.value}"'
        elif isinstance(expr.value, bool):
            return "True" if expr.value else "False"  # Python大写布尔值
        elif expr.value is None:
            return "None"
        else:
            return str(expr.value)

    def visit_variable(self, expr) -> str:
        return expr.variable.name

    def visit_binary_op(self, expr) -> str:
        left = self._visit_expression(expr.left)
        right = self._visit_expression(expr.right)
        python_operators = {
            OperationType.ADD: "+",
            OperationType.SUB: "-",
            OperationType.MUL: "*",
            OperationType.DIV: "/",
            OperationType.MOD: "%",
            OperationType.AND: "&",
            OperationType.OR: "|",
            OperationType.XOR: "^",
            OperationType.CMP_E: "==",
            OperationType.CMP_NE: "!=",
            OperationType.CMP_SLT: "<",
        }
        operator = python_operators.get(expr.operation, "?")
        return f"({left} {operator} {right})"

    def visit_unary_op(self, expr) -> str:
        operand = self._visit_expression(expr.operand)
        return f"not {operand}"  # Python使用not关键字

    def visit_call(self, expr) -> str:
        target = self._visit_expression(expr.target)
        args = [self._visit_expression(arg) for arg in expr.arguments]
        return f"{target}({', '.join(args)})"

    def visit_return(self, expr) -> str:
        if expr.value:
            value = self._visit_expression(expr.value)
            return f"return {value}"
        return "return"

    def visit_if(self, expr) -> str:
        condition = self._visit_expression(expr.condition)
        return f"if {condition}:\n    pass"

    def _visit_expression(self, expr) -> str:
        if isinstance(expr, HLILConstant):
            return self.visit_constant(expr)
        elif isinstance(expr, HLILVariable):
            return self.visit_variable(expr)
        elif isinstance(expr, HLILBinaryOp):
            return self.visit_binary_op(expr)
        elif isinstance(expr, HLILUnaryOp):
            return self.visit_unary_op(expr)
        elif isinstance(expr, HLILCall):
            return self.visit_call(expr)
        elif isinstance(expr, HLILReturn):
            return self.visit_return(expr)
        elif isinstance(expr, HLILIf):
            return self.visit_if(expr)
        else:
            return f"# Unknown: {type(expr).__name__}"

# ===== 演示函数 =====

def create_sample_ir_with_real_types():
    """使用真实类型创建示例IR"""
    from decompiler3.ir.base import IRFunction, IRBasicBlock

    # 创建函数
    function = IRFunction("sample_function", 0x1000)
    block = IRBasicBlock(0x1000)
    function.basic_blocks.append(block)

    # 创建变量
    a_var = function.create_variable("a", 4, IRType.NUMBER)
    b_var = function.create_variable("b", 4, IRType.NUMBER)

    # 创建表达式 - 使用真实的HLIL类型
    var_a = HLILVariable(a_var, IRType.NUMBER)
    const_42 = HLILConstant(42, 4, IRType.NUMBER)

    # 创建二元操作: a + 42
    add_expr = HLILBinaryOp(OperationType.ADD, var_a, const_42, 4, IRType.NUMBER)

    # 创建比较: (a + 42) === b
    var_b = HLILVariable(b_var, IRType.NUMBER)
    cmp_expr = HLILBinaryOp(OperationType.CMP_E, add_expr, var_b, 4, IRType.BOOLEAN)

    # 创建返回语句
    return_expr = HLILReturn(add_expr)

    expressions = [add_expr, cmp_expr, return_expr]

    print("✅ 使用真实类型创建IR表达式:")
    for i, expr in enumerate(expressions, 1):
        print(f"   {i}. {type(expr).__name__}: {expr}")

    return expressions

def demonstrate_correct_design_with_real_types():
    """使用真实类型演示正确的代码生成器设计"""
    print("🏗️  正确的代码生成器设计 - 使用真实系统类型")
    print("=" * 60)

    # 创建示例IR
    expressions = create_sample_ir_with_real_types()
    print("\n" + "=" * 60)

    # 使用不同的代码生成器
    generators = [
        ("TypeScript", TypeScriptGenerator()),
        ("C++", CppGenerator()),
        ("Python", PythonGenerator())
    ]

    for lang_name, generator in generators:
        print(f"\n🔧 {lang_name}代码生成:")
        print(f"   (使用 {generator.__class__.__name__} + 真实IR类型)")

        for i, expr in enumerate(expressions, 1):
            # 正确的方式：使用访问者模式处理真实类型 - 不隐藏错误
            generated_code = generator._visit_expression(expr)
            print(f"  {i}. {generated_code}")

def main():
    """主演示"""
    demonstrate_correct_design_with_real_types()

    print("\n" + "=" * 60)
    print("✅ 关键设计原则（使用真实类型）:")
    print("   • 使用项目内的真实IR类型（HLILConstant, HLILBinaryOp等）")
    print("   • IR的__str__()仅用于调试输出，保持语言中性")
    print("   • 使用访问者模式实现特定语言的代码生成")
    print("   • 每种语言有独立的生成器类")
    print("   • 语言特性差异在生成器中处理")
    print("   • 避免重新定义已有的系统类型")

if __name__ == "__main__":
    main()
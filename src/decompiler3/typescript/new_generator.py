"""
TypeScript Generator for New IR System

Generates TypeScript code from HLIL (High Level Intermediate Language)
"""

from typing import List, Optional, Dict, Set
from ..ir.hlil_new import (
    HighLevelILFunction, HighLevelILInstruction, HighLevelILBasicBlock,
    HighLevelILAdd, HighLevelILSub, HighLevelILMul, HighLevelILDiv,
    HighLevelILVar, HighLevelILAssign, HighLevelILVarInit,
    HighLevelILIf, HighLevelILWhile, HighLevelILFor, HighLevelILDoWhile,
    HighLevelILSwitch, HighLevelILJump, HighLevelILGoto, HighLevelILLabel,
    HighLevelILCall, HighLevelILTailcall, HighLevelILRet,
    HighLevelILBlock, HighLevelILConst
)
from ..ir.mlil_new import Variable


class NewTypeScriptGenerator:
    """Generates TypeScript code from HLIL"""

    def __init__(self, style: str = "pretty"):
        self.style = style
        self.indent_level = 0
        self.indent_size = 2

    def generate_function(self, function: HighLevelILFunction) -> str:
        """Generate TypeScript function from HLIL"""
        lines = []

        # Function signature
        params = self._generate_parameters(function)
        return_type = self._infer_return_type(function)
        lines.append(f"function {function.name}({params}): {return_type} {{")

        # Variable declarations
        var_decls = self._generate_variable_declarations(function)
        if var_decls:
            lines.extend(self._indent_lines(var_decls))
            lines.append("")

        # Function body
        body_lines = self._generate_function_body(function)
        lines.extend(self._indent_lines(body_lines))

        lines.append("}")
        return "\n".join(lines)

    def _generate_parameters(self, function: HighLevelILFunction) -> str:
        """Generate function parameters"""
        # For now, just check if we have any parameter-like variables
        params = []
        for name, var in function.variables.items():
            if name.startswith("param_") or name.startswith("arg_"):
                ts_type = self._map_type_to_typescript(var.var_type)
                params.append(f"{name}: {ts_type}")

        return ", ".join(params)

    def _generate_variable_declarations(self, function: HighLevelILFunction) -> List[str]:
        """Generate variable declarations"""
        lines = []
        declared_vars = set()

        for name, var in function.variables.items():
            if not name.startswith("param_") and not name.startswith("arg_"):
                ts_type = self._map_type_to_typescript(var.var_type)
                lines.append(f"let {name}: {ts_type};")
                declared_vars.add(name)

        return lines

    def _generate_function_body(self, function: HighLevelILFunction) -> List[str]:
        """Generate function body"""
        lines = []

        if self._has_structured_control_flow(function):
            # Generate structured code
            lines.extend(self._generate_structured_body(function))
        else:
            # Generate basic block based code with labels
            lines.extend(self._generate_basic_block_body(function))

        return lines

    def _generate_structured_body(self, function: HighLevelILFunction) -> List[str]:
        """Generate structured control flow"""
        lines = []

        for block in function.basic_blocks:
            for instruction in block.instructions:
                instr_lines = self._generate_instruction(instruction)
                lines.extend(instr_lines)

        return lines

    def _generate_basic_block_body(self, function: HighLevelILFunction) -> List[str]:
        """Generate basic block based code with labels"""
        lines = []

        for block in function.basic_blocks:
            # Block label
            if len(function.basic_blocks) > 1:
                lines.append(f"label_{block.id[:8]}:")

            # Block instructions
            for instruction in block.instructions:
                instr_lines = self._generate_instruction(instruction)
                lines.extend(instr_lines)

        return lines

    def _generate_instruction(self, instruction: HighLevelILInstruction) -> List[str]:
        """Generate TypeScript for a single instruction"""
        if isinstance(instruction, HighLevelILAssign):
            return [f"{self._generate_expression(instruction.dest)} = {self._generate_expression(instruction.src)};"]

        elif isinstance(instruction, HighLevelILVarInit):
            ts_type = self._map_type_to_typescript(instruction.dest.var_type)
            return [f"let {instruction.dest.name}: {ts_type} = {self._generate_expression(instruction.src)};"]

        elif isinstance(instruction, HighLevelILIf):
            return self._generate_if_statement(instruction)

        elif isinstance(instruction, HighLevelILWhile):
            return self._generate_while_loop(instruction)

        elif isinstance(instruction, HighLevelILFor):
            return self._generate_for_loop(instruction)

        elif isinstance(instruction, HighLevelILDoWhile):
            return self._generate_do_while_loop(instruction)

        elif isinstance(instruction, HighLevelILSwitch):
            return self._generate_switch_statement(instruction)

        elif isinstance(instruction, HighLevelILCall):
            call_expr = self._generate_expression(instruction)
            return [f"{call_expr};"]

        elif isinstance(instruction, HighLevelILRet):
            if instruction.src:
                if len(instruction.src) == 1:
                    return [f"return {self._generate_expression(instruction.src[0])};"]
                else:
                    exprs = [self._generate_expression(src) for src in instruction.src]
                    return [f"return [{', '.join(exprs)}];"]
            return ["return;"]

        elif isinstance(instruction, HighLevelILLabel):
            return [f"{instruction.target.name}:"]

        elif isinstance(instruction, HighLevelILGoto):
            return [f"goto {instruction.target.name};"]

        elif isinstance(instruction, HighLevelILBlock):
            lines = ["{"]
            for stmt in instruction.body:
                stmt_lines = self._generate_instruction(stmt)
                lines.extend(self._indent_lines(stmt_lines))
            lines.append("}")
            return lines

        else:
            # Fallback for expressions used as statements
            expr = self._generate_expression(instruction)
            return [f"{expr};"]

    def _generate_expression(self, expression: HighLevelILInstruction) -> str:
        """Generate TypeScript expression"""
        if isinstance(expression, HighLevelILConst):
            return str(expression.constant)

        elif isinstance(expression, HighLevelILVar):
            return expression.src.name

        elif isinstance(expression, HighLevelILAdd):
            left = self._generate_expression(expression.left)
            right = self._generate_expression(expression.right)
            return f"({left} + {right})"

        elif isinstance(expression, HighLevelILSub):
            left = self._generate_expression(expression.left)
            right = self._generate_expression(expression.right)
            return f"({left} - {right})"

        elif isinstance(expression, HighLevelILMul):
            left = self._generate_expression(expression.left)
            right = self._generate_expression(expression.right)
            return f"({left} * {right})"

        elif isinstance(expression, HighLevelILDiv):
            left = self._generate_expression(expression.left)
            right = self._generate_expression(expression.right)
            return f"({left} / {right})"

        elif isinstance(expression, HighLevelILCall):
            func_name = self._generate_expression(expression.dest)
            if expression.params:
                args = [self._generate_expression(param) for param in expression.params]
                return f"{func_name}({', '.join(args)})"
            return f"{func_name}()"

        else:
            return f"/* Unknown expression: {type(expression).__name__} */"

    def _generate_if_statement(self, if_stmt: HighLevelILIf) -> List[str]:
        """Generate if statement"""
        lines = []
        condition = self._generate_expression(if_stmt.condition)
        lines.append(f"if ({condition}) {{")

        true_lines = self._generate_instruction(if_stmt.true)
        lines.extend(self._indent_lines(true_lines))

        if if_stmt.false:
            lines.append("} else {")
            false_lines = self._generate_instruction(if_stmt.false)
            lines.extend(self._indent_lines(false_lines))

        lines.append("}")
        return lines

    def _generate_while_loop(self, while_loop: HighLevelILWhile) -> List[str]:
        """Generate while loop"""
        lines = []
        condition = self._generate_expression(while_loop.condition)
        lines.append(f"while ({condition}) {{")

        body_lines = self._generate_instruction(while_loop.body)
        lines.extend(self._indent_lines(body_lines))

        lines.append("}")
        return lines

    def _generate_for_loop(self, for_loop: HighLevelILFor) -> List[str]:
        """Generate for loop"""
        lines = []
        init = self._generate_expression(for_loop.init)
        condition = self._generate_expression(for_loop.condition)
        update = self._generate_expression(for_loop.update)

        lines.append(f"for ({init}; {condition}; {update}) {{")

        body_lines = self._generate_instruction(for_loop.body)
        lines.extend(self._indent_lines(body_lines))

        lines.append("}")
        return lines

    def _generate_do_while_loop(self, do_while: HighLevelILDoWhile) -> List[str]:
        """Generate do-while loop"""
        lines = ["do {"]

        body_lines = self._generate_instruction(do_while.body)
        lines.extend(self._indent_lines(body_lines))

        condition = self._generate_expression(do_while.condition)
        lines.append(f"}} while ({condition});")
        return lines

    def _generate_switch_statement(self, switch_stmt: HighLevelILSwitch) -> List[str]:
        """Generate switch statement"""
        lines = []
        condition = self._generate_expression(switch_stmt.condition)
        lines.append(f"switch ({condition}) {{")

        for values, body in switch_stmt.cases:
            for value in values:
                lines.append(f"  case {value}:")
            body_lines = self._generate_instruction(body)
            lines.extend(self._indent_lines(body_lines, 2))
            lines.append("    break;")

        if switch_stmt.default:
            lines.append("  default:")
            default_lines = self._generate_instruction(switch_stmt.default)
            lines.extend(self._indent_lines(default_lines, 2))

        lines.append("}")
        return lines

    def _map_type_to_typescript(self, ir_type: Optional[str]) -> str:
        """Map IR type to TypeScript type"""
        if not ir_type:
            return "any"

        type_mapping = {
            "int": "number",
            "int8": "number",
            "int16": "number",
            "int32": "number",
            "int64": "number",
            "float": "number",
            "float32": "number",
            "float64": "number",
            "double": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "char": "string",
            "string": "string",
            "void": "void",
            "ptr": "any",
            "pointer": "any"
        }

        return type_mapping.get(ir_type.lower(), "any")

    def _infer_return_type(self, function: HighLevelILFunction) -> str:
        """Infer function return type"""
        # Look for return statements to infer type
        for block in function.basic_blocks:
            for instruction in block.instructions:
                if isinstance(instruction, HighLevelILRet):
                    if instruction.src:
                        if len(instruction.src) == 1:
                            # Single return value - could infer from variable type
                            return "any"  # Simplified
                        else:
                            # Multiple return values
                            return "any[]"
                    else:
                        return "void"

        return "void"

    def _has_structured_control_flow(self, function: HighLevelILFunction) -> bool:
        """Check if function has structured control flow"""
        for block in function.basic_blocks:
            for instruction in block.instructions:
                if isinstance(instruction, (HighLevelILIf, HighLevelILWhile, HighLevelILFor, HighLevelILSwitch)):
                    return True
        return False

    def _indent_lines(self, lines: List[str], extra_indent: int = 0) -> List[str]:
        """Add indentation to lines"""
        indent = " " * (self.indent_size + extra_indent)
        return [f"{indent}{line}" if line.strip() else line for line in lines]
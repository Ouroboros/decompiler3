'''
TypeScript Code Generator

Generates production-quality TypeScript code from HLIL.
'''

from common import *
from typing import List, Optional
from ir.hlil import *


class TypeScriptGenerator:
    '''Generate TypeScript code from HLIL'''

    @classmethod
    def _format_type(cls, type_hint: Optional[str]) -> str:
        '''Map HLIL type to TypeScript type

        Args:
            type_hint: Type hint string from HLIL

        Returns:
            TypeScript type string
        '''
        if not type_hint:
            return 'any'

        # Map common types
        type_map = {
            'int': 'number',
            'int32': 'number',
            'int64': 'number',
            'float': 'number',
            'float32': 'number',
            'float64': 'number',
            'bool': 'boolean',
            'str': 'string',
            'string': 'string',
            'void': 'void',
        }

        return type_map.get(type_hint.lower(), type_hint)

    @classmethod
    def _infer_return_type(cls, block: HLILBlock) -> str:
        '''Infer function return type from return statements

        Args:
            block: Function body block

        Returns:
            TypeScript return type
        '''
        # Recursively search for return statements
        def find_returns(blk: HLILBlock) -> List[HLILReturn]:
            returns = []
            for stmt in blk.statements:
                if isinstance(stmt, HLILReturn):
                    returns.append(stmt)

                elif isinstance(stmt, HLILIf):
                    returns.extend(find_returns(stmt.true_block))
                    if stmt.false_block:
                        returns.extend(find_returns(stmt.false_block))

                elif isinstance(stmt, HLILWhile):
                    returns.extend(find_returns(stmt.body))

                elif isinstance(stmt, HLILSwitch):
                    for case in stmt.cases:
                        returns.extend(find_returns(case.body))

            return returns

        returns = find_returns(block)

        # If no return statements or all returns are void
        if not returns or all(r.value is None for r in returns):
            return 'void'

        # Check if any return has a value
        for ret in returns:
            if ret.value is not None:
                # Try to infer type from the value
                if isinstance(ret.value, HLILConst):
                    val = ret.value.value
                    if isinstance(val, bool):
                        return 'boolean'
                    elif isinstance(val, int):
                        return 'number'
                    elif isinstance(val, float):
                        return 'number'
                    elif isinstance(val, str):
                        return 'string'

                # Default to number for most return values
                return 'number'

        return 'void'

    @classmethod
    def _needs_parentheses(cls, child_op: str, parent_op: str, is_left: bool) -> bool:
        '''Check if child expression needs parentheses based on operator precedence

        Args:
            child_op: Operator of child expression
            parent_op: Operator of parent expression
            is_left: True if child is left operand, False if right

        Returns:
            True if parentheses are needed
        '''
        # Operator precedence (lower number = lower precedence)
        precedence = {
            '||': 1,
            '&&': 2,
            '|': 3,
            '^': 4,
            '&': 5,
            '==': 6, '!=': 6,
            '<': 7, '<=': 7, '>': 7, '>=': 7,
            '<<': 8, '>>': 8,
            '+': 9, '-': 9,
            '*': 10, '/': 10, '%': 10,
        }

        child_prec = precedence.get(child_op, 100)
        parent_prec = precedence.get(parent_op, 100)

        # Need parentheses if child has lower precedence
        if child_prec < parent_prec:
            return True

        # For same precedence, only right operand needs parentheses for non-associative ops
        if child_prec == parent_prec and not is_left:
            # Non-associative: -, /, %
            if parent_op in {'-', '/', '%'}:
                return True

        return False

    @classmethod
    def _format_expr(cls, expr: HLILExpression) -> str:
        '''Format an HLIL expression as TypeScript

        Args:
            expr: Expression to format

        Returns:
            TypeScript expression string
        '''
        if isinstance(expr, HLILVar):
            return expr.var.name

        elif isinstance(expr, HLILConst):
            # Handle different constant types
            if isinstance(expr.value, str):
                return f'"{expr.value}"'

            elif isinstance(expr.value, bool):
                return 'true' if expr.value else 'false'

            elif isinstance(expr.value, float):
                return format_float(expr.value)

            else:
                return str(expr.value)

        elif isinstance(expr, HLILBinaryOp):
            lhs_str = cls._format_expr(expr.lhs)
            rhs_str = cls._format_expr(expr.rhs)

            # Add parentheses if needed based on precedence
            if isinstance(expr.lhs, HLILBinaryOp):
                if cls._needs_parentheses(expr.lhs.op, expr.op, True):
                    lhs_str = f'({lhs_str})'

            if isinstance(expr.rhs, HLILBinaryOp):
                if cls._needs_parentheses(expr.rhs.op, expr.op, False):
                    rhs_str = f'({rhs_str})'

            return f'{lhs_str} {expr.op} {rhs_str}'

        elif isinstance(expr, HLILUnaryOp):
            operand = cls._format_expr(expr.operand)
            return f'{expr.op}{operand}'

        elif isinstance(expr, HLILCall):
            args = ', '.join(cls._format_expr(arg) for arg in expr.args)
            return f'{expr.func_name}({args})'

        elif isinstance(expr, HLILSyscall):
            args = [
                f'{expr.subsystem}',
                f'{expr.cmd}',
                *[cls._format_expr(arg) for arg in expr.args],
            ]
            return f'syscall({', '.join(args)})'

        elif isinstance(expr, HLILExternCall):
            args = [
                f"'{expr.target}'",
                *[cls._format_expr(arg) for arg in expr.args],
            ]
            return f'extern_call({', '.join(args)})'

        else:
            # Fallback
            return str(expr)

    @classmethod
    def generate_function(cls, func: HighLevelILFunction) -> List[str]:
        '''Generate TypeScript code for an HLIL function

        Args:
            func: HLIL function to generate code for

        Returns:
            List of TypeScript code lines
        '''
        lines = []

        # Function signature with types
        if func.parameters:
            params = ', '.join(
                f'{p.name}: {cls._format_type(p.type_hint)} = {p.default_value}' if p.default_value is not None else f'{p.name}: {cls._format_type(p.type_hint)}'
                for p in func.parameters
            )
        else:
            params = ''

        return_type = cls._infer_return_type(func.body)
        lines.append(f'function {func.name}({params}): {return_type} {{')

        # Variable declarations with types
        if func.variables:
            for var in func.variables:
                var_type = cls._format_type(var.type_hint)
                lines.append(f'{default_indent()}let {var.name}: {var_type};')
            lines.append('')

        # Function body
        body_lines = cls._generate_block(func.body, indent=1)
        lines.extend(body_lines)

        lines.append('}')
        return lines

    @classmethod
    def _generate_block(cls, block: HLILBlock, indent: int = 0) -> List[str]:
        '''Generate TypeScript code for a block of statements

        Args:
            block: Block to generate code for
            indent: Indentation level

        Returns:
            List of TypeScript code lines
        '''
        lines = []

        for stmt in block.statements:
            stmt_lines = cls._generate_statement(stmt, indent)
            lines.extend(stmt_lines)

        return lines

    @classmethod
    def _generate_statement(cls, stmt: HLILStatement, indent: int = 0) -> List[str]:
        '''Generate TypeScript code for a statement

        Args:
            stmt: Statement to generate code for
            indent: Indentation level

        Returns:
            List of TypeScript code lines
        '''
        indent_str = default_indent() * indent
        lines = []

        if isinstance(stmt, HLILIf):
            # if (condition) { ... }
            cond_str = cls._format_expr(stmt.condition)
            lines.append(f'{indent_str}if ({cond_str}) {{')
            lines.extend(cls._generate_block(stmt.true_block, indent + 1))

            if stmt.false_block and stmt.false_block.statements:
                lines.append(f'{indent_str}}} else {{')
                lines.extend(cls._generate_block(stmt.false_block, indent + 1))

            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILWhile):
            # while (condition) { ... }
            cond_str = cls._format_expr(stmt.condition)
            lines.append(f'{indent_str}while ({cond_str}) {{')
            lines.extend(cls._generate_block(stmt.body, indent + 1))
            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILDoWhile):
            # do { ... } while (condition);
            cond_str = cls._format_expr(stmt.condition)
            lines.append(f'{indent_str}do {{')
            lines.extend(cls._generate_block(stmt.body, indent + 1))
            lines.append(f'{indent_str}}} while ({cond_str});')

        elif isinstance(stmt, HLILFor):
            # for (init; cond; update) { ... }
            init_str = cls._format_expr(stmt.init) if stmt.init else ''
            cond_str = cls._format_expr(stmt.condition) if stmt.condition else ''
            update_str = cls._format_expr(stmt.update) if stmt.update else ''
            lines.append(f'{indent_str}for ({init_str}; {cond_str}; {update_str}) {{')
            lines.extend(cls._generate_block(stmt.body, indent + 1))
            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILSwitch):
            # switch (scrutinee) { ... }
            scrutinee_str = cls._format_expr(stmt.scrutinee)
            lines.append(f'{indent_str}switch ({scrutinee_str}) {{')
            case_indent = default_indent()
            case_body_indent = default_indent() * 2
            for case in stmt.cases:
                if case.is_default():
                    lines.append(f'{indent_str}{case_indent}default:')
                else:
                    case_val_str = cls._format_expr(case.value)
                    lines.append(f'{indent_str}{case_indent}case {case_val_str}:')
                lines.extend(cls._generate_block(case.body, indent + 2))

                # Add break if case doesn't end with return/break/continue
                if case.body.statements:
                    last_stmt = case.body.statements[-1]
                    if not isinstance(last_stmt, (HLILReturn, HLILBreak, HLILContinue)):
                        lines.append(f'{indent_str}{case_body_indent}break;')

            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILBreak):
            lines.append(f'{indent_str}break;')

        elif isinstance(stmt, HLILContinue):
            lines.append(f'{indent_str}continue;')

        elif isinstance(stmt, HLILReturn):
            if stmt.value is not None:
                value_str = cls._format_expr(stmt.value)
                lines.append(f'{indent_str}return {value_str};')
            else:
                lines.append(f'{indent_str}return;')

        elif isinstance(stmt, HLILAssign):
            dest_str = cls._format_expr(stmt.dest)
            src_str = cls._format_expr(stmt.src)
            lines.append(f'{indent_str}{dest_str} = {src_str};')

        elif isinstance(stmt, HLILExprStmt):
            expr_str = cls._format_expr(stmt.expr)
            lines.append(f'{indent_str}{expr_str};')

        elif isinstance(stmt, HLILComment):
            lines.append(f'{indent_str}// {stmt.text}')

        elif isinstance(stmt, HLILBlock):
            # Nested block
            lines.append(f'{indent_str}{{')
            lines.extend(cls._generate_block(stmt, indent + 1))
            lines.append(f'{indent_str}}}')

        else:
            lines.append(f'{indent_str}// {stmt}')

        return lines


def generate_typescript_header() -> str:
    '''Generate TypeScript type definitions header

    Returns:
        TypeScript type definitions
    '''
    return '''// Intrinsic function: address-of operator (for output parameters)
// Can be recognized and transformed during compilation
function addr_of<T>(value: T): T { return value; }

// Placeholder function: external script call
function extern_call(target: string, ...args: any[]): any { return undefined; }

// Placeholder function: system call
function syscall(subsystem: number, cmd: number, ...args: any[]): any { return undefined; }

'''


def generate_typescript(func: HighLevelILFunction) -> str:
    '''Generate TypeScript code from HLIL function

    Args:
        func: HLIL function

    Returns:
        Generated TypeScript code
    '''
    lines = TypeScriptGenerator.generate_function(func)
    return '\n'.join(lines)

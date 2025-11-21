'''
HLIL Formatter - Pretty print HLIL code
'''

from common import *
from typing import List
from .hlil import *


class HLILFormatter:
    '''Format HLIL functions as readable code'''

    @classmethod
    def _format_expr(cls, expr: HLILExpression) -> str:
        '''Format an HLIL expression

        Args:
            expr: Expression to format

        Returns:
            Formatted expression string
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
            lhs = cls._format_expr(expr.lhs)
            rhs = cls._format_expr(expr.rhs)
            return f'{lhs} {expr.op} {rhs}'

        elif isinstance(expr, HLILUnaryOp):
            operand = cls._format_expr(expr.operand)
            return f'{expr.op}{operand}'

        elif isinstance(expr, HLILCall):
            args = ', '.join(cls._format_expr(arg) for arg in expr.args)
            return f'{expr.func_name}({args})'

        elif isinstance(expr, HLILSyscall):
            args = ', '.join(cls._format_expr(arg) for arg in expr.args)
            args = [
                f'{expr.subsystem}',
                f'{expr.cmd}',
                *[cls._format_expr(arg) for arg in expr.args],
            ]
            return f'syscall({', '.join(args)})'

        else:
            # Fallback
            return str(expr)

    @classmethod
    def format_function(cls, func: HighLevelILFunction) -> List[str]:
        '''Format an HLIL function

        Args:
            func: HLIL function to format

        Returns:
            List of formatted lines
        '''
        lines = []

        # Function header
        params = ', '.join(p.name for p in func.parameters) if func.parameters else ''
        lines.append(f'function {func.name}({params}) {{')

        # Variables
        if func.variables:
            for var in func.variables:
                lines.append(f'{default_indent()}let {var.name};')
            lines.append('')

        # Function body
        body_lines = cls._format_block(func.body, indent=1)
        lines.extend(body_lines)

        lines.append('}')
        return lines

    @classmethod
    def _format_block(cls, block: HLILBlock, indent: int = 0) -> List[str]:
        '''Format a block of statements

        Args:
            block: Block to format
            indent: Indentation level

        Returns:
            List of formatted lines
        '''
        lines = []

        for stmt in block.statements:
            stmt_lines = cls._format_statement(stmt, indent)
            lines.extend(stmt_lines)

        return lines

    @classmethod
    def _format_statement(cls, stmt: HLILStatement, indent: int = 0) -> List[str]:
        '''Format a statement

        Args:
            stmt: Statement to format
            indent: Indentation level

        Returns:
            List of formatted lines
        '''
        indent_str = default_indent() * indent
        lines = []

        if isinstance(stmt, HLILIf):
            # if (condition) { ... }
            cond_str = cls._format_expr(stmt.condition)
            lines.append(f'{indent_str}if ({cond_str}) {{')
            lines.extend(cls._format_block(stmt.true_block, indent + 1))

            if stmt.false_block and stmt.false_block.statements:
                lines.append(f'{indent_str}}} else {{')
                lines.extend(cls._format_block(stmt.false_block, indent + 1))

            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILWhile):
            # while (condition) { ... }
            cond_str = cls._format_expr(stmt.condition)
            lines.append(f'{indent_str}while ({cond_str}) {{')
            lines.extend(cls._format_block(stmt.body, indent + 1))
            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILDoWhile):
            # do { ... } while (condition);
            cond_str = cls._format_expr(stmt.condition)
            lines.append(f'{indent_str}do {{')
            lines.extend(cls._format_block(stmt.body, indent + 1))
            lines.append(f'{indent_str}}} while ({cond_str});')

        elif isinstance(stmt, HLILFor):
            # for (init; cond; update) { ... }
            init_str = cls._format_expr(stmt.init) if stmt.init else ''
            cond_str = cls._format_expr(stmt.condition) if stmt.condition else ''
            update_str = cls._format_expr(stmt.update) if stmt.update else ''
            lines.append(f'{indent_str}for ({init_str}; {cond_str}; {update_str}) {{')
            lines.extend(cls._format_block(stmt.body, indent + 1))
            lines.append(f'{indent_str}}}')

        elif isinstance(stmt, HLILSwitch):
            # switch (scrutinee) { ... }
            scrutinee_str = cls._format_expr(stmt.scrutinee)
            lines.append(f'{indent_str}switch ({scrutinee_str}) {{')
            for case in stmt.cases:
                if case.is_default():
                    lines.append(f'{indent_str}  default:')
                else:
                    case_val_str = cls._format_expr(case.value)
                    lines.append(f'{indent_str}  case {case_val_str}:')
                lines.extend(cls._format_block(case.body, indent + 2))

                # Add break if case doesn't end with return/break/continue
                if case.body.statements:
                    last_stmt = case.body.statements[-1]
                    if not isinstance(last_stmt, (HLILReturn, HLILBreak, HLILContinue)):
                        lines.append(f'{indent_str}    break;')

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
            lines.extend(cls._format_block(stmt, indent + 1))
            lines.append(f'{indent_str}}}')

        else:
            lines.append(f'{indent_str}// {stmt}')

        return lines

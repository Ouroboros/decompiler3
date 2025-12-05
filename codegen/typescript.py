'''TypeScript Code Generator - HLIL to production TypeScript'''

from common import *
from typing import List, Optional
from ir.hlil import *


class TypeScriptGenerator:
    @classmethod
    def _format_type(cls, type_hint) -> str:
        '''Map HLIL type to TypeScript'''
        if not type_hint:
            return 'any'

        # Handle HLILTypeKind enum
        if isinstance(type_hint, HLILTypeKind):
            type_map = {
                HLILTypeKind.INT: 'number',
                HLILTypeKind.FLOAT: 'number',
                HLILTypeKind.STRING: 'string',
                HLILTypeKind.BOOL: 'boolean',
                HLILTypeKind.VOID: 'void',
            }
            return type_map.get(type_hint, 'any')

        # Handle string type hints (from FalcomTypeInferencePass)
        if isinstance(type_hint, str):
            type_map = {
                'int': 'number',
                'float': 'number',
                'number': 'number',
                'bool': 'boolean',
                'string': 'string',
                'void': 'void',
            }
            return type_map.get(type_hint.lower(), type_hint)

        return 'any'

    @classmethod
    def _infer_return_type(cls, block: HLILBlock) -> str:
        '''Infer return type from return statements'''
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

    BOOLEAN_BINARY_OPS = {
        BinaryOp.EQ, BinaryOp.NE,
        BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE,
        BinaryOp.AND, BinaryOp.OR,
    }

    @classmethod
    def _is_boolean_expr(cls, expr: HLILExpression) -> bool:
        if isinstance(expr, HLILBinaryOp):
            return expr.op in cls.BOOLEAN_BINARY_OPS

        elif isinstance(expr, HLILUnaryOp):
            return expr.op == UnaryOp.NOT

        return False

    @classmethod
    def _format_expr(cls, expr: HLILExpression) -> str:
        if isinstance(expr, HLILVar):
            var = expr.var
            if var.kind == VariableKind.GLOBAL:
                return f'GLOBALS[{var.index}]'

            elif var.kind == VariableKind.REG:
                return f'REGS[{var.index}]'

            else:
                return var.name

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
            # Simplify boolean comparisons with 0
            # (bool_expr) != 0 -> bool_expr
            # (bool_expr) == 0 -> !bool_expr
            if isinstance(expr.rhs, HLILConst) and expr.rhs.value == 0:
                if cls._is_boolean_expr(expr.lhs):
                    if expr.op == BinaryOp.NE:
                        # (bool) != 0 -> bool
                        return cls._format_expr(expr.lhs)

                    elif expr.op == BinaryOp.EQ:
                        # (bool) == 0 -> !bool
                        inner = cls._format_expr(expr.lhs)
                        # Add parentheses if the inner expression has lower precedence than !
                        if isinstance(expr.lhs, HLILBinaryOp) and expr.lhs.op in (BinaryOp.OR, BinaryOp.AND):
                            inner = f'({inner})'
                        return f'!{inner}'

            lhs_str = cls._format_expr(expr.lhs)
            rhs_str = cls._format_expr(expr.rhs)
            op_str = BINARY_OP_STR[expr.op]

            # Add parentheses if needed based on precedence
            if isinstance(expr.lhs, HLILBinaryOp):
                if cls._needs_parentheses(BINARY_OP_STR[expr.lhs.op], op_str, True):
                    lhs_str = f'({lhs_str})'

            if isinstance(expr.rhs, HLILBinaryOp):
                if cls._needs_parentheses(BINARY_OP_STR[expr.rhs.op], op_str, False):
                    rhs_str = f'({rhs_str})'

            return f'{lhs_str} {op_str} {rhs_str}'

        elif isinstance(expr, HLILUnaryOp):
            operand = cls._format_expr(expr.operand)
            return f'{UNARY_OP_STR[expr.op]}{operand}'

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
        lines = []

        for stmt in block.statements:
            stmt_lines = cls._generate_statement(stmt, indent)
            lines.extend(stmt_lines)

        return lines

    @classmethod
    def _get_if_depth(cls, block: HLILBlock) -> int:
        '''Get the maximum if nesting depth in a block (iterative)'''
        if not block or not block.statements:
            return 0

        max_depth = 0
        stack = [(block, 0)]  # (block, current_depth)

        while stack:
            blk, depth = stack.pop()
            if not blk or not blk.statements:
                continue

            for stmt in blk.statements:
                if isinstance(stmt, HLILIf):
                    new_depth = depth + 1
                    max_depth = max(max_depth, new_depth)
                    stack.append((stmt.true_block, new_depth))
                    stack.append((stmt.false_block, new_depth))

        return max_depth

    NEGATION_MAP = {
        BinaryOp.EQ: BinaryOp.NE,
        BinaryOp.NE: BinaryOp.EQ,
        BinaryOp.LT: BinaryOp.GE,
        BinaryOp.GE: BinaryOp.LT,
        BinaryOp.GT: BinaryOp.LE,
        BinaryOp.LE: BinaryOp.GT,
    }

    @classmethod
    def _negate_condition_str(cls, cond: HLILExpression) -> str:
        '''Format negated condition as string'''
        if isinstance(cond, HLILBinaryOp):
            if cond.op in cls.NEGATION_MAP:
                lhs = cls._format_expr(cond.lhs)
                rhs = cls._format_expr(cond.rhs)
                negated_op = BINARY_OP_STR[cls.NEGATION_MAP[cond.op]]
                return f'{lhs} {negated_op} {rhs}'

        # Fallback: wrap original in !()
        return f'!({cls._format_expr(cond)})'

    @classmethod
    def _generate_statement(cls, stmt: HLILStatement, indent: int = 0) -> List[str]:
        indent_str = default_indent() * indent
        lines = []

        if isinstance(stmt, HLILIf):
            condition = stmt.condition
            true_block = stmt.true_block
            false_block = stmt.false_block

            # Swap if true has deeper if nesting than false (reduce nesting)
            true_depth = cls._get_if_depth(true_block)
            false_depth = cls._get_if_depth(false_block)
            if false_block and true_depth > false_depth:
                cond_str = cls._negate_condition_str(condition)
                true_block, false_block = false_block, true_block

            else:
                cond_str = cls._format_expr(condition)

            lines.append(f'{indent_str}if ({cond_str}) {{')
            lines.extend(cls._generate_block(true_block, indent + 1))

            while false_block and false_block.statements:
                # else { single if } -> else if
                if len(false_block.statements) == 1 and isinstance(false_block.statements[0], HLILIf):
                    inner_if = false_block.statements[0]
                    inner_cond = cls._format_expr(inner_if.condition)
                    lines.append(f'{indent_str}}} else if ({inner_cond}) {{')
                    lines.extend(cls._generate_block(inner_if.true_block, indent + 1))
                    false_block = inner_if.false_block

                else:
                    lines.append(f'{indent_str}}} else {{')
                    lines.extend(cls._generate_block(false_block, indent + 1))
                    break

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
                    lines.append(f'{indent_str}{case_indent}default: {{')
                else:
                    case_val_str = cls._format_expr(case.value)
                    lines.append(f'{indent_str}{case_indent}case {case_val_str}: {{')
                lines.extend(cls._generate_block(case.body, indent + 2))

                # Add break if case doesn't end with return/break/continue
                if case.body.statements:
                    last_stmt = case.body.statements[-1]
                    if not isinstance(last_stmt, (HLILReturn, HLILBreak, HLILContinue)):
                        lines.append(f'{indent_str}{case_body_indent}break;')

                lines.append(f'{indent_str}{case_indent}}}')

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
    return '''// VM state
const GLOBALS: any[] = [];
const REGS: any[] = new Array(16);

// Intrinsic function: address-of operator (for output parameters)
function addr_of<T>(value: T): T { return value; }

// Placeholder function: external script call
function extern_call(target: string, ...args: any[]): any { return undefined; }

// Placeholder function: system call
function syscall(subsystem: number, cmd: number, ...args: any[]): any { return undefined; }

'''


def generate_typescript(func: HighLevelILFunction) -> str:
    lines = TypeScriptGenerator.generate_function(func)
    return '\n'.join(lines)

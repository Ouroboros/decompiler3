'''
HLIL Optimizer

Optimizes HLIL code by recognizing patterns and simplifying expressions.
'''

from typing import List, Optional
from .hlil import (
    HighLevelILFunction, HLILBlock, HLILStatement, HLILExpression,
    HLILVariable, HLILVar, HLILConst, HLILAssign, HLILExprStmt,
    HLILReturn, HLILCall, HLILIf, HLILWhile, HLILBinaryOp, HLILUnaryOp,
    HLILSwitch, HLILSwitchCase
)


class HLILOptimizer:
    '''Optimize HLIL functions'''

    @classmethod
    def optimize(cls, func: HighLevelILFunction) -> HighLevelILFunction:
        '''Apply all optimizations to a function

        Args:
            func: Function to optimize

        Returns:
            Optimized function (modifies in place and returns same object)
        '''
        cls._optimize_block(func.body)
        return func

    @classmethod
    def _optimize_block(cls, block: HLILBlock):
        '''Optimize statements in a block'''
        if not block.statements:
            return

        optimized = []
        i = 0

        while i < len(block.statements):
            stmt = block.statements[i]

            # Pattern 1: Call followed by REG[0] read -> merge into assignment
            if isinstance(stmt, HLILExprStmt) and isinstance(stmt.expr, HLILCall):
                # Check if next statement is: var = REG[0]
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if cls._is_reg0_assignment(next_stmt):
                        # Merge: function_call(); var = REG[0] -> var = function_call()
                        dest = next_stmt.dest
                        call_expr = stmt.expr
                        optimized.append(HLILAssign(dest, call_expr))
                        i += 2  # Skip both statements
                        continue

            # Pattern 2: REG[0] = value; return; -> return value;
            if isinstance(stmt, HLILAssign) and cls._is_reg0_var(stmt.dest):
                # Check if next statement is return (with no value)
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if isinstance(next_stmt, HLILReturn) and next_stmt.value is None:
                        # Merge: REG[0] = value; return; -> return value;
                        optimized.append(HLILReturn(stmt.src))
                        i += 2  # Skip both statements
                        continue

            # Recursively optimize nested blocks
            if isinstance(stmt, HLILIf):
                cls._optimize_block(stmt.true_block)
                if stmt.false_block:
                    cls._optimize_block(stmt.false_block)

                # Optimization 1: Convert nested if-chain to switch (BEFORE inversion)
                # if (x != 0) { if (x != 1) { ... } else { ... } } else { ... }
                # -> switch (x) { case 0: ...; case 1: ...; }
                switch_stmt = cls._try_convert_to_switch(stmt)
                if switch_stmt:
                    optimized.append(switch_stmt)
                    i += 1
                    continue

                # Optimization 2: Invert empty if branches (AFTER switch attempt)
                # if (cond) {} else { ... } -> if (!cond) { ... }
                if not stmt.true_block.statements and stmt.false_block and stmt.false_block.statements:
                    stmt.condition = cls._negate_condition(stmt.condition)
                    stmt.true_block = stmt.false_block
                    stmt.false_block = None

            elif isinstance(stmt, HLILWhile):
                cls._optimize_block(stmt.body)

            optimized.append(stmt)
            i += 1

        block.statements = optimized

    @classmethod
    def _try_convert_to_switch(cls, if_stmt: HLILIf) -> Optional[HLILSwitch]:
        '''Try to convert nested if-chain to switch statement

        Detects pattern:
            if (x != val0) {
                if (x != val1) {
                    if (x != val2) {
                        default_case
                    } else {
                        case_val2
                    }
                } else {
                    case_val1
                }
            } else {
                case_val0
            }

        Args:
            if_stmt: If statement to analyze

        Returns:
            Switch statement if pattern is detected, None otherwise
        '''
        # Minimum number of cases to convert to switch
        MIN_CASES = 3

        cases = []
        scrutinee = None
        default_body = None

        current_if = if_stmt

        # Walk through nested if chain
        while current_if:
            # Check if condition is: var != const
            if not isinstance(current_if.condition, HLILBinaryOp):
                break

            if current_if.condition.op != '!=':
                break

            # Extract variable and constant
            var_expr = current_if.condition.lhs
            const_expr = current_if.condition.rhs

            # Ensure we have var != const pattern
            if not isinstance(const_expr, HLILConst):
                break

            if not isinstance(var_expr, HLILVar):
                break

            # First iteration: establish scrutinee variable
            if scrutinee is None:
                scrutinee = var_expr
            else:
                # Ensure all conditions use the same variable
                if scrutinee.var.name != var_expr.var.name:
                    break

            # Extract case value and body from false branch
            case_value = const_expr.value
            case_body = current_if.false_block

            if case_body:
                cases.append((case_value, case_body))

            # Continue to nested if in true branch
            if current_if.true_block and len(current_if.true_block.statements) == 1:
                next_stmt = current_if.true_block.statements[0]
                if isinstance(next_stmt, HLILIf):
                    current_if = next_stmt
                    continue

            # True branch is the default case (or end of chain)
            default_body = current_if.true_block
            break

        # Check if we have enough cases to make a switch
        if len(cases) < MIN_CASES or scrutinee is None:
            return None

        # Build switch statement
        switch_cases = []

        # Add collected cases (in reverse order, since we collected from innermost)
        for case_value, case_body in reversed(cases):
            switch_cases.append(HLILSwitchCase(HLILConst(case_value), case_body))

        # Add default case if present and non-empty
        if default_body and default_body.statements:
            switch_cases.append(HLILSwitchCase(None, default_body))

        return HLILSwitch(scrutinee, switch_cases)

    @classmethod
    def _negate_condition(cls, condition: HLILExpression) -> HLILExpression:
        '''Negate a conditional expression

        Args:
            condition: Conditional expression to negate

        Returns:
            Negated expression
        '''
        # Negate binary comparison operators
        if isinstance(condition, HLILBinaryOp):
            negation_map = {
                '==': '!=',
                '!=': '==',
                '<': '>=',
                '>=': '<',
                '>': '<=',
                '<=': '>',
            }

            if condition.op in negation_map:
                return HLILBinaryOp(
                    negation_map[condition.op],
                    condition.lhs,
                    condition.rhs
                )

            # For logical operators, wrap in == 0
            elif condition.op in ('&&', '||'):
                return HLILBinaryOp('==', condition, HLILConst(0))

        # For other expressions, wrap in == 0
        return HLILBinaryOp('==', condition, HLILConst(0))

    @classmethod
    def _is_reg0_assignment(cls, stmt: HLILStatement) -> bool:
        '''Check if statement is: var = REG[0]'''
        if not isinstance(stmt, HLILAssign):
            return False

        # Check if source is REG[0]
        if isinstance(stmt.src, HLILVar):
            return stmt.src.var.name == 'REG[0]'

        return False

    @classmethod
    def _is_reg0_var(cls, expr: HLILExpression) -> bool:
        '''Check if expression is REG[0] variable'''
        if isinstance(expr, HLILVar):
            return expr.var.name == 'REG[0]'
        return False


def optimize_hlil(func: HighLevelILFunction) -> HighLevelILFunction:
    '''Optimize HLIL function

    Args:
        func: Function to optimize

    Returns:
        Optimized function
    '''
    return HLILOptimizer.optimize(func)

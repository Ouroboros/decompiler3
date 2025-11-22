'''
HLIL Optimizer

Optimizes HLIL code by recognizing patterns and simplifying expressions.
'''

from typing import List, Optional
from .hlil import *


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
        # Post-optimization: merge nested switches
        cls._merge_nested_switches(func.body)
        # Post-optimization: extract common returns
        cls._extract_common_returns(func.body)
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
            # Filter out comments to find the real statement
            if current_if.true_block:
                real_stmts = [s for s in current_if.true_block.statements if not isinstance(s, HLILComment)]
                if len(real_stmts) == 1 and isinstance(real_stmts[0], HLILIf):
                    current_if = real_stmts[0]
                    continue

            # True branch is the default case (or end of chain)
            # But first check if it contains a nested switch pattern
            if current_if.true_block:
                real_stmts = [s for s in current_if.true_block.statements if not isinstance(s, HLILComment)]
                if len(real_stmts) == 1:
                    next_stmt = real_stmts[0]
                    # If the default case is another switch on the same variable, merge it
                    if isinstance(next_stmt, HLILSwitch):
                        if isinstance(next_stmt.scrutinee, HLILVar) and scrutinee and next_stmt.scrutinee.var.name == scrutinee.var.name:
                            # Merge nested switch cases
                            for nested_case in next_stmt.cases:
                                if nested_case.is_default():
                                    default_body = nested_case.body
                                else:
                                    cases.append((nested_case.value.value, nested_case.body))
                            break

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
    def _merge_nested_switches(cls, block: HLILBlock):
        '''Merge nested switch statements on the same variable

        Args:
            block: Block to process
        '''
        for i, stmt in enumerate(block.statements):
            if isinstance(stmt, HLILSwitch):
                # Check each case for nested switches
                for case in stmt.cases:
                    # Recursively process case body
                    cls._merge_nested_switches(case.body)

                    # Check if case body is a single nested switch on the same variable
                    if len(case.body.statements) == 1:
                        nested_stmt = case.body.statements[0]
                        if isinstance(nested_stmt, HLILSwitch):
                            if isinstance(nested_stmt.scrutinee, HLILVar) and isinstance(stmt.scrutinee, HLILVar):
                                if nested_stmt.scrutinee.var.name == stmt.scrutinee.var.name:
                                    # Merge nested switch into parent
                                    # This happens when default case contains another switch
                                    if case.is_default():
                                        # Remove the default case
                                        stmt.cases.remove(case)
                                        # Add all nested cases to parent
                                        stmt.cases.extend(nested_stmt.cases)
                                        # Recurse to check for more nesting
                                        cls._merge_nested_switches(block)
                                        return

            elif isinstance(stmt, HLILIf):
                cls._merge_nested_switches(stmt.true_block)
                if stmt.false_block:
                    cls._merge_nested_switches(stmt.false_block)

            elif isinstance(stmt, HLILWhile):
                cls._merge_nested_switches(stmt.body)

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

    @classmethod
    def _extract_common_returns(cls, block: HLILBlock):
        '''Extract common return statements from control flow branches

        If all branches of a switch/if end with the same return value,
        extract it to a single return after the control flow.
        '''
        i = 0
        while i < len(block.statements):
            stmt = block.statements[i]

            # Recursively process nested blocks first
            if isinstance(stmt, HLILIf):
                cls._extract_common_returns(stmt.true_block)
                if stmt.false_block:
                    cls._extract_common_returns(stmt.false_block)

                # Check if both branches return the same value
                common_return = cls._get_common_if_return(stmt)
                if common_return is not None:
                    # Remove return from both branches
                    if stmt.true_block.statements and isinstance(stmt.true_block.statements[-1], HLILReturn):
                        stmt.true_block.statements.pop()
                    if stmt.false_block and stmt.false_block.statements and isinstance(stmt.false_block.statements[-1], HLILReturn):
                        stmt.false_block.statements.pop()

                    # Add common return after if
                    block.statements.insert(i + 1, common_return)

            elif isinstance(stmt, HLILWhile):
                cls._extract_common_returns(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                # Process each case body
                for case in stmt.cases:
                    cls._extract_common_returns(case.body)

                # Check if all cases end with the same return
                common_return = cls._get_common_return(stmt)
                if common_return is not None:
                    # Remove return from all cases
                    for case in stmt.cases:
                        if case.body.statements and isinstance(case.body.statements[-1], HLILReturn):
                            case.body.statements.pop()

                    # Add common return after switch
                    block.statements.insert(i + 1, common_return)

            i += 1

        # Remove unreachable code after returns/breaks
        cls._remove_unreachable_code(block)

    @classmethod
    def _get_common_return(cls, switch_stmt: HLILSwitch) -> Optional[HLILReturn]:
        '''Check if all switch cases end with the same return value

        Args:
            switch_stmt: Switch statement to check

        Returns:
            Common return statement if all cases have the same return, None otherwise
        '''
        if not switch_stmt.cases:
            return None

        common_return = None

        for case in switch_stmt.cases:
            if not case.body.statements:
                return None

            last_stmt = case.body.statements[-1]
            if not isinstance(last_stmt, HLILReturn):
                return None

            # First case: establish common return
            if common_return is None:
                common_return = last_stmt
                continue

            # Check if return values match
            if not cls._return_values_equal(common_return, last_stmt):
                return None

        return common_return

    @classmethod
    def _return_values_equal(cls, ret1: HLILReturn, ret2: HLILReturn) -> bool:
        '''Check if two return statements return the same value

        Args:
            ret1: First return statement
            ret2: Second return statement

        Returns:
            True if both return the same value
        '''
        # Both return void
        if ret1.value is None and ret2.value is None:
            return True

        # One returns void, other doesn't
        if ret1.value is None or ret2.value is None:
            return False

        # Both return constants
        if isinstance(ret1.value, HLILConst) and isinstance(ret2.value, HLILConst):
            return ret1.value.value == ret2.value.value

        # Both return the same variable
        if isinstance(ret1.value, HLILVar) and isinstance(ret2.value, HLILVar):
            return ret1.value.var.name == ret2.value.var.name

        return False

    @classmethod
    def _get_common_if_return(cls, if_stmt: HLILIf) -> Optional[HLILReturn]:
        '''Check if both if branches end with the same return value

        Args:
            if_stmt: If statement to check

        Returns:
            Common return statement if both branches have the same return, None otherwise
        '''
        # Must have both true and false blocks
        if not if_stmt.true_block or not if_stmt.false_block:
            return None

        # Both blocks must have statements
        if not if_stmt.true_block.statements or not if_stmt.false_block.statements:
            return None

        # Both must end with return
        true_last = if_stmt.true_block.statements[-1]
        false_last = if_stmt.false_block.statements[-1]

        if not isinstance(true_last, HLILReturn) or not isinstance(false_last, HLILReturn):
            return None

        # Check if return values match
        if cls._return_values_equal(true_last, false_last):
            return true_last

        return None

    @classmethod
    def _remove_unreachable_code(cls, block: HLILBlock):
        '''Remove unreachable code after return/break/continue statements

        Args:
            block: Block to clean up
        '''
        # Find the first return/break/continue
        for i, stmt in enumerate(block.statements):
            if isinstance(stmt, (HLILReturn, HLILBreak, HLILContinue)):
                # Remove all statements after this one
                if i + 1 < len(block.statements):
                    block.statements = block.statements[:i + 1]
                break


def optimize_hlil(func: HighLevelILFunction) -> HighLevelILFunction:
    '''Optimize HLIL function

    Args:
        func: Function to optimize

    Returns:
        Optimized function
    '''
    return HLILOptimizer.optimize(func)

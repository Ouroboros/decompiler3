'''HLIL Optimizer - pattern recognition and simplification'''

from typing import List, Optional
from .hlil import *


class HLILOptimizer:
    @classmethod
    def optimize(cls, func: HighLevelILFunction) -> HighLevelILFunction:
        '''Apply optimizations: merge switches, extract common returns'''
        cls._optimize_block(func.body)
        cls._merge_nested_switches(func.body)
        cls._extract_common_returns(func.body)
        return func

    @classmethod
    def _optimize_block(cls, block: HLILBlock):
        if not block.statements:
            return

        optimized = []
        i = 0

        while i < len(block.statements):
            stmt = block.statements[i]

            # call(); var = REG[0] -> var = call()
            if isinstance(stmt, HLILExprStmt) and isinstance(stmt.expr, HLILCall):
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if cls._is_reg0_assignment(next_stmt):
                        optimized.append(HLILAssign(next_stmt.dest, stmt.expr))
                        i += 2
                        continue

            # REG[0] = val; return; -> return val;
            if isinstance(stmt, HLILAssign) and cls._is_reg0_var(stmt.dest):
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if isinstance(next_stmt, HLILReturn) and next_stmt.value is None:
                        optimized.append(HLILReturn(stmt.src))
                        i += 2
                        continue

            if isinstance(stmt, HLILIf):
                cls._optimize_block(stmt.true_block)
                if stmt.false_block:
                    cls._optimize_block(stmt.false_block)

                # Convert nested if-chain to switch
                switch_stmt = cls._try_convert_to_switch(stmt)
                if switch_stmt:
                    optimized.append(switch_stmt)
                    i += 1
                    continue

                # Invert empty if: if (c) {} else {...} -> if (!c) {...}
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
        '''Convert nested if-chain to switch (requires 3+ cases)'''
        MIN_CASES = 3

        cases = []
        scrutinee = None
        default_body = None

        current_if = if_stmt

        while current_if:
            if not isinstance(current_if.condition, HLILBinaryOp):
                break

            if current_if.condition.op != '!=':
                break

            var_expr = current_if.condition.lhs
            const_expr = current_if.condition.rhs

            if not isinstance(const_expr, HLILConst):
                break

            if not isinstance(var_expr, HLILVar):
                break

            if scrutinee is None:
                scrutinee = var_expr

            else:
                if scrutinee.var.name != var_expr.var.name:
                    break

            case_value = const_expr.value
            case_body = current_if.false_block

            if case_body:
                cases.append((case_value, case_body))

            if current_if.true_block:
                real_stmts = [s for s in current_if.true_block.statements if not isinstance(s, HLILComment)]
                if len(real_stmts) == 1 and isinstance(real_stmts[0], HLILIf):
                    current_if = real_stmts[0]
                    continue

            # Check if default contains nested switch on same variable
            if current_if.true_block:
                real_stmts = [s for s in current_if.true_block.statements if not isinstance(s, HLILComment)]
                if len(real_stmts) == 1:
                    next_stmt = real_stmts[0]
                    if isinstance(next_stmt, HLILSwitch):
                        if isinstance(next_stmt.scrutinee, HLILVar) and scrutinee and next_stmt.scrutinee.var.name == scrutinee.var.name:
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
        '''Check if statement assigns from REGS[0]'''
        if not isinstance(stmt, HLILAssign):
            return False

        if isinstance(stmt.src, HLILVar):
            var = stmt.src.var
            return var.kind == VariableKind.REG and var.index == 0

        return False

    @classmethod
    def _is_reg0_var(cls, expr: HLILExpression) -> bool:
        '''Check if expression is REGS[0]'''
        if isinstance(expr, HLILVar):
            var = expr.var
            return var.kind == VariableKind.REG and var.index == 0

        return False

    @classmethod
    def _extract_common_returns(cls, block: HLILBlock):
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
        # Find the first return/break/continue
        for i, stmt in enumerate(block.statements):
            if isinstance(stmt, (HLILReturn, HLILBreak, HLILContinue)):
                # Remove all statements after this one
                if i + 1 < len(block.statements):
                    block.statements = block.statements[:i + 1]
                break


def optimize_hlil(func: HighLevelILFunction) -> HighLevelILFunction:
    return HLILOptimizer.optimize(func)

'''HLIL Passes'''

from typing import Optional
from ir.pipeline import Pass
from ir.mlil.mlil import MediumLevelILFunction
from .hlil import *
from .mlil_to_hlil import MLILToHLILConverter


class MLILToHLILPass(Pass):
    '''MLIL to HLIL conversion pass'''

    def run(self, mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
        converter = MLILToHLILConverter(mlil_func)
        return converter.convert()


class ExpressionSimplificationPass(Pass):
    '''Simplify call/return expressions'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        self._optimize_block(func.body)
        return func

    def _optimize_block(self, block: HLILBlock):
        if not block or not block.statements:
            return

        optimized = []
        i = 0

        while i < len(block.statements):
            stmt = block.statements[i]

            # call(); var = REG[0] -> var = call()
            if isinstance(stmt, HLILExprStmt) and isinstance(stmt.expr, HLILCall):
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if self._is_reg0_assignment(next_stmt):
                        optimized.append(HLILAssign(next_stmt.dest, stmt.expr))
                        i += 2
                        continue

            # REG[0] = val; return; -> return val;
            if isinstance(stmt, HLILAssign) and self._is_reg0_var(stmt.dest):
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if isinstance(next_stmt, HLILReturn) and next_stmt.value is None:
                        optimized.append(HLILReturn(stmt.src))
                        i += 2
                        continue

            # Recurse into nested blocks
            if isinstance(stmt, HLILIf):
                self._optimize_block(stmt.true_block)
                self._optimize_block(stmt.false_block)

            elif isinstance(stmt, HLILWhile):
                self._optimize_block(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._optimize_block(case.body)

            optimized.append(stmt)
            i += 1

        block.statements = optimized

    def _is_reg0_assignment(self, stmt: HLILStatement) -> bool:
        if not isinstance(stmt, HLILAssign):
            return False

        if isinstance(stmt.src, HLILVar):
            var = stmt.src.var
            return var.kind == VariableKind.REG and var.index == 0

        return False

    def _is_reg0_var(self, expr: HLILExpression) -> bool:
        if isinstance(expr, HLILVar):
            var = expr.var
            return var.kind == VariableKind.REG and var.index == 0

        return False


class ControlFlowOptimizationPass(Pass):
    '''Control flow optimizations: if-to-switch, empty-if inversion, else-if flattening, switch merging'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        self._optimize_block(func.body)
        self._merge_nested_switches(func.body)
        return func

    def _optimize_block(self, block: HLILBlock):
        if not block or not block.statements:
            return

        optimized = []

        for stmt in block.statements:
            if isinstance(stmt, HLILIf):
                self._optimize_block(stmt.true_block)
                self._optimize_block(stmt.false_block)

                # Convert nested if-chain to switch
                switch_stmt = self._try_convert_to_switch(stmt)
                if switch_stmt:
                    optimized.append(switch_stmt)
                    continue

                # Invert empty if: if (c) {} else {...} -> if (!c) {...}
                if not stmt.true_block.statements and stmt.false_block and stmt.false_block.statements:
                    stmt.condition = self._negate_condition(stmt.condition)
                    stmt.true_block = stmt.false_block
                    stmt.false_block = None

                # Flatten else-if
                self._try_flatten_else_if(stmt)

            elif isinstance(stmt, HLILWhile):
                self._optimize_block(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._optimize_block(case.body)

            optimized.append(stmt)

        block.statements = optimized

    def _is_boolean_expr(self, expr: HLILExpression) -> bool:
        if isinstance(expr, HLILBinaryOp):
            return expr.op in ('==', '!=', '<', '<=', '>', '>=', '&&', '||')

        elif isinstance(expr, HLILUnaryOp):
            return expr.op == '!'

        return False

    def _try_flatten_else_if(self, if_stmt: HLILIf):
        '''Flatten pattern: else { comment?; var = bool_expr; if (var == 0) {...} }'''
        false_block = if_stmt.false_block
        if not false_block or not false_block.statements:
            return

        stmts = false_block.statements
        idx = 0

        # Optional leading comment
        comment = None
        if isinstance(stmts[idx], HLILComment):
            comment = stmts[idx]
            idx += 1

        if len(stmts) - idx != 2:
            return

        assign_stmt = stmts[idx]
        if not isinstance(assign_stmt, HLILAssign):
            return

        if not isinstance(assign_stmt.dest, HLILVar):
            return

        if not self._is_boolean_expr(assign_stmt.src):
            return

        assigned_var = assign_stmt.dest.var
        condition_expr = assign_stmt.src
        idx += 1

        inner_if = stmts[idx]
        if not isinstance(inner_if, HLILIf):
            return

        cond = inner_if.condition
        if not isinstance(cond, HLILBinaryOp):
            return

        if cond.op not in ('==', '!='):
            return

        if not isinstance(cond.rhs, HLILConst) or cond.rhs.value != 0:
            return

        if not isinstance(cond.lhs, HLILVar):
            return

        if cond.lhs.var != assigned_var:
            return

        has_inner_else = inner_if.false_block and inner_if.false_block.statements
        negate = (cond.op == '==')

        if negate and has_inner_else:
            new_condition = condition_expr
            new_true_block = inner_if.false_block
            new_false_block = inner_if.true_block

        elif negate:
            new_condition = self._negate_condition(condition_expr)
            new_true_block = inner_if.true_block
            new_false_block = None

        else:
            new_condition = condition_expr
            new_true_block = inner_if.true_block
            new_false_block = inner_if.false_block

        if comment and new_true_block:
            new_true_block.statements.insert(0, comment)

        if_stmt.false_block = HLILBlock([
            HLILIf(new_condition, new_true_block, new_false_block)
        ])

    def _try_convert_to_switch(self, if_stmt: HLILIf) -> Optional[HLILSwitch]:
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

        if len(cases) < MIN_CASES or scrutinee is None:
            return None

        switch_cases = []
        for case_value, case_body in reversed(cases):
            switch_cases.append(HLILSwitchCase(HLILConst(case_value), case_body))

        if default_body and default_body.statements:
            switch_cases.append(HLILSwitchCase(None, default_body))

        return HLILSwitch(scrutinee, switch_cases)

    def _merge_nested_switches(self, block: HLILBlock):
        if not block or not block.statements:
            return

        for stmt in block.statements:
            if isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._merge_nested_switches(case.body)

                    if len(case.body.statements) == 1:
                        nested_stmt = case.body.statements[0]
                        if isinstance(nested_stmt, HLILSwitch):
                            if isinstance(nested_stmt.scrutinee, HLILVar) and isinstance(stmt.scrutinee, HLILVar):
                                if nested_stmt.scrutinee.var.name == stmt.scrutinee.var.name:
                                    if case.is_default():
                                        stmt.cases.remove(case)
                                        stmt.cases.extend(nested_stmt.cases)
                                        self._merge_nested_switches(block)
                                        return

            elif isinstance(stmt, HLILIf):
                self._merge_nested_switches(stmt.true_block)
                self._merge_nested_switches(stmt.false_block)

            elif isinstance(stmt, HLILWhile):
                self._merge_nested_switches(stmt.body)

    def _negate_condition(self, condition: HLILExpression) -> HLILExpression:
        if isinstance(condition, HLILBinaryOp):
            negation_map = {
                '==': '!=', '!=': '==',
                '<': '>=', '>=': '<',
                '>': '<=', '<=': '>',
            }

            if condition.op in negation_map:
                return HLILBinaryOp(negation_map[condition.op], condition.lhs, condition.rhs)

            elif condition.op in ('&&', '||'):
                return HLILBinaryOp('==', condition, HLILConst(0))

        return HLILBinaryOp('==', condition, HLILConst(0))


class CommonReturnExtractionPass(Pass):
    '''Extract common return statements from branches'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        self._extract_common_returns(func.body)
        return func

    def _extract_common_returns(self, block: HLILBlock):
        if not block or not block.statements:
            return

        i = 0
        while i < len(block.statements):
            stmt = block.statements[i]

            if isinstance(stmt, HLILIf):
                self._extract_common_returns(stmt.true_block)
                self._extract_common_returns(stmt.false_block)

                common_return = self._get_common_if_return(stmt)
                if common_return is not None:
                    if stmt.true_block.statements and isinstance(stmt.true_block.statements[-1], HLILReturn):
                        stmt.true_block.statements.pop()

                    if stmt.false_block and stmt.false_block.statements and isinstance(stmt.false_block.statements[-1], HLILReturn):
                        stmt.false_block.statements.pop()

                    block.statements.insert(i + 1, common_return)

            elif isinstance(stmt, HLILWhile):
                self._extract_common_returns(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._extract_common_returns(case.body)

                common_return = self._get_common_switch_return(stmt)
                if common_return is not None:
                    for case in stmt.cases:
                        if case.body.statements and isinstance(case.body.statements[-1], HLILReturn):
                            case.body.statements.pop()

                    block.statements.insert(i + 1, common_return)

            i += 1

    def _get_common_switch_return(self, switch_stmt: HLILSwitch) -> Optional[HLILReturn]:
        if not switch_stmt.cases:
            return None

        common_return = None

        for case in switch_stmt.cases:
            if not case.body.statements:
                return None

            last_stmt = case.body.statements[-1]
            if not isinstance(last_stmt, HLILReturn):
                return None

            if common_return is None:
                common_return = last_stmt
                continue

            if not self._return_values_equal(common_return, last_stmt):
                return None

        return common_return

    def _return_values_equal(self, ret1: HLILReturn, ret2: HLILReturn) -> bool:
        if ret1.value is None and ret2.value is None:
            return True

        if ret1.value is None or ret2.value is None:
            return False

        if isinstance(ret1.value, HLILConst) and isinstance(ret2.value, HLILConst):
            return ret1.value.value == ret2.value.value

        if isinstance(ret1.value, HLILVar) and isinstance(ret2.value, HLILVar):
            return ret1.value.var.name == ret2.value.var.name

        return False

    def _get_common_if_return(self, if_stmt: HLILIf) -> Optional[HLILReturn]:
        if not if_stmt.true_block or not if_stmt.false_block:
            return None

        if not if_stmt.true_block.statements or not if_stmt.false_block.statements:
            return None

        true_last = if_stmt.true_block.statements[-1]
        false_last = if_stmt.false_block.statements[-1]

        if not isinstance(true_last, HLILReturn) or not isinstance(false_last, HLILReturn):
            return None

        if self._return_values_equal(true_last, false_last):
            return true_last

        return None


class DeadCodeEliminationPass(Pass):
    '''Remove unreachable code after return/break/continue'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        self._remove_unreachable(func.body)
        return func

    def _remove_unreachable(self, block: HLILBlock):
        if not block or not block.statements:
            return

        for stmt in block.statements:
            if isinstance(stmt, HLILIf):
                self._remove_unreachable(stmt.true_block)
                self._remove_unreachable(stmt.false_block)

            elif isinstance(stmt, HLILWhile):
                self._remove_unreachable(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._remove_unreachable(case.body)

        for i, stmt in enumerate(block.statements):
            if isinstance(stmt, (HLILReturn, HLILBreak, HLILContinue)):
                if i + 1 < len(block.statements):
                    block.statements = block.statements[:i + 1]
                break

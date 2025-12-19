'''Copy Propagation Pass'''

from dataclasses import dataclass
from typing import Optional
from ir.pipeline import Pass
from .hlil import (
    HighLevelILFunction,
    HLILBlock,
    HLILInstruction,
    HLILExpression,
    HLILVar,
    HLILConst,
    HLILBinaryOp,
    HLILUnaryOp,
    HLILCall,
    HLILSyscall,
    HLILExternCall,
    HLILIf,
    HLILWhile,
    HLILDoWhile,
    HLILSwitch,
    HLILAssign,
    HLILExprStmt,
    HLILReturn,
    HLILVariable,
    BinaryOp,
    UnaryOp,
)


@dataclass
class UseInfo:
    '''Information about a variable use location'''
    in_entered_loop: bool = False
    containing_loop: Optional[HLILInstruction] = None


class CopyPropagationPass(Pass):
    '''Propagate single-use variable copies: var = expr; use(var) -> use(expr)'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        self._propagate_copies(func.body)
        return func

    def _propagate_copies(self, block: HLILBlock):
        if not block or not block.statements:
            return

        # Recurse into nested blocks first
        for stmt in block.statements:
            if isinstance(stmt, HLILIf):
                self._propagate_copies(stmt.true_block)
                self._propagate_copies(stmt.false_block)

            elif isinstance(stmt, HLILWhile):
                self._propagate_copies(stmt.body)

            elif isinstance(stmt, HLILDoWhile):
                self._propagate_copies(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._propagate_copies(case.body)

        # Find and propagate single-use copies
        i = 0
        while i < len(block.statements):
            stmt = block.statements[i]

            if not isinstance(stmt, HLILAssign) or not isinstance(stmt.dest, HLILVar):
                i += 1
                continue

            var = stmt.dest.var
            expr = stmt.src

            # Skip boolean expressions (handled by ControlFlowOptimizationPass)
            if self._is_boolean_expr(expr):
                i += 1
                continue

            # Find reachable uses in remaining statements
            remaining = block.statements[i + 1:]
            if not remaining:
                i += 1
                continue

            uses, _ = self._find_reachable_uses_in_list(remaining, var, in_entered_loop=False)

            if len(uses) != 1:
                i += 1
                continue

            use = uses[0]
            source_vars = self._collect_vars(expr)

            # Check loop constraints
            if use.in_entered_loop:
                if self._has_side_effects(expr):
                    i += 1
                    continue

                if use.containing_loop and self._modifies_vars(use.containing_loop, source_vars):
                    i += 1
                    continue

            # Check source not modified before use
            next_stmt = block.statements[i + 1]
            if not self._is_use_at_entry(next_stmt, var):
                if self._modifies_vars(next_stmt, source_vars):
                    i += 1
                    continue

            # Safe to propagate: replace and delete
            self._replace_var(remaining, var, expr)
            block.statements.pop(i)
            # Don't increment i

    BOOLEAN_BINARY_OPS = {
        BinaryOp.EQ, BinaryOp.NE,
        BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE,
        BinaryOp.AND, BinaryOp.OR,
    }

    def _is_boolean_expr(self, expr: HLILExpression) -> bool:
        if isinstance(expr, HLILBinaryOp):
            return expr.op in self.BOOLEAN_BINARY_OPS

        if isinstance(expr, HLILUnaryOp):
            return expr.op == UnaryOp.NOT

        return False

    def _find_reachable_uses_in_list(self, stmts: list, var: HLILVariable, in_entered_loop: bool, containing_loop=None) -> tuple[list[UseInfo], bool]:
        '''Find reachable uses in a list of statements. Returns (uses, any_kill).'''
        uses = []
        any_kill = False
        for stmt in stmts:
            stmt_uses, killed = self._find_reachable_uses(stmt, var, in_entered_loop, containing_loop)
            uses.extend(stmt_uses)
            if killed:
                any_kill = True
                break
        return (uses, any_kill)

    def _find_reachable_uses(self, node, var: HLILVariable, in_entered_loop: bool, containing_loop=None) -> tuple[list[UseInfo], bool]:
        '''Find reachable uses of var in node. Returns (list of UseInfo, any_kill).'''
        if node is None:
            return ([], False)

        # Leaf nodes
        if isinstance(node, HLILVar):
            if node.var == var:
                return ([UseInfo(in_entered_loop=in_entered_loop, containing_loop=containing_loop)], False)
            return ([], False)

        if isinstance(node, HLILConst):
            return ([], False)

        # Expression nodes
        if isinstance(node, HLILBinaryOp):
            left_uses, _ = self._find_reachable_uses(node.lhs, var, in_entered_loop, containing_loop)
            right_uses, _ = self._find_reachable_uses(node.rhs, var, in_entered_loop, containing_loop)
            return (left_uses + right_uses, False)

        if isinstance(node, HLILUnaryOp):
            return self._find_reachable_uses(node.operand, var, in_entered_loop, containing_loop)

        if isinstance(node, (HLILCall, HLILSyscall, HLILExternCall)):
            all_uses = []
            for arg in node.args:
                arg_uses, _ = self._find_reachable_uses(arg, var, in_entered_loop, containing_loop)
                all_uses.extend(arg_uses)
            return (all_uses, False)

        # Statement nodes
        if isinstance(node, HLILAssign):
            src_uses, _ = self._find_reachable_uses(node.src, var, in_entered_loop, containing_loop)
            kills = isinstance(node.dest, HLILVar) and node.dest.var == var
            return (src_uses, kills)

        if isinstance(node, HLILExprStmt):
            return self._find_reachable_uses(node.expr, var, in_entered_loop, containing_loop)

        if isinstance(node, HLILBlock):
            all_uses = []
            for stmt in node.statements:
                stmt_uses, any_kill = self._find_reachable_uses(stmt, var, in_entered_loop, containing_loop)
                all_uses.extend(stmt_uses)
                if any_kill:
                    return (all_uses, True)
            return (all_uses, False)

        if isinstance(node, HLILIf):
            cond_uses, _ = self._find_reachable_uses(node.condition, var, in_entered_loop, containing_loop)
            true_uses, true_kill = self._find_reachable_uses(node.true_block, var, in_entered_loop, containing_loop)

            if node.false_block:
                false_uses, false_kill = self._find_reachable_uses(node.false_block, var, in_entered_loop, containing_loop)

            else:
                false_uses, false_kill = [], False

            any_kill = true_kill or false_kill
            return (cond_uses + true_uses + false_uses, any_kill)

        if isinstance(node, HLILWhile):
            cond_uses, _ = self._find_reachable_uses(node.condition, var, True, node)
            body_uses, _ = self._find_reachable_uses(node.body, var, True, node)
            return (cond_uses + body_uses, False)

        if isinstance(node, HLILDoWhile):
            body_uses, body_kill = self._find_reachable_uses(node.body, var, True, node)
            if body_kill:
                return (body_uses, True)

            cond_uses, _ = self._find_reachable_uses(node.condition, var, True, node)
            return (body_uses + cond_uses, False)

        if isinstance(node, HLILSwitch):
            scrut_uses, _ = self._find_reachable_uses(node.scrutinee, var, in_entered_loop, containing_loop)
            all_uses = scrut_uses
            any_kill = False
            for case in node.cases:
                case_uses, case_kill = self._find_reachable_uses(case.body, var, in_entered_loop, containing_loop)
                all_uses.extend(case_uses)
                if case_kill:
                    any_kill = True

            return (all_uses, any_kill)

        if isinstance(node, HLILReturn):
            if node.value:
                return self._find_reachable_uses(node.value, var, in_entered_loop, containing_loop)
            return ([], False)

        return ([], False)

    def _collect_vars(self, expr: HLILExpression) -> set:
        '''Collect all variables referenced in expression.'''
        result = set()

        if isinstance(expr, HLILVar):
            result.add(expr.var)

        elif isinstance(expr, HLILBinaryOp):
            result.update(self._collect_vars(expr.lhs))
            result.update(self._collect_vars(expr.rhs))

        elif isinstance(expr, HLILUnaryOp):
            result.update(self._collect_vars(expr.operand))

        elif isinstance(expr, (HLILCall, HLILSyscall, HLILExternCall)):
            for arg in expr.args:
                result.update(self._collect_vars(arg))

        return result

    def _has_side_effects(self, expr: HLILExpression) -> bool:
        '''Check if expression has side effects.'''
        if isinstance(expr, (HLILCall, HLILSyscall, HLILExternCall)):
            return True

        if isinstance(expr, HLILBinaryOp):
            return self._has_side_effects(expr.lhs) or self._has_side_effects(expr.rhs)

        if isinstance(expr, HLILUnaryOp):
            return self._has_side_effects(expr.operand)

        return False

    def _modifies_vars(self, node, vars_to_check: set) -> bool:
        '''Check if node modifies any variable in vars_to_check.'''
        if node is None or not vars_to_check:
            return False

        if isinstance(node, HLILAssign):
            if isinstance(node.dest, HLILVar) and node.dest.var in vars_to_check:
                return True
            return self._modifies_vars(node.src, vars_to_check)

        if isinstance(node, (HLILCall, HLILSyscall, HLILExternCall)):
            # Conservative: calls may modify any variable
            return True

        if isinstance(node, HLILExprStmt):
            return self._modifies_vars(node.expr, vars_to_check)

        if isinstance(node, HLILBlock):
            return any(self._modifies_vars(stmt, vars_to_check) for stmt in node.statements)

        if isinstance(node, HLILIf):
            return (self._modifies_vars(node.condition, vars_to_check) or
                    self._modifies_vars(node.true_block, vars_to_check) or
                    self._modifies_vars(node.false_block, vars_to_check))

        if isinstance(node, HLILWhile):
            return (self._modifies_vars(node.condition, vars_to_check) or
                    self._modifies_vars(node.body, vars_to_check))

        if isinstance(node, HLILDoWhile):
            return (self._modifies_vars(node.body, vars_to_check) or
                    self._modifies_vars(node.condition, vars_to_check))

        if isinstance(node, HLILSwitch):
            if self._modifies_vars(node.scrutinee, vars_to_check):
                return True
            return any(self._modifies_vars(case.body, vars_to_check) for case in node.cases)

        return False

    def _is_use_at_entry(self, stmt, var: HLILVariable) -> bool:
        '''Check if var's use is at the entry point of stmt (evaluated first).'''
        if isinstance(stmt, HLILIf):
            return self._contains_var(stmt.condition, var)

        if isinstance(stmt, HLILWhile):
            return self._contains_var(stmt.condition, var)

        if isinstance(stmt, HLILAssign):
            return self._contains_var(stmt.src, var)

        if isinstance(stmt, HLILExprStmt):
            return self._contains_var(stmt.expr, var)

        if isinstance(stmt, HLILReturn):
            return self._contains_var(stmt.value, var)

        return False

    def _contains_var(self, expr, var: HLILVariable) -> bool:
        '''Check if expression contains var.'''
        if expr is None:
            return False

        if isinstance(expr, HLILVar):
            return expr.var == var

        if isinstance(expr, HLILBinaryOp):
            return self._contains_var(expr.lhs, var) or self._contains_var(expr.rhs, var)

        if isinstance(expr, HLILUnaryOp):
            return self._contains_var(expr.operand, var)

        if isinstance(expr, (HLILCall, HLILSyscall, HLILExternCall)):
            return any(self._contains_var(arg, var) for arg in expr.args)

        return False

    def _replace_var(self, stmts: list, var: HLILVariable, replacement: HLILExpression) -> bool:
        '''Replace first occurrence of var with replacement. Returns True if replaced.'''
        for stmt in stmts:
            if self._replace_var_in_node(stmt, var, replacement):
                return True
        return False

    def _replace_var_in_node(self, node, var: HLILVariable, replacement: HLILExpression) -> bool:
        '''Replace var in node. Returns True if replaced.'''
        if node is None:
            return False

        if isinstance(node, HLILBinaryOp):
            if isinstance(node.lhs, HLILVar) and node.lhs.var == var:
                node.lhs = replacement
                return True

            if self._replace_var_in_node(node.lhs, var, replacement):
                return True

            if isinstance(node.rhs, HLILVar) and node.rhs.var == var:
                node.rhs = replacement
                return True

            return self._replace_var_in_node(node.rhs, var, replacement)

        if isinstance(node, HLILUnaryOp):
            if isinstance(node.operand, HLILVar) and node.operand.var == var:
                node.operand = replacement
                return True

            return self._replace_var_in_node(node.operand, var, replacement)

        if isinstance(node, (HLILCall, HLILSyscall, HLILExternCall)):
            for i, arg in enumerate(node.args):
                if isinstance(arg, HLILVar) and arg.var == var:
                    node.args[i] = replacement
                    return True

                if self._replace_var_in_node(arg, var, replacement):
                    return True

            return False

        if isinstance(node, HLILAssign):
            if isinstance(node.src, HLILVar) and node.src.var == var:
                node.src = replacement
                return True

            return self._replace_var_in_node(node.src, var, replacement)

        if isinstance(node, HLILExprStmt):
            if isinstance(node.expr, HLILVar) and node.expr.var == var:
                node.expr = replacement
                return True

            return self._replace_var_in_node(node.expr, var, replacement)

        if isinstance(node, HLILIf):
            if isinstance(node.condition, HLILVar) and node.condition.var == var:
                node.condition = replacement
                return True

            if self._replace_var_in_node(node.condition, var, replacement):
                return True

            if node.true_block:
                if self._replace_var(node.true_block.statements, var, replacement):
                    return True

            if node.false_block:
                return self._replace_var(node.false_block.statements, var, replacement)

            return False

        if isinstance(node, HLILWhile):
            if isinstance(node.condition, HLILVar) and node.condition.var == var:
                node.condition = replacement
                return True

            if self._replace_var_in_node(node.condition, var, replacement):
                return True

            if node.body:
                return self._replace_var(node.body.statements, var, replacement)

            return False

        if isinstance(node, HLILDoWhile):
            if node.body:
                if self._replace_var(node.body.statements, var, replacement):
                    return True

            if isinstance(node.condition, HLILVar) and node.condition.var == var:
                node.condition = replacement
                return True

            return self._replace_var_in_node(node.condition, var, replacement)

        if isinstance(node, HLILSwitch):
            if isinstance(node.scrutinee, HLILVar) and node.scrutinee.var == var:
                node.scrutinee = replacement
                return True

            if self._replace_var_in_node(node.scrutinee, var, replacement):
                return True

            for case in node.cases:
                if self._replace_var(case.body.statements, var, replacement):
                    return True

            return False

        if isinstance(node, HLILReturn):
            if node.value:
                if isinstance(node.value, HLILVar) and node.value.var == var:
                    node.value = replacement
                    return True

                return self._replace_var_in_node(node.value, var, replacement)

            return False

        return False

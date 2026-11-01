'''Expression Simplification Pass'''

from ir.pipeline import Pass
from ..hlil import (
    HighLevelILFunction,
    HLILBlock,
    HLILStatement,
    HLILExpression,
    HLILVar,
    HLILBinaryOp,
    HLILUnaryOp,
    HLILAddressOf,
    HLILCall,
    HLILIf,
    HLILWhile,
    HLILSwitch,
    HLILAssign,
    HLILExprStmt,
    HLILReturn,
    VariableKind,
)


class ExpressionSimplificationPass(Pass):
    '''Simplify call/return expressions'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        # Calls leave their result in this register (target calling convention)
        self.return_register = func.target.calls.return_register
        self._optimize_block(func.body)
        return func

    def _optimize_block(self, block: HLILBlock):
        if not block or not block.statements:
            return

        optimized = []
        i = 0

        while i < len(block.statements):
            stmt = block.statements[i]

            # call(); var = ret_reg -> var = call()   (ret_reg: the return register)
            if isinstance(stmt, HLILExprStmt) and isinstance(stmt.expr, HLILCall):
                if i + 1 < len(block.statements):
                    next_stmt = block.statements[i + 1]
                    if self._is_reg0_assignment(next_stmt):
                        optimized.append(HLILAssign(next_stmt.dest, stmt.expr))
                        i += 2
                        continue

            # ret_reg = call(); var = ret_reg -> var = call()
            # Only for function calls - other assignments may be read in branches
            if isinstance(stmt, HLILAssign) and self._is_reg0_var(stmt.dest):
                if isinstance(stmt.src, HLILCall):
                    if i + 1 < len(block.statements):
                        next_stmt = block.statements[i + 1]
                        if self._is_reg0_assignment(next_stmt):
                            # Check that ret_reg is not read after this until the next call or write
                            if not self._has_reg0_read_before_next_call(block.statements, i + 2):
                                optimized.append(HLILAssign(next_stmt.dest, stmt.src))
                                i += 2
                                continue

            # ret_reg = val; return; -> return val;
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
            return var.kind == VariableKind.REG and var.index == self.return_register

        return False

    def _is_reg0_var(self, expr: HLILExpression) -> bool:
        if isinstance(expr, HLILVar):
            var = expr.var
            return var.kind == VariableKind.REG and var.index == self.return_register

        return False

    def _has_reg0_read_before_next_call(self, statements: list, start_idx: int) -> bool:
        '''Check if the return register is read before the next function call or the return register write'''
        for j in range(start_idx, len(statements)):
            stmt = statements[j]

            # call(); - overwrites the return register
            if isinstance(stmt, HLILExprStmt) and isinstance(stmt.expr, HLILCall):
                return False

            # var = call(); - also overwrites the return register
            if isinstance(stmt, HLILAssign) and isinstance(stmt.src, HLILCall):
                return False

            # ret_reg = expr; - overwrites the return register
            if isinstance(stmt, HLILAssign) and self._is_reg0_var(stmt.dest):
                return False

            # Check if this statement reads the return register
            if self._expr_reads_reg0(stmt):
                return True

        return False

    def _expr_reads_reg0(self, node) -> bool:
        '''Check if expression reads the return register'''
        if node is None:
            return False

        if isinstance(node, HLILVar):
            return node.var.kind == VariableKind.REG and node.var.index == self.return_register

        if isinstance(node, HLILAssign):
            # Only check src, not dest (dest is a write)
            return self._expr_reads_reg0(node.src)

        if isinstance(node, HLILBinaryOp):
            return self._expr_reads_reg0(node.lhs) or self._expr_reads_reg0(node.rhs)

        if isinstance(node, (HLILUnaryOp, HLILAddressOf)):
            return self._expr_reads_reg0(node.operand)

        if isinstance(node, HLILCall):
            return any(self._expr_reads_reg0(arg) for arg in node.args)

        if isinstance(node, HLILExprStmt):
            return self._expr_reads_reg0(node.expr)

        if isinstance(node, HLILReturn):
            return self._expr_reads_reg0(node.value)

        if isinstance(node, HLILIf):
            return self._expr_reads_reg0(node.condition)

        return False

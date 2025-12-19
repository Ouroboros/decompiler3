'''MLIL SSA Optimizer - constant/copy propagation and DCE'''

from typing import Dict, List, Optional, Set
from .mlil import *
from .mlil_ssa import *


class SSAOptimizer:
    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.ssa_defs: Dict[MLILVariableSSA, MLILSetVarSSA] = {}
        self.ssa_uses: Dict[MLILVariableSSA, List[MediumLevelILInstruction]] = {}
        self.constants: Dict[MLILVariableSSA, int] = {}

    def optimize(self) -> MediumLevelILFunction:
        changed = True
        iterations = 0
        max_iterations = 10

        # Run SCCP first for aggressive constant propagation and unreachable code elimination
        sccp = SCCP(self.function)
        sccp.run()

        while changed and iterations < max_iterations:
            changed = False
            changed |= self.propagate_constants()
            # changed |= self.fold_constants()  # Temporarily disabled
            changed |= self.simplify_expressions()
            changed |= self.simplify_conditions()
            changed |= self.propagate_copies()
            changed |= self.eliminate_dead_code()

            # Eliminate dead phi sources (constants only used by Phi with unused result)
            eliminator = DeadPhiSourceEliminator(self.function)
            changed |= eliminator.run()

            iterations += 1

        return self.function

    def _build_def_use_chains(self):
        '''Build SSA def-use chains'''
        self.ssa_defs = {}
        self.ssa_uses = {}
        self.constants = {}

        for block in self.function.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVarSSA):
                    self.ssa_defs[inst.var] = inst

                    if isinstance(inst.value, MLILConst):
                        self.constants[inst.var] = inst.value.value

                elif isinstance(inst, MLILPhi):
                    self.ssa_defs[inst.dest] = inst

                self._collect_uses_in_inst(inst)

    def _collect_uses_in_inst(self, inst: MediumLevelILInstruction):
        '''Collect SSA variable uses in instruction'''
        if isinstance(inst, MLILVarSSA):
            if inst.var not in self.ssa_uses:
                self.ssa_uses[inst.var] = []
            self.ssa_uses[inst.var].append(inst)

        elif isinstance(inst, MLILSetVarSSA):
            self._collect_uses_in_expr(inst.value)

        elif isinstance(inst, MLILPhi):
            for source_var, _ in inst.sources:
                if source_var not in self.ssa_uses:
                    self.ssa_uses[source_var] = []
                self.ssa_uses[source_var].append(inst)

        elif isinstance(inst, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            self._collect_uses_in_expr(inst.lhs)
            self._collect_uses_in_expr(inst.rhs)

        elif isinstance(inst, (MLILNeg, MLILLogicalNot, MLILBitwiseNot, MLILTestZero)):
            self._collect_uses_in_expr(inst.operand)

        elif isinstance(inst, MLILIf):
            self._collect_uses_in_expr(inst.condition)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                self._collect_uses_in_expr(inst.value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            for arg in inst.args:
                self._collect_uses_in_expr(arg)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            self._collect_uses_in_expr(inst.value)

    def _collect_uses_in_expr(self, expr: MediumLevelILInstruction):
        self._collect_uses_in_inst(expr)

    def propagate_constants(self) -> bool:
        '''Replace SSA variables with constant values (no folding)'''
        self._build_def_use_chains()
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._propagate_constants_in_inst(inst)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _propagate_constants_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Propagate constants in a single instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._propagate_constants_in_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._propagate_constants_in_expr(inst.condition)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._propagate_constants_in_expr(inst.value)
                if new_value is not inst.value:
                    return MLILRet(new_value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._propagate_constants_in_expr(arg) for arg in inst.args]
            if any(new_args[i] is not inst.args[i] for i in range(len(inst.args))):
                if isinstance(inst, MLILCall):
                    return MLILCall(inst.target, new_args)
                elif isinstance(inst, MLILSyscall):
                    return MLILSyscall(inst.subsystem, inst.cmd, new_args)
                elif isinstance(inst, MLILCallScript):
                    return MLILCallScript(inst.module, inst.func, new_args)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._propagate_constants_in_expr(inst.value)
            if new_value is not inst.value:
                if isinstance(inst, MLILStoreGlobal):
                    return MLILStoreGlobal(inst.index, new_value)
                else:
                    return MLILStoreReg(inst.index, new_value)

        return inst

    def _propagate_constants_in_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Replace SSA variables with constants'''
        if isinstance(expr, MLILVarSSA):
            if expr.var in self.constants:
                return MLILConst(self.constants[expr.var], is_hex = False)

        # Recurse into binary operations
        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._propagate_constants_in_expr(expr.lhs)
            rhs = self._propagate_constants_in_expr(expr.rhs)

            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILBitwiseNot)):
            operand = self._propagate_constants_in_expr(expr.operand)

            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def _eval_binary_const(self, op_type, lhs: int, rhs: int) -> Optional[int]:
        '''Evaluate binary operation on constants'''
        try:
            if op_type == MLILAdd:
                return lhs + rhs
            elif op_type == MLILSub:
                return lhs - rhs
            elif op_type == MLILMul:
                return lhs * rhs
            elif op_type == MLILDiv:
                return lhs // rhs if rhs != 0 else None
            elif op_type == MLILMod:
                return lhs % rhs if rhs != 0 else None
            elif op_type == MLILAnd:
                return lhs & rhs
            elif op_type == MLILOr:
                return lhs | rhs
            elif op_type == MLILXor:
                return lhs ^ rhs
            elif op_type == MLILShl:
                return lhs << rhs
            elif op_type == MLILShr:
                return lhs >> rhs
            elif op_type == MLILLogicalAnd:
                return 1 if (lhs and rhs) else 0
            elif op_type == MLILLogicalOr:
                return 1 if (lhs or rhs) else 0
            elif op_type == MLILEq:
                return 1 if lhs == rhs else 0
            elif op_type == MLILNe:
                return 1 if lhs != rhs else 0
            elif op_type == MLILLt:
                return 1 if lhs < rhs else 0
            elif op_type == MLILLe:
                return 1 if lhs <= rhs else 0
            elif op_type == MLILGt:
                return 1 if lhs > rhs else 0
            elif op_type == MLILGe:
                return 1 if lhs >= rhs else 0
        except (OverflowError, ZeroDivisionError):
            return None
        return None

    def _eval_unary_const(self, op_type, operand: int) -> Optional[int]:
        '''Evaluate unary operation on constant'''
        try:
            if op_type == MLILNeg:
                return -operand

            elif op_type == MLILLogicalNot:
                return 1 if not operand else 0

            elif op_type == MLILBitwiseNot:
                return ~operand
        except OverflowError:
            return None
        return None

    def fold_constants(self) -> bool:
        '''Evaluate constant expressions (5 + 3 → 8)'''
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._fold_constants_in_inst(inst)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _fold_constants_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Fold constants in a single instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._fold_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._fold_expr(inst.condition)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._fold_expr(inst.value)
                if new_value is not inst.value:
                    return MLILRet(new_value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._fold_expr(arg) for arg in inst.args]
            if any(new_args[i] is not inst.args[i] for i in range(len(inst.args))):
                if isinstance(inst, MLILCall):
                    return MLILCall(inst.target, new_args)
                elif isinstance(inst, MLILSyscall):
                    return MLILSyscall(inst.subsystem, inst.cmd, new_args)
                elif isinstance(inst, MLILCallScript):
                    return MLILCallScript(inst.module, inst.func, new_args)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._fold_expr(inst.value)
            if new_value is not inst.value:
                if isinstance(inst, MLILStoreGlobal):
                    return MLILStoreGlobal(inst.index, new_value)
                else:
                    return MLILStoreReg(inst.index, new_value)

        return inst

    def _fold_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively fold constant expressions'''
        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                             MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                             MLILLogicalAnd, MLILLogicalOr,
                             MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._fold_expr(expr.lhs)
            rhs = self._fold_expr(expr.rhs)

            if isinstance(lhs, MLILConst) and isinstance(rhs, MLILConst):
                result = self._eval_binary_const(type(expr), lhs.value, rhs.value)
                if result is not None:
                    is_bitwise = isinstance(expr, (MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr))
                    return MLILConst(result, is_hex = is_bitwise)

            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILBitwiseNot)):
            operand = self._fold_expr(expr.operand)

            if isinstance(operand, MLILConst):
                result = self._eval_unary_const(type(expr), operand.value)
                if result is not None:
                    is_bitwise = isinstance(expr, MLILBitwiseNot)
                    return MLILConst(result, is_hex = is_bitwise)

            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def propagate_copies(self) -> bool:
        '''Replace SSA variable copies (x#1 = x#0 → use x#0)'''
        self._build_def_use_chains()
        changed = False

        copies: Dict[MLILVariableSSA, MLILVariableSSA] = {}
        for ssa_var, defn in self.ssa_defs.items():
            if isinstance(defn, MLILSetVarSSA):
                if isinstance(defn.value, MLILVarSSA):
                    copies[ssa_var] = defn.value.var

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._replace_copies_in_inst(inst, copies)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _replace_copies_in_inst(self, inst: MediumLevelILInstruction,
                                copies: Dict[MLILVariableSSA, MLILVariableSSA]) -> MediumLevelILInstruction:
        '''Replace copy variables in instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._replace_copies_in_expr(inst.value, copies)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._replace_copies_in_expr(inst.condition, copies)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._replace_copies_in_expr(inst.value, copies)
                if new_value is not inst.value:
                    return MLILRet(new_value)

        return inst

    def _replace_copies_in_expr(self, expr: MediumLevelILInstruction,
                                copies: Dict[MLILVariableSSA, MLILVariableSSA]) -> MediumLevelILInstruction:
        '''Replace copy variables in expression'''
        if isinstance(expr, MLILVarSSA):
            if expr.var in copies:
                return MLILVarSSA(copies[expr.var])

        elif isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                               MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr,
                               MLILLogicalAnd, MLILLogicalOr,
                               MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
            lhs = self._replace_copies_in_expr(expr.lhs, copies)
            rhs = self._replace_copies_in_expr(expr.rhs, copies)
            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILBitwiseNot)):
            operand = self._replace_copies_in_expr(expr.operand, copies)
            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def eliminate_redundant_reg_loads(self) -> bool:
        '''Replace redundant LoadReg with previously loaded value.

        Pattern: var1 = LoadReg(n); ...; var2 = LoadReg(n)
        If no StoreReg(n) between them, replace var2 uses with var1.
        '''
        self._build_def_use_chains()

        # Map: SSA var -> the SSA var it should be replaced with
        reg_copies: Dict[MLILVariableSSA, MLILVariableSSA] = {}

        # DFS traversal following control flow
        visited = set()

        def visit(block_idx: int, reg_state: Dict[int, MLILVariableSSA]):
            if block_idx in visited:
                return
            visited.add(block_idx)

            block = self.function.basic_blocks[block_idx]
            state = reg_state.copy()

            for inst in block.instructions:
                if isinstance(inst, MLILStoreReg):
                    # Register modified, invalidate
                    state.pop(inst.index, None)

                elif isinstance(inst, MLILSetVarSSA):
                    if isinstance(inst.value, MLILLoadReg):
                        reg_idx = inst.value.index
                        if reg_idx in state:
                            reg_copies[inst.var] = state[reg_idx]
                        else:
                            state[reg_idx] = inst.var

            # Visit successors
            last_inst = block.instructions[-1] if block.instructions else None
            if isinstance(last_inst, MLILIf):
                if last_inst.true_target:
                    visit(last_inst.true_target.index, state)
                if last_inst.false_target:
                    visit(last_inst.false_target.index, state)

            elif isinstance(last_inst, MLILGoto):
                if last_inst.target:
                    visit(last_inst.target.index, state)

            elif not isinstance(last_inst, MLILRet):
                if block_idx + 1 < len(self.function.basic_blocks):
                    visit(block_idx + 1, state)

        if self.function.basic_blocks:
            visit(0, {})

        if not reg_copies:
            return False

        # Replace uses of redundant loads
        changed = False
        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._replace_copies_in_inst(inst, reg_copies)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def eliminate_dead_code(self) -> bool:
        '''Remove SSA variable assignments that are never used'''
        self._build_def_use_chains()
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                if not isinstance(inst, (MLILSetVarSSA, MLILPhi)):
                    new_instructions.append(inst)
                    continue

                if isinstance(inst, MLILSetVarSSA):
                    uses = self.ssa_uses.get(inst.var, [])
                    if len(uses) > 0:
                        new_instructions.append(inst)
                    else:
                        # Preserve string constants as debug comments
                        if isinstance(inst.value, MLILConst) and isinstance(inst.value.value, str):
                            debug_comment = MLILDebug('string', inst.value.value)
                            new_instructions.append(debug_comment)

                        changed = True

                elif isinstance(inst, MLILPhi):
                    if len(self.ssa_uses.get(inst.dest, [])) > 0:
                        new_instructions.append(inst)
                    else:
                        changed = True

            block.instructions = new_instructions

        return changed

    def _reconstruct_binary_op(self, expr: MediumLevelILInstruction,
                               lhs: MediumLevelILInstruction,
                               rhs: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Reconstruct binary operation with new operands'''
        if isinstance(expr, MLILAdd):
            return MLILAdd(lhs, rhs)
        elif isinstance(expr, MLILSub):
            return MLILSub(lhs, rhs)
        elif isinstance(expr, MLILMul):
            return MLILMul(lhs, rhs)
        elif isinstance(expr, MLILDiv):
            return MLILDiv(lhs, rhs)
        elif isinstance(expr, MLILMod):
            return MLILMod(lhs, rhs)
        elif isinstance(expr, MLILAnd):
            return MLILAnd(lhs, rhs)
        elif isinstance(expr, MLILOr):
            return MLILOr(lhs, rhs)
        elif isinstance(expr, MLILXor):
            return MLILXor(lhs, rhs)
        elif isinstance(expr, MLILShl):
            return MLILShl(lhs, rhs)
        elif isinstance(expr, MLILShr):
            return MLILShr(lhs, rhs)
        elif isinstance(expr, MLILLogicalAnd):
            return MLILLogicalAnd(lhs, rhs)
        elif isinstance(expr, MLILLogicalOr):
            return MLILLogicalOr(lhs, rhs)
        elif isinstance(expr, MLILEq):
            return MLILEq(lhs, rhs)
        elif isinstance(expr, MLILNe):
            return MLILNe(lhs, rhs)
        elif isinstance(expr, MLILLt):
            return MLILLt(lhs, rhs)
        elif isinstance(expr, MLILLe):
            return MLILLe(lhs, rhs)
        elif isinstance(expr, MLILGt):
            return MLILGt(lhs, rhs)
        elif isinstance(expr, MLILGe):
            return MLILGe(lhs, rhs)
        else:
            raise NotImplementedError(f'Unhandled binary operation: {type(expr).__name__}')

    def _reconstruct_unary_op(self, expr: MediumLevelILInstruction,
                              operand: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Reconstruct unary operation with new operand'''
        if isinstance(expr, MLILNeg):
            return MLILNeg(operand)

        elif isinstance(expr, MLILLogicalNot):
            return MLILLogicalNot(operand)

        elif isinstance(expr, MLILBitwiseNot):
            return MLILBitwiseNot(operand)

        else:
            raise NotImplementedError(f'Unhandled unary operation: {type(expr).__name__}')

    def simplify_expressions(self) -> bool:
        '''Apply algebraic simplification (x + 0 → x, x * 1 → x)'''
        changed = False

        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._simplify_expr_in_inst(inst)
                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _simplify_expr_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Simplify expressions in a single instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._simplify_expr(inst.value)
            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_condition = self._simplify_expr(inst.condition)
            if new_condition is not inst.condition:
                return MLILIf(new_condition, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value is not None:
                new_value = self._simplify_expr(inst.value)
                if new_value is not inst.value:
                    return MLILRet(new_value)

        return inst

    def _simplify_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively apply algebraic simplifications'''
        if isinstance(expr, (MLILAdd, MLILSub, MLILMul, MLILDiv, MLILMod,
                             MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr)):
            lhs = self._simplify_expr(expr.lhs)
            rhs = self._simplify_expr(expr.rhs)

            simplified = self._apply_algebraic_identity(type(expr), lhs, rhs)
            if simplified is not None:
                return simplified

            if lhs is not expr.lhs or rhs is not expr.rhs:
                return self._reconstruct_binary_op(expr, lhs, rhs)

        elif isinstance(expr, (MLILNeg, MLILLogicalNot, MLILBitwiseNot)):
            operand = self._simplify_expr(expr.operand)
            if operand is not expr.operand:
                return self._reconstruct_unary_op(expr, operand)

        return expr

    def _apply_algebraic_identity(self, op_type, lhs, rhs) -> Optional[MediumLevelILInstruction]:
        '''Apply algebraic identity rules'''
        if op_type == MLILAdd:
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs
            if isinstance(lhs, MLILConst) and lhs.value == 0:
                return rhs

        elif op_type == MLILSub:
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs

        elif op_type == MLILMul:
            if isinstance(rhs, MLILConst):
                if rhs.value == 0:
                    return MLILConst(0, is_hex = False)
                elif rhs.value == 1:
                    return lhs
            if isinstance(lhs, MLILConst):
                if lhs.value == 0:
                    return MLILConst(0, is_hex = False)
                elif lhs.value == 1:
                    return rhs

        elif op_type == MLILDiv:
            if isinstance(rhs, MLILConst) and rhs.value == 1:
                return lhs

        elif op_type == MLILAnd:
            if isinstance(rhs, MLILConst):
                if rhs.value == 0:
                    return MLILConst(0, is_hex = False)
                elif rhs.value == 0xFFFFFFFF:
                    return lhs
            if isinstance(lhs, MLILConst):
                if lhs.value == 0:
                    return MLILConst(0, is_hex = False)
                elif lhs.value == 0xFFFFFFFF:
                    return rhs

        elif op_type == MLILOr:
            if isinstance(rhs, MLILConst):
                if rhs.value == 0:
                    return lhs
                elif rhs.value == 0xFFFFFFFF:
                    return MLILConst(0xFFFFFFFF, is_hex = True)
            if isinstance(lhs, MLILConst):
                if lhs.value == 0:
                    return rhs
                elif lhs.value == 0xFFFFFFFF:
                    return MLILConst(0xFFFFFFFF, is_hex = True)

        elif op_type == MLILXor:
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs
            if isinstance(lhs, MLILConst) and lhs.value == 0:
                return rhs

        elif op_type in (MLILShl, MLILShr):
            if isinstance(rhs, MLILConst) and rhs.value == 0:
                return lhs

        return None

    def simplify_conditions(self) -> bool:
        '''Simplify conditionals: (expr == 0) → !expr, (expr != 0) → expr'''
        changed = False

        for block in self.function.basic_blocks:
            for i, inst in enumerate(block.instructions):
                if not isinstance(inst, MLILIf):
                    continue

                condition = inst.condition
                new_condition = None

                if isinstance(condition, MLILLogicalNot):
                    inverted = self._invert_comparison(condition.operand)
                    if inverted is not None:
                        new_condition = inverted

                elif isinstance(condition, MLILEq):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        inverted = self._invert_comparison(condition.lhs)
                        if inverted is not None:
                            new_condition = inverted
                        else:
                            new_condition = MLILLogicalNot(condition.lhs)

                elif isinstance(condition, MLILNe):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        if isinstance(condition.lhs, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
                            new_condition = condition.lhs

                if new_condition is not None:
                    block.instructions[i] = MLILIf(new_condition, inst.true_target, inst.false_target)
                    changed = True

        return changed

    def _invert_comparison(self, expr: MediumLevelILInstruction) -> Optional[MediumLevelILInstruction]:
        '''Invert comparison: >= → <, > → <=, == → !=, etc.'''
        if isinstance(expr, MLILGe):
            return MLILLt(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILGt):
            return MLILLe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILLt):
            return MLILGe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILLe):
            return MLILGt(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILEq):
            return MLILNe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILNe):
            return MLILEq(expr.lhs, expr.rhs)

        else:
            return None


# ============================================================================
# SCCP - Sparse Conditional Constant Propagation
# ============================================================================

class LatticeValue:
    '''Lattice value for SCCP: TOP → CONSTANT → BOTTOM'''
    TOP = 'top'        # Undefined/unknown
    BOTTOM = 'bottom'  # Multiple possible values

    def __init__(self, state: str = TOP, value: any = None):
        self.state = state
        self.value = value  # Only valid when state == 'constant'

    @classmethod
    def top(cls) -> 'LatticeValue':
        return cls(cls.TOP)

    @classmethod
    def constant(cls, value: any) -> 'LatticeValue':
        return cls('constant', value)

    @classmethod
    def bottom(cls) -> 'LatticeValue':
        return cls(cls.BOTTOM)

    def is_top(self) -> bool:
        return self.state == self.TOP

    def is_constant(self) -> bool:
        return self.state == 'constant'

    def is_bottom(self) -> bool:
        return self.state == self.BOTTOM

    def meet(self, other: 'LatticeValue') -> 'LatticeValue':
        '''Meet operation: combine two lattice values'''
        # TOP meet x = x
        if self.is_top():
            return other

        if other.is_top():
            return self

        # BOTTOM meet x = BOTTOM
        if self.is_bottom() or other.is_bottom():
            return LatticeValue.bottom()

        # CONSTANT meet CONSTANT
        if self.value == other.value:
            return self

        return LatticeValue.bottom()

    def __eq__(self, other) -> bool:
        if not isinstance(other, LatticeValue):
            return False
        if self.state != other.state:
            return False
        if self.is_constant():
            return self.value == other.value
        return True

    def __repr__(self) -> str:
        if self.is_top():
            return '⊤'

        elif self.is_bottom():
            return '⊥'

        else:
            return f'C({self.value})'


class SCCP:
    '''Sparse Conditional Constant Propagation'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function

        # Lattice values for SSA variables
        self.var_values: Dict[MLILVariableSSA, LatticeValue] = {}

        # Edge reachability: (from_block, to_block) -> reachable
        self.edge_reachable: Dict[tuple, bool] = {}

        # Block reachability
        self.block_reachable: Dict[MediumLevelILBasicBlock, bool] = {}

        # Worklists
        self.ssa_worklist: List[MLILVariableSSA] = []
        self.cfg_worklist: List[tuple] = []  # (from_block, to_block)

        # SSA def-use chains
        self.ssa_defs: Dict[MLILVariableSSA, MediumLevelILInstruction] = {}
        self.ssa_uses: Dict[MLILVariableSSA, List[MediumLevelILInstruction]] = {}

        # Block containing each instruction
        self.inst_block: Dict[MediumLevelILInstruction, MediumLevelILBasicBlock] = {}

    def run(self) -> bool:
        '''Run SCCP and return True if any changes were made'''
        self._initialize()
        self._propagate()
        return self._apply_results()

    def _initialize(self):
        '''Initialize SCCP state'''
        # Build def-use chains and instruction-to-block mapping
        for block in self.function.basic_blocks:
            self.block_reachable[block] = False

            for inst in block.instructions:
                self.inst_block[inst] = block

                if isinstance(inst, MLILSetVarSSA):
                    self.ssa_defs[inst.var] = inst
                    self._collect_uses(inst.value, inst)

                elif isinstance(inst, MLILPhi):
                    self.ssa_defs[inst.dest] = inst
                    for src_var, _ in inst.sources:
                        if src_var not in self.ssa_uses:
                            self.ssa_uses[src_var] = []
                        self.ssa_uses[src_var].append(inst)

                else:
                    self._collect_uses_in_stmt(inst, inst)

        # Initialize all variables to TOP
        for var in self.ssa_defs:
            self.var_values[var] = LatticeValue.top()

        # Initialize all edges to unreachable
        for block in self.function.basic_blocks:
            for succ in block.outgoing_edges:
                self.edge_reachable[(block, succ)] = False

        # Mark entry block as reachable
        if self.function.basic_blocks:
            entry = self.function.basic_blocks[0]
            self.block_reachable[entry] = True
            # Add entry block's instructions to worklist
            self._process_block(entry)

    def _collect_uses(self, expr: MediumLevelILInstruction, user: MediumLevelILInstruction):
        '''Collect variable uses in expression'''
        if isinstance(expr, MLILVarSSA):
            if expr.var not in self.ssa_uses:
                self.ssa_uses[expr.var] = []
            self.ssa_uses[expr.var].append(user)

        elif isinstance(expr, MLILBinaryOp):
            self._collect_uses(expr.lhs, user)
            self._collect_uses(expr.rhs, user)

        elif isinstance(expr, MLILUnaryOp):
            self._collect_uses(expr.operand, user)

    def _collect_uses_in_stmt(self, stmt: MediumLevelILInstruction, user: MediumLevelILInstruction):
        '''Collect variable uses in statement'''
        if isinstance(stmt, MLILIf):
            self._collect_uses(stmt.condition, user)

        elif isinstance(stmt, MLILRet):
            if stmt.value:
                self._collect_uses(stmt.value, user)

        elif isinstance(stmt, (MLILCall, MLILSyscall, MLILCallScript)):
            for arg in stmt.args:
                self._collect_uses(arg, user)

        elif isinstance(stmt, (MLILStoreGlobal, MLILStoreReg)):
            self._collect_uses(stmt.value, user)

    def _propagate(self):
        '''Main SCCP propagation loop'''
        while self.ssa_worklist or self.cfg_worklist:
            # Process CFG edges first
            while self.cfg_worklist:
                from_block, to_block = self.cfg_worklist.pop(0)

                if self.edge_reachable.get((from_block, to_block), False):
                    continue

                self.edge_reachable[(from_block, to_block)] = True

                # Check if this is first time block becomes reachable
                was_reachable = self.block_reachable.get(to_block, False)
                self.block_reachable[to_block] = True

                if not was_reachable:
                    # First time visiting this block
                    self._process_block(to_block)

                else:
                    # Block already reachable, just update Phis
                    self._update_phis(to_block)

            # Process SSA worklist
            while self.ssa_worklist:
                var = self.ssa_worklist.pop(0)

                for user in self.ssa_uses.get(var, []):
                    user_block = self.inst_block.get(user)

                    if user_block and self.block_reachable.get(user_block, False):
                        self._visit_instruction(user)

    def _process_block(self, block: MediumLevelILBasicBlock):
        '''Process all instructions in a newly reachable block'''
        for inst in block.instructions:
            self._visit_instruction(inst)

    def _update_phis(self, block: MediumLevelILBasicBlock):
        '''Update Phi nodes when a new edge becomes reachable'''
        for inst in block.instructions:
            if isinstance(inst, MLILPhi):
                self._visit_phi(inst)

    def _visit_instruction(self, inst: MediumLevelILInstruction):
        '''Visit and evaluate an instruction'''
        if isinstance(inst, MLILSetVarSSA):
            self._visit_assignment(inst)

        elif isinstance(inst, MLILPhi):
            self._visit_phi(inst)

        elif isinstance(inst, MLILIf):
            self._visit_branch(inst)

        elif isinstance(inst, MLILGoto):
            self._visit_goto(inst)

        elif isinstance(inst, MLILRet):
            pass  # No successors

        else:
            # Other instructions: mark all successors reachable
            block = self.inst_block.get(inst)
            if block:
                for succ in block.outgoing_edges:
                    self._mark_edge_reachable(block, succ)

    def _visit_assignment(self, inst: MLILSetVarSSA):
        '''Evaluate assignment instruction'''
        new_value = self._evaluate_expr(inst.value)
        self._update_var_value(inst.var, new_value)

    def _visit_phi(self, inst: MLILPhi):
        '''Evaluate Phi node'''
        result = LatticeValue.top()
        block = self.inst_block.get(inst)

        for src_var, pred_block in inst.sources:
            # Only consider reachable edges
            if self.edge_reachable.get((pred_block, block), False):
                # If source variable has no definition, it's uninitialized (BOTTOM)
                if src_var not in self.ssa_defs:
                    src_value = LatticeValue.bottom()
                else:
                    src_value = self.var_values.get(src_var, LatticeValue.top())
                result = result.meet(src_value)

        self._update_var_value(inst.dest, result)

    def _visit_branch(self, inst: MLILIf):
        '''Evaluate conditional branch'''
        block = self.inst_block.get(inst)
        if not block:
            return

        cond_value = self._evaluate_expr(inst.condition)

        if cond_value.is_constant():
            # Condition is constant, only one branch is reachable
            if cond_value.value:
                self._mark_edge_reachable(block, inst.true_target)

            else:
                self._mark_edge_reachable(block, inst.false_target)

        else:
            # Condition unknown, both branches reachable
            self._mark_edge_reachable(block, inst.true_target)
            self._mark_edge_reachable(block, inst.false_target)

    def _visit_goto(self, inst: MLILGoto):
        '''Evaluate unconditional jump'''
        block = self.inst_block.get(inst)
        if block and inst.target:
            self._mark_edge_reachable(block, inst.target)

    def _evaluate_expr(self, expr: MediumLevelILInstruction) -> LatticeValue:
        '''Evaluate expression to lattice value'''
        if isinstance(expr, MLILConst):
            return LatticeValue.constant(expr.value)

        elif isinstance(expr, MLILVarSSA):
            # If variable has no definition, it's uninitialized - could be any value (BOTTOM)
            # This prevents incorrect constant propagation through Phi nodes
            if expr.var not in self.ssa_defs:
                return LatticeValue.bottom()
            return self.var_values.get(expr.var, LatticeValue.top())

        elif isinstance(expr, MLILBinaryOp):
            lhs = self._evaluate_expr(expr.lhs)
            rhs = self._evaluate_expr(expr.rhs)

            # If either is BOTTOM, result is BOTTOM
            if lhs.is_bottom() or rhs.is_bottom():
                return LatticeValue.bottom()

            # If either is TOP, result is TOP
            if lhs.is_top() or rhs.is_top():
                return LatticeValue.top()

            # Both constants, evaluate
            return self._eval_binary_op(type(expr), lhs.value, rhs.value)

        elif isinstance(expr, MLILUnaryOp):
            operand = self._evaluate_expr(expr.operand)

            if operand.is_bottom():
                return LatticeValue.bottom()

            if operand.is_top():
                return LatticeValue.top()

            return self._eval_unary_op(type(expr), operand.value)

        else:
            # Unknown expression type
            return LatticeValue.bottom()

    def _eval_binary_op(self, op_type, lhs, rhs) -> LatticeValue:
        '''Evaluate binary operation on constants'''
        try:
            if op_type == MLILAdd:
                return LatticeValue.constant(lhs + rhs)

            elif op_type == MLILSub:
                return LatticeValue.constant(lhs - rhs)

            elif op_type == MLILMul:
                return LatticeValue.constant(lhs * rhs)

            elif op_type == MLILDiv:
                if rhs == 0:
                    return LatticeValue.bottom()
                return LatticeValue.constant(lhs // rhs)

            elif op_type == MLILMod:
                if rhs == 0:
                    return LatticeValue.bottom()
                return LatticeValue.constant(lhs % rhs)

            elif op_type == MLILAnd:
                return LatticeValue.constant(lhs & rhs)

            elif op_type == MLILOr:
                return LatticeValue.constant(lhs | rhs)

            elif op_type == MLILXor:
                return LatticeValue.constant(lhs ^ rhs)

            elif op_type == MLILShl:
                return LatticeValue.constant(lhs << rhs)

            elif op_type == MLILShr:
                return LatticeValue.constant(lhs >> rhs)

            elif op_type == MLILLogicalAnd:
                return LatticeValue.constant(1 if (lhs and rhs) else 0)

            elif op_type == MLILLogicalOr:
                return LatticeValue.constant(1 if (lhs or rhs) else 0)

            elif op_type == MLILEq:
                return LatticeValue.constant(1 if lhs == rhs else 0)

            elif op_type == MLILNe:
                return LatticeValue.constant(1 if lhs != rhs else 0)

            elif op_type == MLILLt:
                return LatticeValue.constant(1 if lhs < rhs else 0)

            elif op_type == MLILLe:
                return LatticeValue.constant(1 if lhs <= rhs else 0)

            elif op_type == MLILGt:
                return LatticeValue.constant(1 if lhs > rhs else 0)

            elif op_type == MLILGe:
                return LatticeValue.constant(1 if lhs >= rhs else 0)

        except (OverflowError, TypeError):
            pass

        return LatticeValue.bottom()

    def _eval_unary_op(self, op_type, operand) -> LatticeValue:
        '''Evaluate unary operation on constant'''
        try:
            if op_type == MLILNeg:
                return LatticeValue.constant(-operand)

            elif op_type == MLILLogicalNot:
                return LatticeValue.constant(1 if not operand else 0)

            elif op_type == MLILBitwiseNot:
                return LatticeValue.constant(~operand)

            elif op_type == MLILTestZero:
                return LatticeValue.constant(1 if operand == 0 else 0)

        except (OverflowError, TypeError):
            pass

        return LatticeValue.bottom()

    def _update_var_value(self, var: MLILVariableSSA, new_value: LatticeValue):
        '''Update variable value and add to worklist if changed'''
        old_value = self.var_values.get(var, LatticeValue.top())

        if new_value != old_value:
            self.var_values[var] = new_value
            self.ssa_worklist.append(var)

    def _mark_edge_reachable(self, from_block: MediumLevelILBasicBlock,
                              to_block: MediumLevelILBasicBlock):
        '''Mark CFG edge as reachable'''
        if not self.edge_reachable.get((from_block, to_block), False):
            self.cfg_worklist.append((from_block, to_block))

    def _apply_results(self) -> bool:
        '''Apply SCCP results: remove unreachable code, replace constants'''
        changed = False

        # Remove unreachable blocks
        reachable_blocks = [b for b in self.function.basic_blocks
                           if self.block_reachable.get(b, False)]

        if len(reachable_blocks) < len(self.function.basic_blocks):
            self.function.basic_blocks = reachable_blocks
            changed = True

        # Remove definitions that are only used by unreachable code
        # This is handled by DCE after SCCP

        # Replace constant variables
        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                new_inst = self._replace_constants_in_inst(inst)

                # Remove dead assignments (var = const where var is never used)
                if isinstance(new_inst, MLILSetVarSSA):
                    val = self.var_values.get(new_inst.var, LatticeValue.bottom())

                    if val.is_constant():
                        # Keep the assignment but it might be eliminated by DCE
                        pass

                new_instructions.append(new_inst)

                if new_inst is not inst:
                    changed = True

            block.instructions = new_instructions

        return changed

    def _replace_constants_in_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Replace constant SSA variables in instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._replace_constants_in_expr(inst.value)

            if new_value is not inst.value:
                return MLILSetVarSSA(inst.var, new_value)

        elif isinstance(inst, MLILIf):
            new_cond = self._replace_constants_in_expr(inst.condition)

            if new_cond is not inst.condition:
                return MLILIf(new_cond, inst.true_target, inst.false_target)

        elif isinstance(inst, MLILRet):
            if inst.value:
                new_value = self._replace_constants_in_expr(inst.value)

                if new_value is not inst.value:
                    return MLILRet(new_value)

        elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._replace_constants_in_expr(arg) for arg in inst.args]

            if any(new_args[i] is not inst.args[i] for i in range(len(inst.args))):
                if isinstance(inst, MLILCall):
                    return MLILCall(inst.target, new_args)

                elif isinstance(inst, MLILSyscall):
                    return MLILSyscall(inst.subsystem, inst.cmd, new_args)

                elif isinstance(inst, MLILCallScript):
                    return MLILCallScript(inst.module, inst.func, new_args)

        elif isinstance(inst, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._replace_constants_in_expr(inst.value)

            if new_value is not inst.value:
                if isinstance(inst, MLILStoreGlobal):
                    return MLILStoreGlobal(inst.index, new_value)

                else:
                    return MLILStoreReg(inst.index, new_value)

        return inst

    def _replace_constants_in_expr(self, expr: MediumLevelILInstruction, is_bitwise: bool = False) -> MediumLevelILInstruction:
        '''Replace constant SSA variables with constants'''
        if isinstance(expr, MLILVarSSA):
            val = self.var_values.get(expr.var, LatticeValue.bottom())

            if val.is_constant():
                return MLILConst(val.value, is_hex=is_bitwise)

            return expr

        elif isinstance(expr, MLILConst):
            # Update existing const to hex format if in bitwise context
            if is_bitwise and not expr.is_hex:
                return MLILConst(expr.value, is_hex=True)
            return expr

        elif isinstance(expr, MLILBinaryOp):
            # Bitwise operations use hex format for constants
            child_is_bitwise = isinstance(expr, (MLILAnd, MLILOr, MLILXor, MLILShl, MLILShr))
            new_lhs = self._replace_constants_in_expr(expr.lhs, child_is_bitwise)
            new_rhs = self._replace_constants_in_expr(expr.rhs, child_is_bitwise)

            if new_lhs is not expr.lhs or new_rhs is not expr.rhs:
                return self._rebuild_binary_op(expr, new_lhs, new_rhs)

        elif isinstance(expr, MLILUnaryOp):
            child_is_bitwise = isinstance(expr, MLILBitwiseNot)
            new_operand = self._replace_constants_in_expr(expr.operand, child_is_bitwise)

            if new_operand is not expr.operand:
                return self._rebuild_unary_op(expr, new_operand)

        return expr

    def _rebuild_binary_op(self, expr, lhs, rhs) -> MediumLevelILInstruction:
        '''Rebuild binary operation with new operands'''
        if isinstance(expr, MLILAdd):
            return MLILAdd(lhs, rhs)

        elif isinstance(expr, MLILSub):
            return MLILSub(lhs, rhs)

        elif isinstance(expr, MLILMul):
            return MLILMul(lhs, rhs)

        elif isinstance(expr, MLILDiv):
            return MLILDiv(lhs, rhs)

        elif isinstance(expr, MLILMod):
            return MLILMod(lhs, rhs)

        elif isinstance(expr, MLILAnd):
            return MLILAnd(lhs, rhs)

        elif isinstance(expr, MLILOr):
            return MLILOr(lhs, rhs)

        elif isinstance(expr, MLILXor):
            return MLILXor(lhs, rhs)

        elif isinstance(expr, MLILShl):
            return MLILShl(lhs, rhs)

        elif isinstance(expr, MLILShr):
            return MLILShr(lhs, rhs)

        elif isinstance(expr, MLILLogicalAnd):
            return MLILLogicalAnd(lhs, rhs)

        elif isinstance(expr, MLILLogicalOr):
            return MLILLogicalOr(lhs, rhs)

        elif isinstance(expr, MLILEq):
            return MLILEq(lhs, rhs)

        elif isinstance(expr, MLILNe):
            return MLILNe(lhs, rhs)

        elif isinstance(expr, MLILLt):
            return MLILLt(lhs, rhs)

        elif isinstance(expr, MLILLe):
            return MLILLe(lhs, rhs)

        elif isinstance(expr, MLILGt):
            return MLILGt(lhs, rhs)

        elif isinstance(expr, MLILGe):
            return MLILGe(lhs, rhs)

        else:
            return expr

    def _rebuild_unary_op(self, expr, operand) -> MediumLevelILInstruction:
        '''Rebuild unary operation with new operand'''
        if isinstance(expr, MLILNeg):
            return MLILNeg(operand)

        elif isinstance(expr, MLILLogicalNot):
            return MLILLogicalNot(operand)

        elif isinstance(expr, MLILBitwiseNot):
            return MLILBitwiseNot(operand)

        elif isinstance(expr, MLILTestZero):
            return MLILTestZero(operand)

        else:
            return expr


def run_sccp(function: MediumLevelILFunction) -> bool:
    '''Run SCCP on function'''
    sccp = SCCP(function)
    return sccp.run()


# ============================================================================
# Dead Phi Source Elimination
# ============================================================================

class DeadPhiSourceEliminator:
    '''Eliminate dead Phi nodes and their source definitions.

    A Phi node is "dead" if its result is never used except by other dead Phi nodes.
    This eliminates entire chains of Phi nodes that contribute to nothing.
    '''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.ssa_defs: Dict[MLILVariableSSA, MediumLevelILInstruction] = {}
        self.ssa_uses: Dict[MLILVariableSSA, List[MediumLevelILInstruction]] = {}
        self.inst_block: Dict[MediumLevelILInstruction, MediumLevelILBasicBlock] = {}
        self.block_index: Dict[MediumLevelILBasicBlock, int] = {}

    def run(self) -> bool:
        '''Run dead phi elimination'''
        self._build_info()
        return self._eliminate_dead_phis()

    def _build_info(self):
        '''Build def-use chains and mappings'''
        for idx, block in enumerate(self.function.basic_blocks):
            self.block_index[block] = idx

            for inst in block.instructions:
                self.inst_block[inst] = block

                if isinstance(inst, MLILSetVarSSA):
                    self.ssa_defs[inst.var] = inst
                    self._collect_uses(inst.value, inst)

                elif isinstance(inst, MLILPhi):
                    self.ssa_defs[inst.dest] = inst
                    for src_var, _ in inst.sources:
                        if src_var not in self.ssa_uses:
                            self.ssa_uses[src_var] = []
                        self.ssa_uses[src_var].append(inst)

                else:
                    self._collect_uses_in_stmt(inst, inst)

    def _collect_uses(self, expr, user):
        '''Collect variable uses in expression'''
        if isinstance(expr, MLILVarSSA):
            if expr.var not in self.ssa_uses:
                self.ssa_uses[expr.var] = []
            self.ssa_uses[expr.var].append(user)

        elif isinstance(expr, MLILBinaryOp):
            self._collect_uses(expr.lhs, user)
            self._collect_uses(expr.rhs, user)

        elif isinstance(expr, MLILUnaryOp):
            self._collect_uses(expr.operand, user)

    def _collect_uses_in_stmt(self, stmt, user):
        '''Collect variable uses in statement'''
        if isinstance(stmt, MLILIf):
            self._collect_uses(stmt.condition, user)

        elif isinstance(stmt, MLILRet):
            if stmt.value:
                self._collect_uses(stmt.value, user)

        elif isinstance(stmt, (MLILCall, MLILSyscall, MLILCallScript)):
            for arg in stmt.args:
                self._collect_uses(arg, user)

        elif isinstance(stmt, (MLILStoreGlobal, MLILStoreReg)):
            self._collect_uses(stmt.value, user)

    def _eliminate_dead_phis(self) -> bool:
        '''Eliminate dead Phi nodes and definitions feeding them'''
        changed = False

        # Find all Phi nodes with no real use
        dead_phis: Set[MLILPhi] = set()
        for block in self.function.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILPhi):
                    if not self._has_real_use(inst.dest):
                        dead_phis.add(inst)

        if not dead_phis:
            return False

        # Collect all variables defined by dead Phis
        dead_phi_vars = {phi.dest for phi in dead_phis}

        # Find definitions that only feed dead Phis
        dead_defs: Set[MLILVariableSSA] = set()
        for var, uses in self.ssa_uses.items():
            if not uses:
                continue

            # Check if all uses are dead Phi nodes
            if all(isinstance(u, MLILPhi) and u in dead_phis for u in uses):
                # This variable only feeds dead Phis, so it's dead too
                dead_defs.add(var)

        # Remove dead Phi nodes and dead definitions
        for block in self.function.basic_blocks:
            new_instructions = []

            for inst in block.instructions:
                # Skip dead Phi nodes
                if isinstance(inst, MLILPhi) and inst in dead_phis:
                    changed = True
                    continue

                # Skip definitions that only feed dead Phis
                if isinstance(inst, MLILSetVarSSA) and inst.var in dead_defs:
                    # Preserve string constants as debug comments
                    if isinstance(inst.value, MLILConst) and isinstance(inst.value.value, str):
                        debug_comment = MLILDebug('string', inst.value.value)
                        new_instructions.append(debug_comment)

                    changed = True
                    continue

                new_instructions.append(inst)

            block.instructions = new_instructions

        return changed

    def _has_real_use(self, start_var: MLILVariableSSA) -> bool:
        '''Check if variable has any real (non-Phi) use through Phi chains'''
        visited = set()
        queue = [start_var]

        while queue:
            var = queue.pop(0)
            if var in visited:
                continue
            visited.add(var)

            for use in self.ssa_uses.get(var, []):
                if isinstance(use, MLILPhi):
                    # Follow Phi chain
                    queue.append(use.dest)

                else:
                    # Found a real use
                    return True

        return False


def eliminate_dead_phi_sources(function: MediumLevelILFunction) -> bool:
    '''Eliminate dead phi sources'''
    eliminator = DeadPhiSourceEliminator(function)
    return eliminator.run()

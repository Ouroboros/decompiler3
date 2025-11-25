'''MLIL SSA - SSA form with dominance analysis and Phi placement'''

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Deque
from collections import deque, defaultdict
from .mlil import *


# ============================================================================
# SSA Types
# ============================================================================

class MLILVariableSSA:
    '''SSA-versioned variable (e.g., var_s0#1, var_s0#2)'''

    def __init__(self, base_var: MLILVariable, version: int):
        self.base_var = base_var
        self.version = version

    @property
    def name(self) -> str:
        return f'{self.base_var.name}#{self.version}'

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'SSA({self.base_var.name}#{self.version})'

    def __eq__(self, other) -> bool:
        return (isinstance(other, MLILVariableSSA) and
                self.base_var == other.base_var and
                self.version == other.version)

    def __hash__(self) -> int:
        return hash((self.base_var, self.version))


class MLILVarSSA(MediumLevelILExpr):
    '''Load SSA variable value'''

    def __init__(self, var: MLILVariableSSA):
        super().__init__(MediumLevelILOperation.MLIL_VAR_SSA)
        self.var = var

    def __str__(self) -> str:
        return str(self.var)


class MLILSetVarSSA(MediumLevelILStatement):
    '''Assign to SSA variable'''

    def __init__(self, var: MLILVariableSSA, value: MediumLevelILInstruction):
        super().__init__(MediumLevelILOperation.MLIL_SET_VAR_SSA)
        self.var = var
        self.value = value

    def __str__(self) -> str:
        return f'{self.var} = {self.value}'


class MLILPhi(MediumLevelILStatement):
    '''Phi node: merge values from predecessors'''

    def __init__(self, dest: MLILVariableSSA, sources: List[Tuple[MLILVariableSSA, MediumLevelILBasicBlock]]):
        super().__init__(MediumLevelILOperation.MLIL_PHI)
        self.dest = dest
        self.sources = sources  # [(ssa_var, predecessor_block), ...]

    def __str__(self) -> str:
        if not self.sources:
            return f'{self.dest} = φ()'

        src_strs = ', '.join(f'{var} from {block.label}' for var, block in self.sources)
        return f'{self.dest} = φ({src_strs})'


# ============================================================================
# Dominance Analysis (Iterative)
# ============================================================================

class DominanceAnalysis:
    '''Compute dominance tree and dominance frontiers (iterative)'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.blocks = self._compute_rpo(function.basic_blocks)

        # Results
        self.idom: Dict[MediumLevelILBasicBlock, Optional[MediumLevelILBasicBlock]] = {}
        self.dom_tree: Dict[MediumLevelILBasicBlock, List[MediumLevelILBasicBlock]] = defaultdict(list)
        self.dom_frontier: Dict[MediumLevelILBasicBlock, Set[MediumLevelILBasicBlock]] = defaultdict(set)

    def _compute_rpo(self, blocks: List[MediumLevelILBasicBlock]) -> List[MediumLevelILBasicBlock]:
        '''Compute Reverse Postorder via DFS (only reachable blocks)'''
        if not blocks:
            return []

        visited = set()
        postorder = []
        stack = [(blocks[0], False)]

        while stack:
            block, processed = stack.pop()

            if processed:
                postorder.append(block)
                continue

            if block in visited:
                continue

            visited.add(block)
            stack.append((block, True))

            for succ in reversed(block.outgoing_edges):
                if succ not in visited:
                    stack.append((succ, False))

        return list(reversed(postorder))

    def analyze(self):
        '''Run dominance analysis'''
        self._compute_dominators()
        self._build_dom_tree()
        self._compute_dom_frontiers()

    def _compute_dominators(self):
        '''Compute immediate dominators (iterative dataflow)'''
        if not self.blocks:
            return

        entry = self.blocks[0]

        # Initialize: entry dominates itself, others dominated by all
        self.idom[entry] = entry
        for block in self.blocks[1:]:
            self.idom[block] = None

        # Build predecessor map
        preds: Dict[MediumLevelILBasicBlock, List[MediumLevelILBasicBlock]] = defaultdict(list)
        for block in self.blocks:
            for succ in block.outgoing_edges:
                preds[succ].append(block)

        # Iterate until convergence
        changed = True
        while changed:
            changed = False

            for block in self.blocks[1:]:  # Skip entry
                if not preds[block]:
                    continue

                # New idom = intersect of all processed predecessors
                new_idom = None
                for pred in preds[block]:
                    if self.idom[pred] is not None:
                        if new_idom is None:
                            new_idom = pred

                        else:
                            new_idom = self._intersect(pred, new_idom)

                if new_idom != self.idom[block]:
                    self.idom[block] = new_idom
                    changed = True

    def _intersect(self, b1: MediumLevelILBasicBlock, b2: MediumLevelILBasicBlock) -> MediumLevelILBasicBlock:
        '''Find common dominator of b1 and b2'''
        finger1 = b1
        finger2 = b2

        while finger1 != finger2:
            while self.blocks.index(finger1) > self.blocks.index(finger2):
                finger1 = self.idom[finger1]

            while self.blocks.index(finger2) > self.blocks.index(finger1):
                finger2 = self.idom[finger2]

        return finger1

    def _build_dom_tree(self):
        '''Build dominator tree from immediate dominators'''
        for block, dominator in self.idom.items():
            if dominator is not None and dominator != block:
                self.dom_tree[dominator].append(block)

    def _compute_dom_frontiers(self):
        '''Compute dominance frontiers (iterative)'''
        # Build predecessor map
        preds: Dict[MediumLevelILBasicBlock, List[MediumLevelILBasicBlock]] = defaultdict(list)
        for block in self.blocks:
            for succ in block.outgoing_edges:
                preds[succ].append(block)

        for block in self.blocks:
            if len(preds[block]) < 2:
                continue

            for pred in preds[block]:
                runner = pred

                while runner != self.idom.get(block):
                    self.dom_frontier[runner].add(block)
                    runner = self.idom.get(runner)

                    if runner is None:
                        break


# ============================================================================
# SSA Construction (Iterative)
# ============================================================================

class SSAConstructor:
    '''Convert MLIL to SSA form (iterative algorithms)'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function
        self.dom_analysis = DominanceAnalysis(function)

        # Variable tracking
        self.var_defs: Dict[MLILVariable, Set[MediumLevelILBasicBlock]] = defaultdict(set)
        self.var_versions: Dict[MLILVariable, int] = {}
        self.var_stack: Dict[MLILVariable, List[int]] = defaultdict(list)

    def construct(self) -> MediumLevelILFunction:
        '''Convert function to SSA form (modifies in-place)'''
        # Step 1: Dominance analysis
        self.dom_analysis.analyze()

        # Remove unreachable blocks from function
        reachable_set = set(self.dom_analysis.blocks)
        self.function.basic_blocks = [b for b in self.function.basic_blocks if b in reachable_set]

        # Step 2: Collect variable definitions
        self._collect_defs()

        # Step 3: Insert Phi nodes
        self._insert_phi_nodes()

        # Step 4: Rename variables (iterative)
        self._rename_variables_iterative()

        return self.function

    def _collect_defs(self):
        '''Collect blocks where each variable is defined'''
        for block in self.function.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar):
                    self.var_defs[inst.var].add(block)

    def _insert_phi_nodes(self):
        '''Insert Phi nodes at dominance frontiers (worklist algorithm)'''
        for var, def_blocks in self.var_defs.items():
            worklist: Deque[MediumLevelILBasicBlock] = deque(def_blocks)
            phi_placed: Set[MediumLevelILBasicBlock] = set()

            while worklist:
                block = worklist.popleft()

                for df_block in self.dom_analysis.dom_frontier[block]:
                    if df_block in phi_placed:
                        continue

                    # Insert Phi at beginning
                    phi = MLILPhi(
                        dest = MLILVariableSSA(var, 0),  # Placeholder, updated in renaming
                        sources = []
                    )
                    df_block.instructions.insert(0, phi)
                    phi_placed.add(df_block)

                    # If df_block wasn't in def_blocks, add to worklist
                    if df_block not in def_blocks:
                        worklist.append(df_block)

    def _rename_variables_iterative(self):
        '''Rename variables to SSA form (iterative DFS)'''
        if not self.function.basic_blocks:
            return

        # Initialize all function variables (especially parameters) to version 0
        # Parameters and globals are "defined" at function entry
        for var in self.function.parameters:
            if var is not None:
                self.var_versions[var] = 0
                self.var_stack[var].append(0)

        for var in self.function.locals.values():
            self.var_versions[var] = 0
            self.var_stack[var].append(0)

        entry = self.function.basic_blocks[0]

        # Stack: (block, phase)
        # phase 0: process block
        # phase 1: process successors' phis
        # phase 2: recurse to children
        # phase 3: pop versions
        stack: List[Tuple[MediumLevelILBasicBlock, int, List[MLILVariable]]] = [(entry, 0, [])]
        visited: Set[MediumLevelILBasicBlock] = set()

        while stack:
            block, phase, pushed_vars = stack.pop()

            if phase == 0:
                # Process block instructions
                if block in visited:
                    continue

                visited.add(block)
                pushed = []

                new_insts = []
                for inst in block.instructions:
                    new_inst = self._rename_inst(inst, pushed)
                    new_insts.append(new_inst)

                block.instructions = new_insts

                # Schedule remaining phases
                stack.append((block, 3, pushed))  # Phase 3: pop versions
                stack.append((block, 2, []))       # Phase 2: recurse children
                stack.append((block, 1, []))       # Phase 1: update successor phis

            elif phase == 1:
                # Update Phi nodes in successors
                for succ in block.outgoing_edges:
                    for inst in succ.instructions:
                        if isinstance(inst, MLILPhi):
                            var = inst.dest.base_var

                            if var in self.var_stack and self.var_stack[var]:
                                current_ver = self.var_stack[var][-1]
                                ssa_var = MLILVariableSSA(var, current_ver)
                                inst.sources.append((ssa_var, block))

            elif phase == 2:
                # Recurse to dominated children
                for child in self.dom_analysis.dom_tree[block]:
                    stack.append((child, 0, []))

            elif phase == 3:
                # Pop versions
                for var in pushed_vars:
                    if self.var_stack[var]:
                        self.var_stack[var].pop()

    def _rename_inst(self, inst: MediumLevelILInstruction, pushed: List[MLILVariable]) -> MediumLevelILInstruction:
        '''Rename variables in single instruction'''
        if isinstance(inst, MLILPhi):
            # Allocate new version for Phi dest
            new_ver = self._new_version(inst.dest.base_var)
            pushed.append(inst.dest.base_var)
            inst.dest = MLILVariableSSA(inst.dest.base_var, new_ver)
            return inst

        elif isinstance(inst, MLILSetVar):
            # Rename uses in RHS
            new_value = self._rename_expr(inst.value)

            # Allocate new version for LHS
            new_ver = self._new_version(inst.var)
            pushed.append(inst.var)

            return MLILSetVarSSA(MLILVariableSSA(inst.var, new_ver), new_value)

        else:
            # Other statements: rename expressions
            return self._rename_stmt(inst)

    def _rename_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Recursively rename variables in expression'''
        if isinstance(expr, MLILVar):
            # Replace with SSA version
            var = expr.var

            if var not in self.var_stack or not self.var_stack[var]:
                raise ValueError(f'Use of undefined variable: {var.name}')

            current_ver = self.var_stack[var][-1]
            return MLILVarSSA(MLILVariableSSA(var, current_ver))

        elif isinstance(expr, MLILVarSSA):
            # Already SSA, keep it
            return expr

        elif isinstance(expr, MLILConst):
            return expr

        elif isinstance(expr, MLILBinaryOp):
            new_lhs = self._rename_expr(expr.lhs)
            new_rhs = self._rename_expr(expr.rhs)

            if new_lhs is expr.lhs and new_rhs is expr.rhs:
                return expr

            # Explicit reconstruction (no type() hack)
            return self._rebuild_binary_op(expr, new_lhs, new_rhs)

        elif isinstance(expr, MLILUnaryOp):
            new_operand = self._rename_expr(expr.operand)

            if new_operand is expr.operand:
                return expr

            return self._rebuild_unary_op(expr, new_operand)

        else:
            # Other expressions (LoadGlobal, LoadReg, etc.)
            return expr

    def _rename_stmt(self, stmt: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Rename variables in statement'''
        if isinstance(stmt, MLILIf):
            new_cond = self._rename_expr(stmt.condition)

            if new_cond is not stmt.condition:
                return MLILIf(new_cond, stmt.true_target, stmt.false_target)

        elif isinstance(stmt, MLILRet):
            if stmt.value is not None:
                new_value = self._rename_expr(stmt.value)

                if new_value is not stmt.value:
                    return MLILRet(new_value)

        elif isinstance(stmt, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._rename_expr(arg) for arg in stmt.args]

            if any(new_args[i] is not stmt.args[i] for i in range(len(stmt.args))):
                if isinstance(stmt, MLILCall):
                    return MLILCall(stmt.target, new_args)

                elif isinstance(stmt, MLILSyscall):
                    return MLILSyscall(stmt.subsystem, stmt.cmd, new_args)

                elif isinstance(stmt, MLILCallScript):
                    return MLILCallScript(stmt.module, stmt.func, new_args)

        elif isinstance(stmt, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._rename_expr(stmt.value)

            if new_value is not stmt.value:
                if isinstance(stmt, MLILStoreGlobal):
                    return MLILStoreGlobal(stmt.index, new_value)

                else:
                    return MLILStoreReg(stmt.index, new_value)

        return stmt

    def _rebuild_binary_op(self, expr: MLILBinaryOp, lhs, rhs) -> MediumLevelILInstruction:
        '''Rebuild binary operation (explicit, not type())'''
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
            raise NotImplementedError(f'Unknown binary op: {type(expr).__name__}')

    def _rebuild_unary_op(self, expr: MLILUnaryOp, operand) -> MediumLevelILInstruction:
        '''Rebuild unary operation (explicit, not type())'''
        if isinstance(expr, MLILNeg):
            return MLILNeg(operand)

        elif isinstance(expr, MLILLogicalNot):
            return MLILLogicalNot(operand)

        elif isinstance(expr, MLILTestZero):
            return MLILTestZero(operand)

        elif isinstance(expr, MLILAddressOf):
            return MLILAddressOf(operand)

        else:
            raise NotImplementedError(f'Unknown unary op: {type(expr).__name__}')

    def _new_version(self, var: MLILVariable) -> int:
        '''Allocate new SSA version for variable'''
        if var not in self.var_versions:
            self.var_versions[var] = 0

        version = self.var_versions[var]
        self.var_versions[var] += 1
        self.var_stack[var].append(version)
        return version


# ============================================================================
# SSA Deconstruction
# ============================================================================

class SSADeconstructor:
    '''Convert SSA back to non-SSA form'''

    def __init__(self, function: MediumLevelILFunction):
        self.function = function

    def deconstruct(self) -> MediumLevelILFunction:
        '''Convert from SSA to non-SSA (modifies in-place)'''
        # Step 1: Eliminate Phi nodes (insert copies in predecessors)
        self._eliminate_phi_nodes()

        # Step 2: Replace SSA variables with base variables
        self._remove_ssa_versions()

        return self.function

    def _eliminate_phi_nodes(self):
        '''Replace Phi nodes with copies in predecessor blocks'''
        for block in self.function.basic_blocks:
            phi_nodes = [inst for inst in block.instructions if isinstance(inst, MLILPhi)]

            if not phi_nodes:
                continue

            # Remove Phi nodes from this block
            block.instructions = [inst for inst in block.instructions if not isinstance(inst, MLILPhi)]

            # Insert copies in predecessors
            for phi in phi_nodes:
                for ssa_var, pred_block in phi.sources:
                    # Skip self-assignment (var_s3 = var_s3)
                    if phi.dest.base_var == ssa_var.base_var:
                        continue

                    # Insert: phi.dest.base_var = ssa_var.base_var at end of pred_block
                    copy = MLILSetVar(phi.dest.base_var, MLILVar(ssa_var.base_var))
                    # Insert before terminal instruction
                    if pred_block.instructions and pred_block.has_terminal:
                        pred_block.instructions.insert(-1, copy)

                    else:
                        pred_block.instructions.append(copy)

    def _remove_ssa_versions(self):
        '''Replace MLILVarSSA/MLILSetVarSSA with MLILVar/MLILSetVar'''
        for block in self.function.basic_blocks:
            new_insts = []

            for inst in block.instructions:
                new_inst = self._deversion_inst(inst)
                new_insts.append(new_inst)

            block.instructions = new_insts

    def _deversion_inst(self, inst: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Remove SSA versions from instruction'''
        if isinstance(inst, MLILSetVarSSA):
            new_value = self._deversion_expr(inst.value)
            return MLILSetVar(inst.var.base_var, new_value)

        elif isinstance(inst, MLILPhi):
            # Should have been eliminated already
            raise RuntimeError('Phi node not eliminated')

        else:
            return self._deversion_stmt(inst)

    def _deversion_expr(self, expr: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Remove SSA versions from expression'''
        if isinstance(expr, MLILVarSSA):
            return MLILVar(expr.var.base_var)

        elif isinstance(expr, MLILBinaryOp):
            new_lhs = self._deversion_expr(expr.lhs)
            new_rhs = self._deversion_expr(expr.rhs)

            if new_lhs is expr.lhs and new_rhs is expr.rhs:
                return expr

            # Use same rebuild as constructor
            constructor = SSAConstructor(self.function)
            return constructor._rebuild_binary_op(expr, new_lhs, new_rhs)

        elif isinstance(expr, MLILUnaryOp):
            new_operand = self._deversion_expr(expr.operand)

            if new_operand is expr.operand:
                return expr

            constructor = SSAConstructor(self.function)
            return constructor._rebuild_unary_op(expr, new_operand)

        else:
            return expr

    def _deversion_stmt(self, stmt: MediumLevelILInstruction) -> MediumLevelILInstruction:
        '''Remove SSA versions from statement'''
        if isinstance(stmt, MLILIf):
            new_cond = self._deversion_expr(stmt.condition)

            if new_cond is not stmt.condition:
                return MLILIf(new_cond, stmt.true_target, stmt.false_target)

        elif isinstance(stmt, MLILRet):
            if stmt.value:
                new_value = self._deversion_expr(stmt.value)

                if new_value is not stmt.value:
                    return MLILRet(new_value)

        elif isinstance(stmt, (MLILCall, MLILSyscall, MLILCallScript)):
            new_args = [self._deversion_expr(arg) for arg in stmt.args]

            if any(new_args[i] is not stmt.args[i] for i in range(len(stmt.args))):
                if isinstance(stmt, MLILCall):
                    return MLILCall(stmt.target, new_args)

                elif isinstance(stmt, MLILSyscall):
                    return MLILSyscall(stmt.subsystem, stmt.cmd, new_args)

                elif isinstance(stmt, MLILCallScript):
                    return MLILCallScript(stmt.module, stmt.func, new_args)

        elif isinstance(stmt, (MLILStoreGlobal, MLILStoreReg)):
            new_value = self._deversion_expr(stmt.value)

            if new_value is not stmt.value:
                if isinstance(stmt, MLILStoreGlobal):
                    return MLILStoreGlobal(stmt.index, new_value)

                else:
                    return MLILStoreReg(stmt.index, new_value)

        return stmt


# ============================================================================
# Public API
# ============================================================================

def convert_to_ssa(function: MediumLevelILFunction) -> MediumLevelILFunction:
    '''Convert MLIL to SSA (in-place)'''
    constructor = SSAConstructor(function)
    return constructor.construct()


def convert_from_ssa(function: MediumLevelILFunction) -> MediumLevelILFunction:
    '''Convert from SSA to non-SSA (in-place)'''
    deconstructor = SSADeconstructor(function)
    return deconstructor.deconstruct()

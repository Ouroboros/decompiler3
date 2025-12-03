'''MLIL to HLIL Converter - SSA-based Graph Rewriting'''

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Union

from ir.mlil.mlil import *
from ir.mlil.mlil_types import MLILType, MLILTypeKind
from .hlil import *


# Operator mapping tables
_BINARY_OP_MAP = {
    MediumLevelILOperation.MLIL_ADD         : '+',
    MediumLevelILOperation.MLIL_SUB         : '-',
    MediumLevelILOperation.MLIL_MUL         : '*',
    MediumLevelILOperation.MLIL_DIV         : '/',
    MediumLevelILOperation.MLIL_MOD         : '%',
    MediumLevelILOperation.MLIL_AND         : '&',
    MediumLevelILOperation.MLIL_OR          : '|',
    MediumLevelILOperation.MLIL_XOR         : '^',
    MediumLevelILOperation.MLIL_SHL         : '<<',
    MediumLevelILOperation.MLIL_SHR         : '>>',
    MediumLevelILOperation.MLIL_LOGICAL_AND : '&&',
    MediumLevelILOperation.MLIL_LOGICAL_OR  : '||',
    MediumLevelILOperation.MLIL_EQ          : '==',
    MediumLevelILOperation.MLIL_NE          : '!=',
    MediumLevelILOperation.MLIL_LT          : '<',
    MediumLevelILOperation.MLIL_LE          : '<=',
    MediumLevelILOperation.MLIL_GT          : '>',
    MediumLevelILOperation.MLIL_GE          : '>=',
}

_UNARY_OP_MAP = {
    MediumLevelILOperation.MLIL_NEG : '-',
}

_MLIL_TYPE_MAP = {
    MLILTypeKind.UNKNOWN  : HLILTypeKind.UNKNOWN,
    MLILTypeKind.INT      : HLILTypeKind.INT,
    MLILTypeKind.FLOAT    : HLILTypeKind.FLOAT,
    MLILTypeKind.STRING   : HLILTypeKind.STRING,
    MLILTypeKind.BOOL     : HLILTypeKind.INT,
    MLILTypeKind.POINTER  : HLILTypeKind.INT,
    MLILTypeKind.VARIANT  : HLILTypeKind.UNKNOWN,
    MLILTypeKind.VOID     : HLILTypeKind.VOID,
}


def _mlil_type_to_hlil(mlil_type: MLILType) -> HLILTypeKind:
    return _MLIL_TYPE_MAP[mlil_type.kind]


# ============================================================================
# SSA-based Graph Rewriting for Register Resolution
# ============================================================================

@dataclass
class SSADef:
    '''Definition of an SSA variable'''
    block_idx: int
    instr_idx: int
    reg_index: int
    expr: MediumLevelILInstruction  # The defining expression
    is_call: bool
    is_live_out: bool = False
    use_count: int = 0
    deleted: bool = False

    def __hash__(self):
        # Hash by position (identity)
        return hash((self.block_idx, self.instr_idx))

    def __eq__(self, other):
        if not isinstance(other, SSADef):
            return False
        return self.block_idx == other.block_idx and self.instr_idx == other.instr_idx


@dataclass
class SSAUse:
    '''A use of an SSA variable'''
    block_idx: int
    instr_idx: int
    def_ref: SSADef  # Points to the definition


class RegisterResolver:
    '''
    SSA-based Graph Rewriting for Register Resolution.

    Pipeline:
      Phase A: Build def-use chains, compute liveness (with cross-block reaching defs)
      Phase B: Graph rewriting (rewire operands)
      Phase C: Safe DCE
      Phase D: Used during code generation
    '''

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func

        # Definitions: (block_idx, instr_idx) -> SSADef
        self.definitions: Dict[Tuple[int, int], SSADef] = {}

        # Uses: (block_idx, instr_idx, reg_index) -> SSAUse
        self.uses: Dict[Tuple[int, int, int], SSAUse] = {}

        # Rewiring map: (block_idx, instr_idx, reg_index) -> SSADef (the def to use instead)
        self.rewired: Dict[Tuple[int, int, int], SSADef] = {}

        # Deleted instructions
        self.deleted: Set[Tuple[int, int]] = set()

        # CFG info
        self.predecessors: Dict[int, List[int]] = {}
        self.successors: Dict[int, List[int]] = {}

        # Reaching definitions at block entry: block_idx -> {reg_idx -> Set[SSADef]}
        self.reaching_defs_entry: Dict[int, Dict[int, Set[SSADef]]] = {}

    def analyze(self):
        '''Run the full analysis and rewriting pipeline'''
        self._build_cfg()
        self._phase_a_analysis()
        self._phase_b_rewriting()
        self._phase_c_dce()

    def _build_cfg(self):
        '''Build predecessor/successor maps'''
        for i, block in enumerate(self.mlil_func.basic_blocks):
            self.predecessors[i] = []
            self.successors[i] = []

        for i, block in enumerate(self.mlil_func.basic_blocks):
            if not block.instructions:
                if i + 1 < len(self.mlil_func.basic_blocks):
                    self.successors[i].append(i + 1)
                    self.predecessors[i + 1].append(i)
                continue

            last_instr = block.instructions[-1]
            if isinstance(last_instr, MLILIf):
                if last_instr.true_target is not None:
                    succ = last_instr.true_target.index
                    self.successors[i].append(succ)
                    self.predecessors[succ].append(i)
                if last_instr.false_target is not None:
                    succ = last_instr.false_target.index
                    self.successors[i].append(succ)
                    self.predecessors[succ].append(i)

            elif isinstance(last_instr, MLILGoto):
                if last_instr.target is not None:
                    succ = last_instr.target.index
                    self.successors[i].append(succ)
                    self.predecessors[succ].append(i)

            elif not isinstance(last_instr, MLILRet):
                if i + 1 < len(self.mlil_func.basic_blocks):
                    self.successors[i].append(i + 1)
                    self.predecessors[i + 1].append(i)

    def _phase_a_analysis(self):
        '''Phase A: Build def-use chains, compute liveness'''

        # Pass 1: Collect all definitions and compute local exit defs
        local_exit_defs: Dict[int, Dict[int, SSADef]] = {}  # block_idx -> {reg_idx -> SSADef}

        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            current_def: Dict[int, SSADef] = {}

            for instr_idx, instr in enumerate(block.instructions):
                key = (block_idx, instr_idx)

                # Record definitions
                if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                    ssa_def = SSADef(block_idx, instr_idx, 0, instr, is_call=True)
                    self.definitions[key] = ssa_def
                    current_def[0] = ssa_def

                elif isinstance(instr, MLILStoreReg):
                    is_call = isinstance(instr.value, (MLILCall, MLILSyscall, MLILCallScript))
                    ssa_def = SSADef(block_idx, instr_idx, instr.index, instr, is_call=is_call)
                    self.definitions[key] = ssa_def
                    current_def[instr.index] = ssa_def

            local_exit_defs[block_idx] = current_def

        # Pass 2: Compute reaching definitions (forward dataflow)
        self._compute_reaching_definitions(local_exit_defs)

        # Pass 3: Link uses to definitions (with cross-block support)
        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            # Start with reaching defs at block entry
            current_def: Dict[int, SSADef] = {}
            entry_defs = self.reaching_defs_entry.get(block_idx, {})
            for reg_idx, defs in entry_defs.items():
                # If multiple defs reach, we can't inline (set to None marker)
                if len(defs) == 1:
                    current_def[reg_idx] = next(iter(defs))

            for instr_idx, instr in enumerate(block.instructions):
                key = (block_idx, instr_idx)

                # Count uses with current reaching defs
                self._count_uses(instr, block_idx, instr_idx, current_def)

                # Update current_def after processing uses
                if key in self.definitions:
                    ssa_def = self.definitions[key]
                    current_def[ssa_def.reg_index] = ssa_def

        # Pass 4: Liveness analysis - mark live-out registers before RETURN
        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            for instr_idx, instr in enumerate(block.instructions):
                if isinstance(instr, MLILRet):
                    self._mark_live_out_before(block_idx, instr_idx)

        # Pass 5: Mark defs as live-out if they reach successor blocks
        self._mark_cross_block_live_out(local_exit_defs)

    def _compute_reaching_definitions(self, local_exit_defs: Dict[int, Dict[int, SSADef]]):
        '''Forward dataflow: compute reaching definitions at each block entry'''
        num_blocks = len(self.mlil_func.basic_blocks)

        # Use (block_idx, instr_idx) tuples as def identifiers (hashable)
        # exit_defs: block_idx -> {reg_idx -> Set[(block_idx, instr_idx)]}
        exit_defs: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}
        entry_defs: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}

        for i in range(num_blocks):
            entry_defs[i] = {}
            exit_defs[i] = {}

        # Iterate until fixed point
        changed = True
        iterations = 0
        max_iterations = num_blocks * 2 + 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for block_idx in range(num_blocks):
                # Entry = union of predecessors' exits
                new_entry: Dict[int, Set[Tuple[int, int]]] = {}
                for pred_idx in self.predecessors.get(block_idx, []):
                    for reg_idx, def_keys in exit_defs.get(pred_idx, {}).items():
                        if reg_idx not in new_entry:
                            new_entry[reg_idx] = set()
                        new_entry[reg_idx] |= def_keys

                # Exit = local defs override entry
                new_exit: Dict[int, Set[Tuple[int, int]]] = {k: set(v) for k, v in new_entry.items()}
                if block_idx in local_exit_defs:
                    for reg_idx, ssa_def in local_exit_defs[block_idx].items():
                        new_exit[reg_idx] = {(ssa_def.block_idx, ssa_def.instr_idx)}

                # Check for changes
                if new_exit != exit_defs.get(block_idx, {}):
                    exit_defs[block_idx] = new_exit
                    changed = True

                entry_defs[block_idx] = new_entry

        # Convert entry_defs to reaching_defs_entry with SSADef references
        for block_idx, reg_defs in entry_defs.items():
            self.reaching_defs_entry[block_idx] = {}
            for reg_idx, def_keys in reg_defs.items():
                defs = set()
                for def_key in def_keys:
                    if def_key in self.definitions:
                        defs.add(self.definitions[def_key])
                if defs:
                    self.reaching_defs_entry[block_idx][reg_idx] = defs

    def _mark_cross_block_live_out(self, local_exit_defs: Dict[int, Dict[int, SSADef]]):
        '''Mark definitions as live-out if they may be used in successor blocks'''
        # Collect which registers each block uses at entry (before any local def)
        block_entry_uses: Dict[int, Set[int]] = {}
        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            entry_uses: Set[int] = set()
            local_defs: Set[int] = set()

            for instr in block.instructions:
                # Collect uses of registers not yet defined locally
                self._collect_reg_uses(instr, entry_uses, local_defs)
                # Collect local definitions
                if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                    local_defs.add(0)

                elif isinstance(instr, MLILStoreReg):
                    local_defs.add(instr.index)

            block_entry_uses[block_idx] = entry_uses

        # Mark defs as live-out if successor uses them
        for block_idx in range(len(self.mlil_func.basic_blocks)):
            if block_idx not in local_exit_defs:
                continue

            for succ_idx in self.successors.get(block_idx, []):
                entry_uses = block_entry_uses.get(succ_idx, set())
                for reg_idx in entry_uses:
                    if reg_idx in local_exit_defs[block_idx]:
                        local_exit_defs[block_idx][reg_idx].is_live_out = True

    def _collect_reg_uses(self, instr: MediumLevelILInstruction,
                          entry_uses: Set[int], local_defs: Set[int]):
        '''Collect register uses that are not locally defined'''
        def visit(expr):
            if isinstance(expr, MLILLoadReg):
                if expr.index not in local_defs:
                    entry_uses.add(expr.index)

            elif isinstance(expr, MLILBinaryOp):
                visit(expr.lhs)
                visit(expr.rhs)

            elif isinstance(expr, MLILUnaryOp):
                visit(expr.operand)

            elif isinstance(expr, MLILSetVar):
                visit(expr.value)

            elif isinstance(expr, MLILStoreReg):
                visit(expr.value)

            elif isinstance(expr, MLILIf):
                visit(expr.condition)

            elif isinstance(expr, MLILRet):
                if expr.value:
                    visit(expr.value)

            elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in expr.args:
                    visit(arg)

        visit(instr)

    def _count_uses(self, instr: MediumLevelILInstruction,
                    block_idx: int, instr_idx: int,
                    current_def: Dict[int, SSADef]):
        '''Count register uses in an instruction'''

        def visit(expr):
            if isinstance(expr, MLILLoadReg):
                ssa_def = current_def.get(expr.index)
                if ssa_def is not None:
                    ssa_def.use_count += 1
                    use_key = (block_idx, instr_idx, expr.index)
                    self.uses[use_key] = SSAUse(block_idx, instr_idx, ssa_def)

            elif isinstance(expr, MLILBinaryOp):
                visit(expr.lhs)
                visit(expr.rhs)

            elif isinstance(expr, MLILUnaryOp):
                visit(expr.operand)

            elif isinstance(expr, MLILAddressOf):
                visit(expr.operand)

            elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in expr.args:
                    visit(arg)

            elif isinstance(expr, MLILSetVar):
                visit(expr.value)

            elif isinstance(expr, MLILStoreReg):
                visit(expr.value)

            elif isinstance(expr, MLILStoreGlobal):
                visit(expr.value)

            elif isinstance(expr, MLILIf):
                visit(expr.condition)

            elif isinstance(expr, MLILRet):
                if expr.value:
                    visit(expr.value)

        visit(instr)

    def _mark_live_out_before(self, block_idx: int, instr_idx: int):
        '''Mark registers as live-out before a return instruction'''
        # Find the last definition of REG[0] before this point
        block = self.mlil_func.basic_blocks[block_idx]
        for i in range(instr_idx - 1, -1, -1):
            instr = block.instructions[i]
            key = (block_idx, i)

            if key in self.definitions:
                ssa_def = self.definitions[key]
                if ssa_def.reg_index == 0:  # REG[0] is the return value register
                    ssa_def.is_live_out = True
                    break

            if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                # Call defines REG[0]
                if key in self.definitions:
                    self.definitions[key].is_live_out = True
                break

    def _phase_b_rewriting(self):
        '''Phase B: Graph rewriting - rewire operands to bypass registers'''

        for key, ssa_def in list(self.definitions.items()):
            if ssa_def.reg_index < 0:
                continue  # Not a register definition

            # Find all uses of this register definition
            uses_to_rewire = []
            for use_key, ssa_use in self.uses.items():
                if ssa_use.def_ref is ssa_def:
                    uses_to_rewire.append(use_key)

            if not uses_to_rewire:
                continue

            # Check if we can rewire: need to find the source of this register
            block = self.mlil_func.basic_blocks[ssa_def.block_idx]
            instr = block.instructions[ssa_def.instr_idx]

            source_def = None

            if isinstance(instr, MLILStoreReg):
                # REG[n] = value - check if value comes from a call or another reg
                if isinstance(instr.value, MLILLoadReg):
                    # REG[n] = REG[m] - follow the chain
                    source_key = None
                    for use_k, use in self.uses.items():
                        if (use_k[0] == ssa_def.block_idx and
                            use_k[1] == ssa_def.instr_idx and
                            use_k[2] == instr.value.index):
                            source_def = use.def_ref
                            break

            elif isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                # Call directly defines REG[0]
                source_def = ssa_def  # The call itself is the source

            if source_def is not None and source_def is not ssa_def:
                # Rewire all uses to the source
                for use_key in uses_to_rewire:
                    self.rewired[use_key] = source_def
                    source_def.use_count += 1
                    ssa_def.use_count -= 1

    def _phase_c_dce(self):
        '''Phase C: Safe DCE - remove dead register assignments'''

        for key, ssa_def in self.definitions.items():
            if ssa_def.use_count == 0 and not ssa_def.is_live_out:
                # Dead and not live-out - can delete
                if not ssa_def.is_call:
                    # Only delete non-call assignments (calls have side effects)
                    self.deleted.add(key)

    def get_inlined_expr(self, block_idx: int, instr_idx: int, reg_index: int) -> Optional[SSADef]:
        '''Get the definition to inline for a LoadReg'''
        use_key = (block_idx, instr_idx, reg_index)

        # Check if rewired
        if use_key in self.rewired:
            return self.rewired[use_key]

        # Check original use
        if use_key in self.uses:
            return self.uses[use_key].def_ref

        return None

    def is_deleted(self, block_idx: int, instr_idx: int) -> bool:
        '''Check if an instruction was deleted by DCE'''
        return (block_idx, instr_idx) in self.deleted

    def should_emit_as_stmt(self, block_idx: int, instr_idx: int) -> bool:
        '''Check if a call should be emitted as ExprStmt (unused result)'''
        key = (block_idx, instr_idx)
        if key not in self.definitions:
            return False
        ssa_def = self.definitions[key]
        return ssa_def.is_call and ssa_def.use_count == 0 and not ssa_def.is_live_out

    def get_use_count(self, block_idx: int, instr_idx: int) -> int:
        '''Get use count for a definition'''
        key = (block_idx, instr_idx)
        if key in self.definitions:
            return self.definitions[key].use_count
        return 0


# ============================================================================
# MLIL to HLIL Converter
# ============================================================================

class MLILToHLILConverter:

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = HighLevelILFunction(mlil_func.name, mlil_func.start_addr)

        self.block_successors: Dict[int, List[int]] = {}
        self.visited_blocks: Set[int] = set()
        self.globally_processed: Set[int] = set()

        # SSA-based register resolution
        self.resolver = RegisterResolver(mlil_func)

        # Expression cache for inlining
        self.expr_cache: Dict[Tuple[int, int], HLILExpression] = {}

        # Temp variable generation for multi-use calls
        self.temp_vars: Dict[Tuple[int, int], str] = {}
        self.temp_counter: int = 0

        # Current context
        self.current_block_idx: int = 0
        self.current_instr_idx: int = 0

    def convert(self) -> HighLevelILFunction:
        # Phase 0: Run SSA analysis
        self.resolver.analyze()

        # Phase 1: Identify multi-use calls and assign temp vars
        self._assign_temp_vars()

        # Phase 2: Convert variables
        self._convert_variables()

        # Phase 3: Build CFG
        self._build_cfg()

        # Phase 4: Reconstruct control flow
        if self.mlil_func.basic_blocks:
            self._reconstruct_control_flow(0, self.hlil_func.body)

        return self.hlil_func

    def _assign_temp_vars(self):
        '''Assign temp variable names to multi-use call results'''
        for key, ssa_def in self.resolver.definitions.items():
            if ssa_def.is_call and ssa_def.use_count > 1:
                temp_name = f'_temp{self.temp_counter}' if self.temp_counter > 0 else '_temp'
                self.temp_vars[key] = temp_name
                self.temp_counter += 1

    def _convert_variables(self):
        defined_vars = set()
        read_vars = set()

        def collect_read_vars(expr):
            if isinstance(expr, MLILVar):
                read_vars.add(expr.var.name)
            elif isinstance(expr, MLILBinaryOp):
                collect_read_vars(expr.lhs)
                collect_read_vars(expr.rhs)
            elif isinstance(expr, MLILUnaryOp):
                collect_read_vars(expr.operand)
            elif isinstance(expr, MLILAddressOf):
                collect_read_vars(expr.operand)
            elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in expr.args:
                    collect_read_vars(arg)

        for block in self.mlil_func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar):
                    defined_vars.add(inst.var.name)
                    collect_read_vars(inst.value)
                elif isinstance(inst, MLILIf):
                    collect_read_vars(inst.condition)
                elif isinstance(inst, MLILRet) and inst.value:
                    collect_read_vars(inst.value)
                elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
                    for arg in inst.args:
                        collect_read_vars(arg)
                elif isinstance(inst, MLILStoreGlobal):
                    collect_read_vars(inst.value)
                elif isinstance(inst, MLILStoreReg):
                    collect_read_vars(inst.value)

        used_vars = defined_vars & read_vars

        for mlil_var in self.mlil_func.parameters:
            if mlil_var is None or mlil_var.name not in read_vars:
                continue
            type_hint = self._get_type_hint(mlil_var.name)
            hlil_var = HLILVariable(mlil_var.name, type_hint=type_hint, kind=VariableKind.PARAM)
            self.hlil_func.parameters.append(hlil_var)

        for mlil_var in self.mlil_func.locals.values():
            if mlil_var.name not in used_vars:
                continue
            type_hint = self._get_type_hint(mlil_var.name)
            hlil_var = HLILVariable(mlil_var.name, type_hint=type_hint)
            self.hlil_func.variables.append(hlil_var)

    def _get_type_hint(self, var_name: str) -> Optional[HLILTypeKind]:
        mlil_type = self.mlil_func.var_types.get(var_name)
        if mlil_type is None:
            return None
        hlil_type = _mlil_type_to_hlil(mlil_type)
        if hlil_type == HLILTypeKind.UNKNOWN:
            return None
        return hlil_type

    def _build_cfg(self):
        for i, block in enumerate(self.mlil_func.basic_blocks):
            successors = []
            if block.instructions:
                last_instr = block.instructions[-1]
                if isinstance(last_instr, MLILIf):
                    if last_instr.true_target is not None:
                        successors.append(last_instr.true_target.index)
                    if last_instr.false_target is not None:
                        successors.append(last_instr.false_target.index)
                elif isinstance(last_instr, MLILGoto):
                    if last_instr.target is not None:
                        successors.append(last_instr.target.index)
                elif not isinstance(last_instr, MLILRet):
                    if i + 1 < len(self.mlil_func.basic_blocks):
                        successors.append(i + 1)
            self.block_successors[i] = successors

    def _get_reachable(self, start_idx: int, max_depth: int = 10) -> Dict[int, int]:
        reachable = {}
        queue = deque([(start_idx, 0)])
        visited = set()
        while queue:
            idx, depth = queue.popleft()
            if idx in visited or depth > max_depth:
                continue
            visited.add(idx)
            reachable[idx] = depth
            for succ in self.block_successors.get(idx, []):
                if succ not in visited:
                    queue.append((succ, depth + 1))
        return reachable

    def _find_merge_block(self, block1_idx: int, block2_idx: int) -> Optional[int]:
        if block1_idx == block2_idx:
            return block1_idx
        succ1 = set(self.block_successors.get(block1_idx, []))
        succ2 = set(self.block_successors.get(block2_idx, []))
        common = succ1 & succ2
        if common:
            return min(common)
        reachable1 = self._get_reachable(block1_idx)
        reachable2 = self._get_reachable(block2_idx)
        if block1_idx in reachable2:
            return block1_idx
        if block2_idx in reachable1:
            return block2_idx
        common_blocks = set(reachable1.keys()) & set(reachable2.keys())
        if not common_blocks:
            return None
        return min(common_blocks, key=lambda idx: reachable1[idx] + reachable2[idx])

    def _is_passthrough_block(self, block_idx: int) -> bool:
        if block_idx >= len(self.mlil_func.basic_blocks):
            return False
        block = self.mlil_func.basic_blocks[block_idx]
        if not block.instructions:
            return False
        return (
            all(isinstance(instr, (MLILDebug, MLILNop)) for instr in block.instructions[:-1]) and
            isinstance(block.instructions[-1], MLILGoto)
        )

    def _reconstruct_control_flow(self, block_idx: int, target_block: HLILBlock,
                                   stop_at: Optional[int] = None) -> Optional[int]:
        seen_passthrough = set()
        while self._is_passthrough_block(block_idx) and block_idx not in seen_passthrough and block_idx != stop_at:
            seen_passthrough.add(block_idx)
            mlil_block = self.mlil_func.basic_blocks[block_idx]
            last_instr = mlil_block.instructions[-1]
            if isinstance(last_instr, MLILGoto) and last_instr.target:
                block_idx = last_instr.target.index
            else:
                break

        if block_idx >= len(self.mlil_func.basic_blocks):
            return None
        if block_idx in self.visited_blocks:
            return None
        if block_idx in self.globally_processed:
            return None
        if stop_at is not None and block_idx == stop_at:
            self.visited_blocks.add(block_idx)
            return block_idx

        self.visited_blocks.add(block_idx)
        self.globally_processed.add(block_idx)
        self.current_block_idx = block_idx
        mlil_block = self.mlil_func.basic_blocks[block_idx]

        for instr_idx, instr in enumerate(mlil_block.instructions[:-1]):
            self.current_instr_idx = instr_idx
            stmts = self._convert_instruction(instr, block_idx, instr_idx)
            for stmt in stmts:
                target_block.add_statement(stmt)

        if mlil_block.instructions:
            last_instr = mlil_block.instructions[-1]
            last_instr_idx = len(mlil_block.instructions) - 1
            self.current_instr_idx = last_instr_idx

            if isinstance(last_instr, MLILIf):
                condition = self._convert_expr(last_instr.condition)
                true_target_idx = last_instr.true_target.index if last_instr.true_target else None
                false_target_idx = last_instr.false_target.index if last_instr.false_target else None

                merge_block_idx = None
                if true_target_idx is not None and false_target_idx is not None:
                    merge_block_idx = self._find_merge_block(true_target_idx, false_target_idx)

                true_block = HLILBlock()
                false_block = HLILBlock()
                branch_stop = merge_block_idx if merge_block_idx is not None else stop_at

                saved_visited = self.visited_blocks.copy()
                branch_visited_base = saved_visited.copy()
                if stop_at is not None and stop_at != merge_block_idx:
                    branch_visited_base.add(stop_at)

                if true_target_idx is not None:
                    self.visited_blocks = branch_visited_base.copy()
                    self._reconstruct_control_flow(true_target_idx, true_block, stop_at=branch_stop)

                true_visited = self.visited_blocks
                if stop_at is not None:
                    true_visited = true_visited - {stop_at}

                if false_target_idx is not None:
                    self.visited_blocks = branch_visited_base | true_visited
                    self._reconstruct_control_flow(false_target_idx, false_block, stop_at=branch_stop)

                false_visited = self.visited_blocks
                if stop_at is not None:
                    false_visited = false_visited - {stop_at}
                self.visited_blocks = saved_visited | true_visited | false_visited

                if_stmt = HLILIf(condition, true_block, false_block if false_block.statements else None)
                target_block.add_statement(if_stmt)

                if merge_block_idx is not None:
                    if stop_at is not None and merge_block_idx == stop_at:
                        return merge_block_idx
                    self.visited_blocks.discard(merge_block_idx)
                    return self._reconstruct_control_flow(merge_block_idx, target_block, stop_at=stop_at)
                return None

            elif isinstance(last_instr, MLILGoto):
                if last_instr.target is not None:
                    return self._reconstruct_control_flow(last_instr.target.index, target_block, stop_at=stop_at)
                return None

            elif isinstance(last_instr, MLILRet):
                stmts = self._convert_instruction(last_instr, block_idx, last_instr_idx)
                for stmt in stmts:
                    target_block.add_statement(stmt)
                return None

            else:
                stmts = self._convert_instruction(last_instr, block_idx, last_instr_idx)
                for stmt in stmts:
                    target_block.add_statement(stmt)
                if block_idx + 1 < len(self.mlil_func.basic_blocks):
                    return self._reconstruct_control_flow(block_idx + 1, target_block, stop_at=stop_at)

        return None

    def _convert_instruction(self, instr: MediumLevelILInstruction,
                              block_idx: int, instr_idx: int) -> List[HLILStatement]:
        '''Convert instruction, may return multiple statements for temp vars'''
        key = (block_idx, instr_idx)
        result = []

        # Check if deleted by DCE
        if self.resolver.is_deleted(block_idx, instr_idx):
            return result

        if isinstance(instr, MLILDebug):
            result.append(HLILComment(f'{instr.debug_type}({instr.value})'))

        elif isinstance(instr, MLILNop):
            pass

        elif isinstance(instr, MLILRet):
            if instr.value:
                result.append(HLILReturn(self._convert_expr(instr.value)))
            else:
                result.append(HLILReturn())

        elif isinstance(instr, MLILSetVar):
            result.append(HLILAssign(
                HLILVar(HLILVariable(instr.var.name, None)),
                self._convert_expr(instr.value)
            ))

        elif isinstance(instr, MLILStoreReg):
            # Explicit register writes are global state - ALWAYS emit
            # (Unlike call return values which can be inlined)
            var = HLILVariable(kind=VariableKind.REG, index=instr.index)
            result.append(HLILAssign(HLILVar(var), self._convert_expr(instr.value)))
            # Also cache for same-block LoadReg inlining
            self.expr_cache[key] = self._convert_expr(instr.value)

        elif isinstance(instr, MLILStoreGlobal):
            var = HLILVariable(kind=VariableKind.GLOBAL, index=instr.index)
            result.append(HLILAssign(HLILVar(var), self._convert_expr(instr.value)))

        elif isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
            # Call implicitly writes to REG[0]
            call_expr = self._convert_expr(instr)
            use_count = self.resolver.get_use_count(block_idx, instr_idx)
            ssa_def = self.resolver.definitions.get(key)

            if use_count == 1 and not (ssa_def and ssa_def.is_live_out):
                # Single use, not live-out: will be inlined at LoadReg site
                self.expr_cache[key] = call_expr

            elif use_count == 0 and not (ssa_def and ssa_def.is_live_out):
                # Unused result: emit as statement for side effects
                result.append(HLILExprStmt(call_expr))

            else:
                # Multiple uses or live-out: emit REGS[0] = call()
                var = HLILVariable(kind=VariableKind.REG, index=0)
                result.append(HLILAssign(HLILVar(var), call_expr))

        elif isinstance(instr, (MLILIf, MLILGoto)):
            pass

        elif isinstance(instr, MediumLevelILExpr):
            result.append(HLILExprStmt(self._convert_expr(instr)))

        return result

    def _convert_expr(self, expr: MediumLevelILExpr) -> HLILExpression:
        if isinstance(expr, MLILVar):
            return HLILVar(HLILVariable(expr.var.name, None))

        elif isinstance(expr, MLILConst):
            return HLILConst(expr.value)

        elif isinstance(expr, MLILBinaryOp):
            op = _BINARY_OP_MAP.get(expr.operation, '?')
            return HLILBinaryOp(op, self._convert_expr(expr.lhs), self._convert_expr(expr.rhs))

        elif isinstance(expr, MLILAddressOf):
            operand = self._convert_expr(expr.operand)
            return HLILCall('addr_of', [operand])

        elif isinstance(expr, MLILUnaryOp):
            if expr.operation in (MediumLevelILOperation.MLIL_LOGICAL_NOT, MediumLevelILOperation.MLIL_TEST_ZERO):
                operand = self._convert_expr(expr.operand)
                return HLILBinaryOp('==', operand, HLILConst(0))
            op = _UNARY_OP_MAP.get(expr.operation, '?')
            return HLILUnaryOp(op, self._convert_expr(expr.operand))

        elif isinstance(expr, MLILLoadReg):
            # Check if we can inline a call result
            inlined_def = self.resolver.get_inlined_expr(
                self.current_block_idx, self.current_instr_idx, expr.index
            )

            if inlined_def is not None:
                def_key = (inlined_def.block_idx, inlined_def.instr_idx)

                # Only inline if it's a Call (not StoreReg)
                if inlined_def.is_call and def_key in self.expr_cache:
                    return self.expr_cache[def_key]

            # Default: emit REGS[n]
            var = HLILVariable(kind=VariableKind.REG, index=expr.index)
            return HLILVar(var)

        elif isinstance(expr, MLILLoadGlobal):
            var = HLILVariable(kind=VariableKind.GLOBAL, index=expr.index)
            return HLILVar(var)

        elif isinstance(expr, MLILCall):
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILCall(str(expr.target), args)

        elif isinstance(expr, MLILSyscall):
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILSyscall(expr.subsystem, expr.cmd, args)

        elif isinstance(expr, MLILCallScript):
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILExternCall(f'{expr.module}:{expr.func}', args)

        else:
            return HLILConst(f'<{type(expr).__name__}>')


def convert_mlil_to_hlil(mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
    converter = MLILToHLILConverter(mlil_func)
    return converter.convert()

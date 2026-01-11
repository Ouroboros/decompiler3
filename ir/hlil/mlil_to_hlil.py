'''MLIL to HLIL Converter - SSA-based Graph Rewriting'''

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Union

from ir.mlil.mlil import *
from ir.mlil.mlil_types import MLILType, MLILTypeKind
from .hlil import *
from .structural_analysis import StructuralAnalyzer


# Operator mapping tables
_BINARY_OP_MAP = {
    MediumLevelILOperation.MLIL_ADD         : BinaryOp.ADD,
    MediumLevelILOperation.MLIL_SUB         : BinaryOp.SUB,
    MediumLevelILOperation.MLIL_MUL         : BinaryOp.MUL,
    MediumLevelILOperation.MLIL_DIV         : BinaryOp.DIV,
    MediumLevelILOperation.MLIL_MOD         : BinaryOp.MOD,
    MediumLevelILOperation.MLIL_AND         : BinaryOp.BIT_AND,
    MediumLevelILOperation.MLIL_OR          : BinaryOp.BIT_OR,
    MediumLevelILOperation.MLIL_XOR         : BinaryOp.BIT_XOR,
    MediumLevelILOperation.MLIL_SHL         : BinaryOp.SHL,
    MediumLevelILOperation.MLIL_SHR         : BinaryOp.SHR,
    MediumLevelILOperation.MLIL_LOGICAL_AND : BinaryOp.AND,
    MediumLevelILOperation.MLIL_LOGICAL_OR  : BinaryOp.OR,
    MediumLevelILOperation.MLIL_EQ          : BinaryOp.EQ,
    MediumLevelILOperation.MLIL_NE          : BinaryOp.NE,
    MediumLevelILOperation.MLIL_LT          : BinaryOp.LT,
    MediumLevelILOperation.MLIL_LE          : BinaryOp.LE,
    MediumLevelILOperation.MLIL_GT          : BinaryOp.GT,
    MediumLevelILOperation.MLIL_GE          : BinaryOp.GE,
}

_UNARY_OP_MAP = {
    MediumLevelILOperation.MLIL_NEG : UnaryOp.NEG,
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

# Negation map for comparison operators
_NEGATE_CMP_OP = {
    BinaryOp.EQ : BinaryOp.NE,
    BinaryOp.NE : BinaryOp.EQ,
    BinaryOp.LT : BinaryOp.GE,
    BinaryOp.LE : BinaryOp.GT,
    BinaryOp.GT : BinaryOp.LE,
    BinaryOp.GE : BinaryOp.LT,
}


def _negate_condition(cond: HLILExpression) -> HLILExpression:
    '''Negate a condition, simplifying where possible'''
    # Double negation: !!a -> a
    if isinstance(cond, HLILUnaryOp) and cond.op == UnaryOp.NOT:
        return cond.operand

    # Comparison negation: !(a == b) -> a != b
    if isinstance(cond, HLILBinaryOp) and cond.op in _NEGATE_CMP_OP:
        return HLILBinaryOp(_NEGATE_CMP_OP[cond.op], cond.lhs, cond.rhs)

    # Default: wrap with NOT
    return HLILUnaryOp(UnaryOp.NOT, cond)


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
    is_global: bool = False  # True for GLOBALS, False for REGS
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

    Optimization:
      Each REGS[n]/GLOBALS[n] write creates a local variable.
      All reads before the next write use that local variable.
    '''

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func

        # Definitions: (block_idx, instr_idx) -> SSADef
        self.definitions: Dict[Tuple[int, int], SSADef] = {}

        # Uses: (block_idx, instr_idx, type, index) -> SSAUse
        # type is 'reg' or 'global'
        self.uses: Dict[Tuple[int, int, str, int], SSAUse] = {}

        # Rewiring map: (block_idx, instr_idx, type, index) -> SSADef
        self.rewired: Dict[Tuple[int, int, str, int], SSADef] = {}

        # Deleted instructions
        self.deleted: Set[Tuple[int, int]] = set()

        # CFG info
        self.predecessors: Dict[int, List[int]] = {}
        self.successors: Dict[int, List[int]] = {}

        # Reaching definitions at block entry: block_idx -> {var_key -> Set[SSADef]}
        # var_key is ('reg', index) or ('global', index)
        self.reaching_defs_entry: Dict[int, Dict[Tuple[str, int], Set[SSADef]]] = {}

        # Local variable names for REGS/GLOBALS optimization
        # (block_idx, instr_idx) -> var_name
        self.def_var_names: Dict[Tuple[int, int], str] = {}
        self._var_counters: Dict[str, int] = {}  # base_name -> counter

    def analyze(self):
        '''Run the full analysis and rewriting pipeline'''
        self._build_cfg()
        self._phase_a_analysis()
        self._phase_b_rewriting()
        self._phase_c_dce()
        self._assign_var_names()

    def _gen_var_name(self, base: str) -> str:
        '''Generate unique variable name'''
        count = self._var_counters.get(base, 0)
        self._var_counters[base] = count + 1
        if count == 0:
            return base
        return f'{base}_v{count}'

    def _assign_var_names(self):
        '''Assign local variable names to REGS/GLOBALS definitions that have uses'''
        for key, ssa_def in self.definitions.items():
            if ssa_def.use_count == 0 and not ssa_def.is_live_out:
                continue  # Dead, no var needed

            # Generate var name based on type
            if ssa_def.is_global:
                base = f'global{ssa_def.reg_index}'

            elif ssa_def.is_call:
                base = 'reg0'

            elif ssa_def.reg_index >= 0:
                base = f'reg{ssa_def.reg_index}'

            else:
                continue  # Not a register/global def

            self.def_var_names[key] = self._gen_var_name(base)

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
                    # Skip removed blocks
                    if succ in self.predecessors:
                        self.successors[i].append(succ)
                        self.predecessors[succ].append(i)
                if last_instr.false_target is not None:
                    succ = last_instr.false_target.index
                    # Skip removed blocks
                    if succ in self.predecessors:
                        self.successors[i].append(succ)
                        self.predecessors[succ].append(i)

            elif isinstance(last_instr, MLILGoto):
                if last_instr.target is not None:
                    succ = last_instr.target.index
                    # Skip removed blocks
                    if succ in self.predecessors:
                        self.successors[i].append(succ)
                        self.predecessors[succ].append(i)

            elif not isinstance(last_instr, MLILRet):
                if i + 1 < len(self.mlil_func.basic_blocks):
                    self.successors[i].append(i + 1)
                    self.predecessors[i + 1].append(i)

    def _phase_a_analysis(self):
        '''Phase A: Build def-use chains, compute liveness'''

        # Pass 1: Collect all definitions and compute local exit defs
        # Key format: ('reg', index) or ('global', index)
        local_exit_defs: Dict[int, Dict[Tuple[str, int], SSADef]] = {}

        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            current_def: Dict[Tuple[str, int], SSADef] = {}

            for instr_idx, instr in enumerate(block.instructions):
                key = (block_idx, instr_idx)

                # Record definitions
                if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                    ssa_def = SSADef(block_idx, instr_idx, 0, instr, is_call=True)
                    self.definitions[key] = ssa_def
                    current_def[('reg', 0)] = ssa_def

                elif isinstance(instr, MLILStoreReg):
                    is_call = isinstance(instr.value, (MLILCall, MLILSyscall, MLILCallScript))
                    ssa_def = SSADef(block_idx, instr_idx, instr.index, instr, is_call=is_call)
                    self.definitions[key] = ssa_def
                    current_def[('reg', instr.index)] = ssa_def

                elif isinstance(instr, MLILStoreGlobal):
                    ssa_def = SSADef(block_idx, instr_idx, instr.index, instr, is_call=False, is_global=True)
                    self.definitions[key] = ssa_def
                    current_def[('global', instr.index)] = ssa_def

            local_exit_defs[block_idx] = current_def

        # Pass 2: Compute reaching definitions (forward dataflow)
        self._compute_reaching_definitions(local_exit_defs)

        # Pass 3: Link uses to definitions (with cross-block support)
        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            # Start with reaching defs at block entry
            current_def: Dict[Tuple[str, int], SSADef] = {}
            entry_defs = self.reaching_defs_entry.get(block_idx, {})
            for var_key, defs in entry_defs.items():
                # If multiple defs reach, we can't inline (set to None marker)
                if len(defs) == 1:
                    current_def[var_key] = next(iter(defs))

            for instr_idx, instr in enumerate(block.instructions):
                key = (block_idx, instr_idx)

                # Count uses with current reaching defs
                self._count_uses(instr, block_idx, instr_idx, current_def)

                # Update current_def after processing uses
                if key in self.definitions:
                    ssa_def = self.definitions[key]
                    var_key = ('global', ssa_def.reg_index) if ssa_def.is_global else ('reg', ssa_def.reg_index)
                    current_def[var_key] = ssa_def

        # Pass 4: Liveness analysis - mark live-out registers before RETURN
        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            for instr_idx, instr in enumerate(block.instructions):
                if isinstance(instr, MLILRet):
                    self._mark_live_out_before(block_idx, instr_idx)

        # Pass 5: Mark defs as live-out if they reach successor blocks
        self._mark_cross_block_live_out(local_exit_defs)

    def _compute_reaching_definitions(self, local_exit_defs: Dict[int, Dict[Tuple[str, int], SSADef]]):
        '''Forward dataflow: compute reaching definitions at each block entry'''
        num_blocks = len(self.mlil_func.basic_blocks)

        # Use (block_idx, instr_idx) tuples as def identifiers (hashable)
        # exit_defs: block_idx -> {var_key -> Set[(block_idx, instr_idx)]}
        exit_defs: Dict[int, Dict[Tuple[str, int], Set[Tuple[int, int]]]] = {}
        entry_defs: Dict[int, Dict[Tuple[str, int], Set[Tuple[int, int]]]] = {}

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
                new_entry: Dict[Tuple[str, int], Set[Tuple[int, int]]] = {}
                for pred_idx in self.predecessors.get(block_idx, []):
                    for var_key, def_keys in exit_defs.get(pred_idx, {}).items():
                        if var_key not in new_entry:
                            new_entry[var_key] = set()
                        new_entry[var_key] |= def_keys

                # Exit = local defs override entry
                new_exit: Dict[Tuple[str, int], Set[Tuple[int, int]]] = {k: set(v) for k, v in new_entry.items()}
                if block_idx in local_exit_defs:
                    for var_key, ssa_def in local_exit_defs[block_idx].items():
                        new_exit[var_key] = {(ssa_def.block_idx, ssa_def.instr_idx)}

                # Check for changes
                if new_exit != exit_defs.get(block_idx, {}):
                    exit_defs[block_idx] = new_exit
                    changed = True

                entry_defs[block_idx] = new_entry

        # Convert entry_defs to reaching_defs_entry with SSADef references
        for block_idx, var_defs in entry_defs.items():
            self.reaching_defs_entry[block_idx] = {}
            for var_key, def_keys in var_defs.items():
                defs = set()
                for def_key in def_keys:
                    if def_key in self.definitions:
                        defs.add(self.definitions[def_key])
                if defs:
                    self.reaching_defs_entry[block_idx][var_key] = defs

    def _mark_cross_block_live_out(self, local_exit_defs: Dict[int, Dict[Tuple[str, int], SSADef]]):
        '''Mark definitions as live-out if they may be used in successor blocks'''
        # Collect which vars each block uses at entry (before any local def)
        block_entry_uses: Dict[int, Set[Tuple[str, int]]] = {}
        for block_idx, block in enumerate(self.mlil_func.basic_blocks):
            entry_uses: Set[Tuple[str, int]] = set()
            local_defs: Set[Tuple[str, int]] = set()

            for instr in block.instructions:
                # Collect uses of vars not yet defined locally
                self._collect_var_uses(instr, entry_uses, local_defs)
                # Collect local definitions
                if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                    local_defs.add(('reg', 0))

                elif isinstance(instr, MLILStoreReg):
                    local_defs.add(('reg', instr.index))

                elif isinstance(instr, MLILStoreGlobal):
                    local_defs.add(('global', instr.index))

            block_entry_uses[block_idx] = entry_uses

        # Mark defs as live-out if successor uses them
        for block_idx in range(len(self.mlil_func.basic_blocks)):
            if block_idx not in local_exit_defs:
                continue

            for succ_idx in self.successors.get(block_idx, []):
                entry_uses = block_entry_uses.get(succ_idx, set())
                for var_key in entry_uses:
                    if var_key in local_exit_defs[block_idx]:
                        local_exit_defs[block_idx][var_key].is_live_out = True

    def _collect_var_uses(self, instr: MediumLevelILInstruction,
                          entry_uses: Set[Tuple[str, int]], local_defs: Set[Tuple[str, int]]):
        '''Collect REGS/GLOBALS uses that are not locally defined'''
        def visit(expr):
            if isinstance(expr, MLILLoadReg):
                var_key = ('reg', expr.index)
                if var_key not in local_defs:
                    entry_uses.add(var_key)

            elif isinstance(expr, MLILLoadGlobal):
                var_key = ('global', expr.index)
                if var_key not in local_defs:
                    entry_uses.add(var_key)

            elif isinstance(expr, MLILBinaryOp):
                visit(expr.lhs)
                visit(expr.rhs)

            elif isinstance(expr, MLILUnaryOp):
                visit(expr.operand)

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

            elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in expr.args:
                    visit(arg)

        visit(instr)

    def _count_uses(self, instr: MediumLevelILInstruction,
                    block_idx: int, instr_idx: int,
                    current_def: Dict[Tuple[str, int], SSADef]):
        '''Count REGS/GLOBALS uses in an instruction'''

        def visit(expr):
            if isinstance(expr, MLILLoadReg):
                var_key = ('reg', expr.index)
                ssa_def = current_def.get(var_key)
                if ssa_def is not None:
                    ssa_def.use_count += 1
                    use_key = (block_idx, instr_idx, 'reg', expr.index)
                    self.uses[use_key] = SSAUse(block_idx, instr_idx, ssa_def)

            elif isinstance(expr, MLILLoadGlobal):
                var_key = ('global', expr.index)
                ssa_def = current_def.get(var_key)
                if ssa_def is not None:
                    ssa_def.use_count += 1
                    use_key = (block_idx, instr_idx, 'global', expr.index)
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
                            use_k[2] == 'reg' and
                            use_k[3] == instr.value.index):
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

    def get_inlined_expr(self, block_idx: int, instr_idx: int,
                         var_type: str, index: int) -> Optional[SSADef]:
        '''Get the definition to inline for a LoadReg/LoadGlobal'''
        use_key = (block_idx, instr_idx, var_type, index)

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

    def get_def_var_name(self, block_idx: int, instr_idx: int) -> Optional[str]:
        '''Get the local variable name for a definition'''
        return self.def_var_names.get((block_idx, instr_idx))

    def get_use_var_name(self, block_idx: int, instr_idx: int,
                         var_type: str, index: int) -> Optional[str]:
        '''Get the local variable name for a use site (LoadReg/LoadGlobal)'''
        use_key = (block_idx, instr_idx, var_type, index)

        # Find the definition this use refers to
        def_ref = None
        if use_key in self.rewired:
            def_ref = self.rewired[use_key]

        elif use_key in self.uses:
            def_ref = self.uses[use_key].def_ref

        if def_ref is None:
            return None

        def_key = (def_ref.block_idx, def_ref.instr_idx)
        return self.def_var_names.get(def_key)

    def get_return_var_name(self, block_idx: int, instr_idx: int) -> Optional[str]:
        '''Get the return value variable name at a return instruction'''
        # Find the last definition of REG[0] before this return
        block = self.mlil_func.basic_blocks[block_idx]
        for i in range(instr_idx - 1, -1, -1):
            key = (block_idx, i)
            if key in self.definitions:
                ssa_def = self.definitions[key]
                # REG[0] is the return value register (not GLOBALS)
                if ssa_def.reg_index == 0 and not ssa_def.is_global:
                    return self.def_var_names.get(key)

        return None


# ============================================================================
# MLIL to HLIL Converter
# ============================================================================

class MLILToHLILConverter:

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = HighLevelILFunction(mlil_func.name, mlil_func.start_addr, is_common_func=mlil_func.is_common_func)

        self.block_successors: Dict[int, List[int]] = {}
        self.visited_blocks: Set[int] = set()
        self.globally_processed: Set[int] = set()

        # Loop detection
        self.loop_headers: Set[int] = set()  # Blocks that are loop headers
        self.loop_ends: Dict[int, int] = {}  # back_edge_source -> loop_header

        # Structural analyzer for accurate merge point detection
        self.structural_analyzer: Optional[StructuralAnalyzer] = None

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

        # Phase 3: Build CFG and detect loops
        self._build_cfg()
        self._detect_loops()

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

        for i, mlil_var in enumerate(self.mlil_func.parameters):
            if mlil_var is None:
                continue

            type_hint = self._get_type_hint(mlil_var.name)

            # Get default value from source_params
            default_value = None
            if i < len(self.mlil_func.source_params):
                default_value = self.mlil_func.source_params[i].default_value

            hlil_var = HLILVariable(mlil_var.name, type_hint=type_hint,
                                    default_value=default_value, kind=VariableKind.PARAM)
            self.hlil_func.parameters.append(hlil_var)

        for mlil_var in self.mlil_func.locals.values():
            if mlil_var.name not in used_vars:
                continue
            type_hint = self._get_type_hint(mlil_var.name)
            hlil_var = HLILVariable(mlil_var.name, type_hint=type_hint)
            self.hlil_func.variables.append(hlil_var)

        # Add REGS/GLOBALS local variables (they hold numeric values)
        added_var_names: Set[str] = set()
        for var_name in self.resolver.def_var_names.values():
            if var_name not in added_var_names:
                added_var_names.add(var_name)
                hlil_var = HLILVariable(var_name, type_hint=HLILTypeKind.INT)
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

        # Initialize structural analyzer with the built CFG
        num_blocks = len(self.mlil_func.basic_blocks)
        if num_blocks > 0:
            self.structural_analyzer = StructuralAnalyzer(num_blocks, self.block_successors)

    def _detect_loops(self):
        '''Detect loops using structural analyzer'''
        if not self.mlil_func.basic_blocks or self.structural_analyzer is None:
            return

        # Use structural analyzer's loop detection
        for header, loop_info in self.structural_analyzer.loops.items():
            self.loop_headers.add(header)
            # Record back edges
            for tail, _ in loop_info.back_edges:
                self.loop_ends[tail] = header

    def _get_reachable(self, start_idx: int) -> Dict[int, int]:
        reachable = {}
        queue = deque([(start_idx, 0)])
        while queue:
            idx, depth = queue.popleft()
            if idx in reachable:
                continue
            reachable[idx] = depth
            for succ in self.block_successors.get(idx, []):
                if succ not in reachable:
                    queue.append((succ, depth + 1))
        return reachable

    # Set to True to use legacy reachability-based merge point detection
    USE_LEGACY_MERGE_DETECTION = False

    def _find_merge_block(self, cond_block_idx: int, true_target_idx: int, false_target_idx: int) -> Optional[int]:
        '''Find merge point for if-else using structural analyzer.'''
        if true_target_idx == false_target_idx:
            return true_target_idx

        if self.USE_LEGACY_MERGE_DETECTION:
            return self._find_merge_block_legacy(true_target_idx, false_target_idx)

        # Use structural analyzer
        if self.structural_analyzer is not None:
            return self.structural_analyzer.find_merge_point(
                cond_block_idx, true_target_idx, false_target_idx
            )

        return None

    def _find_merge_block_legacy(self, true_target_idx: int, false_target_idx: int) -> Optional[int]:
        '''
        Legacy: reachability-based heuristic for merge point detection.
        Kept for debugging/comparison. Enable via USE_LEGACY_MERGE_DETECTION.
        '''
        succ1 = set(self.block_successors.get(true_target_idx, []))
        succ2 = set(self.block_successors.get(false_target_idx, []))

        # Check if one branch directly targets the other
        if true_target_idx in succ2:
            return true_target_idx

        if false_target_idx in succ1:
            return false_target_idx

        # Check common immediate successors
        common = succ1 & succ2
        if common:
            return min(common)

        # Use reachability analysis
        reachable1 = self._get_reachable(true_target_idx)
        reachable2 = self._get_reachable(false_target_idx)
        common_blocks = set(reachable1.keys()) & set(reachable2.keys())

        # Check indirect merge
        if true_target_idx in reachable2:
            return true_target_idx

        if false_target_idx in reachable1:
            return false_target_idx

        # Find closest common reachable block
        common_blocks.discard(true_target_idx)
        common_blocks.discard(false_target_idx)
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
            all(is_nop_instr(instr) for instr in block.instructions[:-1]) and
            isinstance(block.instructions[-1], MLILGoto)
        )

    def _process_loop(self, header_idx: int, target_block: HLILBlock,
                      stop_at: Optional[int]) -> Optional[int]:
        '''Process a loop starting at header_idx'''
        self.visited_blocks.add(header_idx)
        self.globally_processed.add(header_idx)

        mlil_block = self.mlil_func.basic_blocks[header_idx]
        last_instr = mlil_block.instructions[-1] if mlil_block.instructions else None

        # Find the exit block (where loop exits to)
        exit_block_idx = None
        loop_body_start = None

        if isinstance(last_instr, MLILIf):
            # while (cond) pattern: if (cond) goto exit else body
            # or: if (!cond) goto body else exit
            true_target = last_instr.true_target.index if last_instr.true_target else None
            false_target = last_instr.false_target.index if last_instr.false_target else None

            # Determine which branch is the loop body and which is exit
            # The exit branch leads outside the loop (not back to header)
            true_reaches_header = self._can_reach(true_target, header_idx, set())
            false_reaches_header = self._can_reach(false_target, header_idx, set())

            if true_reaches_header and not false_reaches_header:
                # true branch is loop body, false is exit
                loop_body_start = true_target
                exit_block_idx = false_target
                condition = self._convert_expr(last_instr.condition)

            elif false_reaches_header and not true_reaches_header:
                # false branch is loop body, true is exit
                loop_body_start = false_target
                exit_block_idx = true_target
                condition = _negate_condition(self._convert_expr(last_instr.condition))

            else:
                # Both or neither reach header - complex loop, use while(true)
                condition = HLILConst(1)
                loop_body_start = true_target
                exit_block_idx = None

        else:
            # Infinite loop: while (true)
            condition = HLILConst(1)
            if isinstance(last_instr, MLILGoto) and last_instr.target:
                loop_body_start = last_instr.target.index

        # Process instructions before the terminator in header block
        self.current_block_idx = header_idx
        for instr_idx, instr in enumerate(mlil_block.instructions[:-1]):
            self.current_instr_idx = instr_idx
            stmts = self._convert_instruction(instr, header_idx, instr_idx)
            for stmt in stmts:
                target_block.add_statement(stmt)

        # Create loop body
        loop_body = HLILBlock()
        if loop_body_start is not None:
            self._reconstruct_control_flow(loop_body_start, loop_body, stop_at=header_idx)

        # Create while statement
        while_stmt = HLILWhile(condition, loop_body)
        target_block.add_statement(while_stmt)

        # Process code after loop (exit block)
        if exit_block_idx is not None:
            self.visited_blocks.discard(exit_block_idx)
            return self._reconstruct_control_flow(exit_block_idx, target_block, stop_at=stop_at)

        return None

    def _can_reach(self, from_idx: Optional[int], to_idx: int, visited: Set[int]) -> bool:
        '''Check if from_idx can reach to_idx through CFG without crossing other loop headers'''
        if from_idx is None:
            return False

        stack = [from_idx]
        while stack:
            idx = stack.pop()
            if idx == to_idx:
                return True
            if idx in visited:
                continue
            # Don't cross other loop headers (except the target)
            if idx in self.loop_headers and idx != to_idx:
                continue
            visited.add(idx)
            stack.extend(self.block_successors.get(idx, []))

        return False

    def _process_if_statement(self, if_instr: MLILIf, block_idx: int,
                               target_block: HLILBlock, stop_at: Optional[int]) -> Optional[int]:
        '''Process an if statement, detecting and handling else-if chains'''
        condition = self._convert_expr(if_instr.condition)
        true_target_idx = if_instr.true_target.index if if_instr.true_target else None
        false_target_idx = if_instr.false_target.index if if_instr.false_target else None

        # Handle degenerate case: if (C) goto A else A
        # Both branches go to same block. Condition C may have side effects,
        # so we generate: if (C || true) { A } to preserve C's evaluation
        if true_target_idx == false_target_idx and true_target_idx is not None:
            always_true = HLILBinaryOp(BinaryOp.OR, condition, HLILConst(1))
            body = HLILBlock()
            self._reconstruct_control_flow(true_target_idx, body, stop_at=stop_at)
            if_stmt = HLILIf(always_true, body, None)
            target_block.add_statement(HLILComment(f'if (C || true) {{ A }}'))
            target_block.add_statement(if_stmt)
            return None

        # Find merge block for the entire if/else-if chain
        merge_block_idx = None
        if true_target_idx is not None and false_target_idx is not None:
            merge_block_idx = self._find_merge_block(block_idx, true_target_idx, false_target_idx)

        branch_stop = merge_block_idx if merge_block_idx is not None else stop_at
        saved_visited = self.visited_blocks.copy()

        # Check if one branch is empty (goes directly to merge)
        true_is_empty = merge_block_idx == true_target_idx
        false_is_empty = merge_block_idx == false_target_idx

        # Track if we detected an else-if pattern (skip early return handling for these)
        is_else_if_pattern = False

        # Check if else-if chain is through true_target (inverted condition pattern)
        # e.g., switch-case: if (!match) goto next_check else case_body
        # Use structural analysis to detect: true_target has 2 successors (condition block)
        # Don't invert if true_target is already the merge (empty true branch)
        if (self.structural_analyzer is not None and
            true_target_idx is not None and false_target_idx is not None and
            not true_is_empty and
            self.structural_analyzer.should_invert_condition(true_target_idx, false_target_idx)):
            # Swap branches and negate condition
            condition = _negate_condition(condition)
            old_true_target = true_target_idx  # Save for checking merge block conflict
            true_target_idx, false_target_idx = false_target_idx, true_target_idx
            true_is_empty, false_is_empty = false_is_empty, False  # else-if block is never empty
            is_else_if_pattern = True
            # Merge block conflict: if merge_block equals the old true_target (now false),
            # processing false branch would stop immediately. Use outer stop_at instead.
            if merge_block_idx == old_true_target:
                branch_stop = stop_at

        # Process true branch (skip if empty - it's just the merge point)
        # MLIL: if (C) goto true_target else false_target
        # C true -> true_target, C false -> false_target
        true_block = HLILBlock()
        if true_target_idx is not None and not true_is_empty:
            self.visited_blocks = saved_visited.copy()
            # Add outer stop_at to visited, but not if it's our true_target
            if stop_at is not None and stop_at != merge_block_idx and stop_at != true_target_idx:
                self.visited_blocks.add(stop_at)
            self._reconstruct_control_flow(true_target_idx, true_block, stop_at=branch_stop)

        all_visited = self.visited_blocks.copy()

        # Check if false_target is an else-if block (skip if empty)
        # Structural check: false_target is a condition block if it has 2 successors
        # Exclude loop headers - they have 2 successors but are not else-if blocks
        false_is_condition = (
            self.structural_analyzer is not None and
            len(self.structural_analyzer.original_successors.get(false_target_idx, [])) == 2 and
            not self.structural_analyzer.is_loop_header(false_target_idx)
        )
        false_block = HLILBlock()
        if false_target_idx is not None and not false_is_empty and false_is_condition:
            # Else-if chain: recursively build nested if structure
            self.visited_blocks = saved_visited.copy()
            if stop_at is not None and stop_at != merge_block_idx and stop_at != false_target_idx:
                self.visited_blocks.add(stop_at)
            self._reconstruct_control_flow(false_target_idx, false_block, stop_at=branch_stop)
            all_visited |= self.visited_blocks

        elif false_target_idx is not None and not false_is_empty:
            # Normal else branch
            self.visited_blocks = saved_visited.copy()
            if stop_at is not None and stop_at != merge_block_idx and stop_at != false_target_idx:
                self.visited_blocks.add(stop_at)
            self._reconstruct_control_flow(false_target_idx, false_block, stop_at=branch_stop)
            all_visited |= self.visited_blocks

        self.visited_blocks = all_visited
        if stop_at is not None:
            self.visited_blocks.discard(stop_at)

        # Check if branches end with return (skip for else-if patterns)
        true_ends_with_return = (not is_else_if_pattern and true_block.statements and
            isinstance(true_block.statements[-1], HLILReturn))
        false_ends_with_return = (not is_else_if_pattern and false_block.statements and
            isinstance(false_block.statements[-1], HLILReturn))

        # MLIL: if (C) goto true_target else false_target
        # HLIL: if (C) { true_block } else { false_block }

        # Early return patterns (skip for else-if chains to preserve structure)
        if true_ends_with_return and false_ends_with_return:
            if len(true_block.statements) <= len(false_block.statements):
                if_stmt = HLILIf(condition, true_block, None)
                target_block.add_statement(if_stmt)
                for stmt in false_block.statements:
                    target_block.add_statement(stmt)

            else:
                if_stmt = HLILIf(_negate_condition(condition), false_block, None)
                target_block.add_statement(if_stmt)
                for stmt in true_block.statements:
                    target_block.add_statement(stmt)

        elif true_ends_with_return:
            if_stmt = HLILIf(condition, true_block, None)
            target_block.add_statement(if_stmt)
            for stmt in false_block.statements:
                target_block.add_statement(stmt)

        elif false_ends_with_return:
            if_stmt = HLILIf(_negate_condition(condition), false_block, None)
            target_block.add_statement(if_stmt)
            for stmt in true_block.statements:
                target_block.add_statement(stmt)

        else:
            # MLIL: if (C) goto true_target else false_target
            # HLIL: if (C) { true_block } else { false_block }
            # Handle empty branches: negate condition if true branch is empty
            if not true_block.statements and false_block.statements:
                # true branch is empty, negate and use false as body
                if_stmt = HLILIf(_negate_condition(condition), false_block, None)

            elif true_block.statements and not false_block.statements:
                # false branch is empty, use true as body
                if_stmt = HLILIf(condition, true_block, None)

            else:
                if_stmt = HLILIf(condition, true_block, false_block if false_block.statements else None)

            target_block.add_statement(if_stmt)

        # Process merge block
        if merge_block_idx is not None:
            if stop_at is not None and merge_block_idx == stop_at:
                return merge_block_idx
            self.visited_blocks.discard(merge_block_idx)
            return self._reconstruct_control_flow(merge_block_idx, target_block, stop_at=stop_at)

        return None

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

        # Skip already processed blocks
        if block_idx in self.visited_blocks or block_idx in self.globally_processed:
            return None
        if stop_at is not None and block_idx == stop_at:
            # Reached the merge point - stop here, don't process this block
            # The merge block will be processed after the if-else by the outer scope
            self.visited_blocks.add(block_idx)
            return block_idx

        # Check if this is a loop header
        if block_idx in self.loop_headers:
            return self._process_loop(block_idx, target_block, stop_at)

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
                return self._process_if_statement(
                    last_instr, block_idx, target_block, stop_at
                )

            elif isinstance(last_instr, MLILGoto):
                if last_instr.target is not None:
                    target_idx = last_instr.target.index
                    # Check if this is a back edge (loop continue)
                    if target_idx in self.loop_headers and target_idx in self.globally_processed:
                        # Back edge - loop continues naturally
                        return None
                    return self._reconstruct_control_flow(target_idx, target_block, stop_at=stop_at)
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

    def _set_hlil_source_info(self, hlil_instr: HLILInstruction,
                               mlil_instr: MediumLevelILInstruction,
                               mlil_index: int) -> None:
        '''Propagate source address info from MLIL to HLIL'''
        hlil_instr.address = mlil_instr.address
        hlil_instr.mlil_index = mlil_index

    def _convert_instruction(self, instr: MediumLevelILInstruction,
                              block_idx: int, instr_idx: int) -> List[HLILStatement]:
        '''Convert instruction, may return multiple statements for temp vars'''
        key = (block_idx, instr_idx)
        result = []

        # Get global instruction index for MLIL
        mlil_index = instr.inst_index

        # Check if deleted by DCE
        if self.resolver.is_deleted(block_idx, instr_idx):
            return result

        if isinstance(instr, MLILDebug):
            stmt = HLILComment(f'{instr.debug_type}({instr.value})')
            self._set_hlil_source_info(stmt, instr, mlil_index)
            result.append(stmt)

        elif isinstance(instr, MLILNop):
            pass

        elif isinstance(instr, MLILRet):
            if instr.value:
                stmt = HLILReturn(self._convert_expr(instr.value))

            else:
                # Check if there's a _reg0 variable as the return value
                ret_var = self.resolver.get_return_var_name(block_idx, instr_idx)
                if ret_var is not None:
                    stmt = HLILReturn(HLILVar(HLILVariable(ret_var, None)))

                else:
                    stmt = HLILReturn()

            self._set_hlil_source_info(stmt, instr, mlil_index)
            result.append(stmt)

        elif isinstance(instr, MLILSetVar):
            stmt = HLILAssign(
                HLILVar(HLILVariable(instr.var.name, None)),
                self._convert_expr(instr.value)
            )
            self._set_hlil_source_info(stmt, instr, mlil_index)
            result.append(stmt)

        elif isinstance(instr, MLILStoreReg):
            # Use local variable if has uses, otherwise skip (dead store)
            var_name = self.resolver.get_def_var_name(block_idx, instr_idx)
            if var_name is not None:
                var = HLILVariable(var_name, None)
                stmt = HLILAssign(HLILVar(var), self._convert_expr(instr.value))
                self._set_hlil_source_info(stmt, instr, mlil_index)
                result.append(stmt)
                # Cache for single-use inlining
                self.expr_cache[key] = self._convert_expr(instr.value)

        elif isinstance(instr, MLILStoreGlobal):
            # Use local variable if has uses, otherwise skip (dead store)
            var_name = self.resolver.get_def_var_name(block_idx, instr_idx)
            if var_name is not None:
                var = HLILVariable(var_name, None)
                stmt = HLILAssign(HLILVar(var), self._convert_expr(instr.value))
                self._set_hlil_source_info(stmt, instr, mlil_index)
                result.append(stmt)

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
                stmt = HLILExprStmt(call_expr)
                self._set_hlil_source_info(stmt, instr, mlil_index)
                result.append(stmt)

            else:
                # Multiple uses or live-out: assign to local variable
                var_name = self.resolver.get_def_var_name(block_idx, instr_idx)
                if var_name is not None:
                    var = HLILVariable(var_name, None)
                    stmt = HLILAssign(HLILVar(var), call_expr)
                    self._set_hlil_source_info(stmt, instr, mlil_index)
                    result.append(stmt)

                else:
                    # No var name (shouldn't happen), emit as statement
                    stmt = HLILExprStmt(call_expr)
                    self._set_hlil_source_info(stmt, instr, mlil_index)
                    result.append(stmt)

        elif isinstance(instr, (MLILIf, MLILGoto)):
            pass

        elif isinstance(instr, MediumLevelILExpr):
            stmt = HLILExprStmt(self._convert_expr(instr))
            self._set_hlil_source_info(stmt, instr, mlil_index)
            result.append(stmt)

        return result

    def _convert_expr(self, expr: MediumLevelILExpr) -> HLILExpression:
        if isinstance(expr, MLILVar):
            return HLILVar(HLILVariable(expr.var.name, None))

        elif isinstance(expr, MLILConst):
            return HLILConst(expr.value, expr.is_hex)

        elif isinstance(expr, MLILBinaryOp):
            op = _BINARY_OP_MAP.get(expr.operation)
            if op is None:
                raise ValueError(f'Unknown binary op: {expr.operation}')
            return HLILBinaryOp(op, self._convert_expr(expr.lhs), self._convert_expr(expr.rhs))

        elif isinstance(expr, MLILAddressOf):
            operand = self._convert_expr(expr.operand)
            return HLILAddressOf(operand)

        elif isinstance(expr, MLILUnaryOp):
            if expr.operation in (MediumLevelILOperation.MLIL_LOGICAL_NOT, MediumLevelILOperation.MLIL_TEST_ZERO):
                operand = self._convert_expr(expr.operand)
                return HLILBinaryOp(BinaryOp.EQ, operand, HLILConst(0))
            op = _UNARY_OP_MAP.get(expr.operation)
            if op is None:
                raise ValueError(f'Unknown unary op: {expr.operation}')
            return HLILUnaryOp(op, self._convert_expr(expr.operand))

        elif isinstance(expr, MLILLoadReg):
            # Check if we can inline a call result (single use)
            inlined_def = self.resolver.get_inlined_expr(
                self.current_block_idx, self.current_instr_idx, 'reg', expr.index
            )

            if inlined_def is not None:
                def_key = (inlined_def.block_idx, inlined_def.instr_idx)

                # Inline single-use calls
                if inlined_def.is_call and def_key in self.expr_cache:
                    return self.expr_cache[def_key]

            # Check if there's a local variable for this use
            var_name = self.resolver.get_use_var_name(
                self.current_block_idx, self.current_instr_idx, 'reg', expr.index
            )
            if var_name is not None:
                return HLILVar(HLILVariable(var_name, None))

            # Default: emit REGS[n]
            var = HLILVariable(kind=VariableKind.REG, index=expr.index)
            return HLILVar(var)

        elif isinstance(expr, MLILLoadGlobal):
            # Check if there's a local variable for this use
            var_name = self.resolver.get_use_var_name(
                self.current_block_idx, self.current_instr_idx, 'global', expr.index
            )
            if var_name is not None:
                return HLILVar(HLILVariable(var_name, None))

            # Default: emit GLOBALS[n]
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

'''MLIL to HLIL Converter - goto/label to if/while/switch'''

from collections import deque
from typing import Dict, List, Set, Optional

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
    MLILTypeKind.BOOL     : HLILTypeKind.INT,      # Bool is represented as int
    MLILTypeKind.POINTER  : HLILTypeKind.INT,      # Pointer is represented as int
    MLILTypeKind.VARIANT  : HLILTypeKind.UNKNOWN,
    MLILTypeKind.VOID     : HLILTypeKind.VOID,
}


def _mlil_type_to_hlil(mlil_type: MLILType) -> HLILTypeKind:
    '''Convert MLIL type to HLIL type kind'''
    return _MLIL_TYPE_MAP[mlil_type.kind]


class MLILToHLILConverter:

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = HighLevelILFunction(mlil_func.name, mlil_func.start_addr)

        self.block_successors: Dict[int, List[int]] = {}
        self.visited_blocks: Set[int] = set()
        self.globally_processed: Set[int] = set()  # Prevent duplicate processing across branches
        self.last_call: Optional[HLILExpression] = None

    def _uses_reg0(self, instr: MediumLevelILInstruction) -> bool:
        if isinstance(instr, MLILLoadReg) and instr.index == 0:
            return True

        if isinstance(instr, MLILBinaryOp):
            return self._uses_reg0(instr.lhs) or self._uses_reg0(instr.rhs)

        elif isinstance(instr, MLILUnaryOp):
            return self._uses_reg0(instr.operand)

        elif isinstance(instr, MLILAddressOf):
            return self._uses_reg0(instr.operand)

        elif isinstance(instr, (MLILSetVar, MLILStoreGlobal, MLILStoreReg)):
            return self._uses_reg0(instr.value)

        elif isinstance(instr, MLILIf):
            return self._uses_reg0(instr.condition)

        elif isinstance(instr, MLILRet):
            if instr.value:
                return self._uses_reg0(instr.value)

        elif isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
            # Check arguments
            return any(self._uses_reg0(arg) for arg in instr.args)

        return False

    def convert(self) -> HighLevelILFunction:
        # Phase 1: Convert variables
        self._convert_variables()

        # Phase 2: Build control flow graph
        self._build_cfg()

        # Phase 3: Reconstruct structured control flow
        if self.mlil_func.basic_blocks:
            self._reconstruct_control_flow(0, self.hlil_func.body)

        return self.hlil_func

    def _convert_variables(self):
        '''Convert used MLIL variables to HLIL'''
        defined_vars = set()
        read_vars = set()

        def collect_read_vars(expr):
            '''Recursively collect variables read in expression'''
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

        # Scan all instructions
        for block in self.mlil_func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar):
                    defined_vars.add(inst.var.name)
                    collect_read_vars(inst.value)

                elif isinstance(inst, MLILIf):
                    collect_read_vars(inst.condition)

                elif isinstance(inst, MLILRet):
                    if inst.value:
                        collect_read_vars(inst.value)

                elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
                    for arg in inst.args:
                        collect_read_vars(arg)

                elif isinstance(inst, MLILStoreGlobal):
                    collect_read_vars(inst.value)

                elif isinstance(inst, MLILStoreReg):
                    collect_read_vars(inst.value)

        # Only keep variables that are both defined and read
        used_vars = defined_vars & read_vars

        # Convert parameters (already ordered in MLIL)
        for mlil_var in self.mlil_func.parameters:
            if mlil_var is None or mlil_var.name not in read_vars:
                continue

            type_hint = self._get_type_hint(mlil_var.name)
            hlil_var = HLILVariable(mlil_var.name, type_hint=type_hint, kind=VariableKind.PARAM)
            self.hlil_func.parameters.append(hlil_var)

        # Convert local variables
        for mlil_var in self.mlil_func.locals.values():
            if mlil_var.name not in used_vars:
                continue

            type_hint = self._get_type_hint(mlil_var.name)
            hlil_var = HLILVariable(mlil_var.name, type_hint=type_hint)
            self.hlil_func.variables.append(hlil_var)

    def _get_type_hint(self, var_name: str) -> Optional[HLILTypeKind]:
        '''Get HLIL type hint from MLIL var_types'''
        mlil_type = self.mlil_func.var_types.get(var_name)
        if mlil_type is None:
            return None

        hlil_type = _mlil_type_to_hlil(mlil_type)
        if hlil_type == HLILTypeKind.UNKNOWN:
            return None

        return hlil_type

    def _build_cfg(self):
        '''Build CFG from MLIL blocks'''
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
        '''BFS to get reachable blocks with their depths'''
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
        '''Find merge block where branches converge'''
        # Same target means immediate merge
        if block1_idx == block2_idx:
            return block1_idx

        # Fast path: check direct successors
        succ1 = set(self.block_successors.get(block1_idx, []))
        succ2 = set(self.block_successors.get(block2_idx, []))
        common = succ1 & succ2
        if common:
            return min(common)

        # Check if one branch reaches the other
        reachable1 = self._get_reachable(block1_idx)
        reachable2 = self._get_reachable(block2_idx)

        if block1_idx in reachable2:
            return block1_idx

        if block2_idx in reachable1:
            return block2_idx

        # Find common reachable block with minimum combined depth
        common_blocks = set(reachable1.keys()) & set(reachable2.keys())
        if not common_blocks:
            return None

        return min(common_blocks, key=lambda idx: reachable1[idx] + reachable2[idx])

    def _is_passthrough_block(self, block_idx: int) -> bool:
        '''Check if block is passthrough (only goto)'''
        if block_idx >= len(self.mlil_func.basic_blocks):
            return False

        block = self.mlil_func.basic_blocks[block_idx]
        if not block.instructions:
            return False

        # All non-terminator instructions must be debug/nop, last must be goto
        return (
            all(isinstance(instr, (MLILDebug, MLILNop)) for instr in block.instructions[:-1]) and
            isinstance(block.instructions[-1], MLILGoto)
        )

    def _reconstruct_control_flow(self, block_idx: int, target_block: HLILBlock, stop_at: Optional[int] = None) -> Optional[int]:
        '''Reconstruct structured control flow from goto/label'''
        # Skip passthrough blocks (with cycle detection, but not past stop_at)
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

        # Prevent duplicate processing across different branch paths
        if block_idx in self.globally_processed:
            return None

        # Stop if we reached the merge block
        if stop_at is not None and block_idx == stop_at:
            self.visited_blocks.add(block_idx)  # Mark as visited before returning
            return block_idx

        self.visited_blocks.add(block_idx)
        self.globally_processed.add(block_idx)
        mlil_block = self.mlil_func.basic_blocks[block_idx]

        # Convert all instructions except the last (terminator)
        instructions = mlil_block.instructions[:-1]
        for i, instr in enumerate(instructions):
            # Check if this is a call and next instruction uses REG[0]
            if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                # Check next instruction (or terminator)
                if i + 1 < len(instructions):
                    next_instr = instructions[i + 1]

                else:
                    next_instr = mlil_block.instructions[-1]  # terminator

                should_skip = False
                if self._uses_reg0(next_instr):
                    should_skip = True

                # Special case: if terminator is goto, check target block's first instruction
                elif isinstance(next_instr, MLILGoto) and next_instr.target:
                    target_block_mlil = self.mlil_func.basic_blocks[next_instr.target.index]
                    if target_block_mlil.instructions and self._uses_reg0(target_block_mlil.instructions[0]):
                        should_skip = True

                if should_skip:
                    call_expr = self._convert_expr(instr)
                    self.last_call = call_expr
                    continue  # Skip outputting this call

            stmt = self._convert_instruction(instr)
            if stmt:
                target_block.add_statement(stmt)

        # Handle terminator instruction
        if mlil_block.instructions:
            last_instr = mlil_block.instructions[-1]

            if isinstance(last_instr, MLILIf):
                # Create if statement
                condition = self._convert_expr(last_instr.condition)

                true_target_idx = last_instr.true_target.index if last_instr.true_target else None
                false_target_idx = last_instr.false_target.index if last_instr.false_target else None

                # Find merge block
                merge_block_idx = None
                if true_target_idx is not None and false_target_idx is not None:
                    merge_block_idx = self._find_merge_block(true_target_idx, false_target_idx)

                # Process branches (stop at merge block or parent's stop_at)
                true_block = HLILBlock()
                false_block = HLILBlock()

                # Use merge_block if found, otherwise inherit parent's stop_at
                branch_stop = merge_block_idx if merge_block_idx is not None else stop_at

                saved_visited = self.visited_blocks.copy()
                saved_last_call = self.last_call

                # Mark parent's stop_at as visited to prevent branches from processing it
                branch_visited_base = saved_visited.copy()
                if stop_at is not None and stop_at != merge_block_idx:
                    branch_visited_base.add(stop_at)

                if true_target_idx is not None:
                    self.visited_blocks = branch_visited_base.copy()
                    self.last_call = saved_last_call
                    self._reconstruct_control_flow(true_target_idx, true_block, stop_at=branch_stop)

                true_visited = self.visited_blocks
                # Remove parent's stop_at from true_visited (shouldn't pollute merge)
                if stop_at is not None:
                    true_visited = true_visited - {stop_at}

                if false_target_idx is not None:
                    # Include true_visited to prevent re-visiting blocks already processed
                    self.visited_blocks = branch_visited_base | true_visited
                    self.last_call = saved_last_call
                    self._reconstruct_control_flow(false_target_idx, false_block, stop_at=branch_stop)

                # Merge visited from both branches (exclude parent's stop_at)
                false_visited = self.visited_blocks
                if stop_at is not None:
                    false_visited = false_visited - {stop_at}
                self.visited_blocks = saved_visited | true_visited | false_visited
                self.last_call = None  # Clear after branches

                # Add if statement
                if_stmt = HLILIf(condition, true_block, false_block if false_block.statements else None)
                target_block.add_statement(if_stmt)

                # Continue with merge block
                if merge_block_idx is not None:
                    # Don't process merge block if it's our stop point (belongs to parent)
                    if stop_at is not None and merge_block_idx == stop_at:
                        return merge_block_idx
                    # Remove from visited (was marked during branch processing at stop_at)
                    self.visited_blocks.discard(merge_block_idx)
                    return self._reconstruct_control_flow(merge_block_idx, target_block, stop_at=stop_at)

                return None

            elif isinstance(last_instr, MLILGoto):
                # Process jump target (pass stop_at to respect parent's merge block)
                if last_instr.target is not None:
                    return self._reconstruct_control_flow(last_instr.target.index, target_block, stop_at=stop_at)
                return None

            elif isinstance(last_instr, MLILRet):
                # Convert return statement
                stmt = self._convert_instruction(last_instr)
                if stmt:
                    target_block.add_statement(stmt)
                return None

            else:
                # Process last instruction as statement
                stmt = self._convert_instruction(last_instr)
                if stmt:
                    target_block.add_statement(stmt)

                # Fall through to next block (pass stop_at to respect parent's merge block)
                if block_idx + 1 < len(self.mlil_func.basic_blocks):
                    return self._reconstruct_control_flow(block_idx + 1, target_block, stop_at=stop_at)

        return None

    def _convert_instruction(self, instr: MediumLevelILInstruction) -> Optional[HLILStatement]:
        # Convert debug information to comments
        if isinstance(instr, MLILDebug):
            return HLILComment(f'{instr.debug_type}({instr.value})')

        # Skip nop instructions
        elif isinstance(instr, MLILNop):
            return None

        # Return statement
        if isinstance(instr, MLILRet):
            if instr.value:
                return HLILReturn(self._convert_expr(instr.value))
            return HLILReturn()

        # Assignment
        elif isinstance(instr, MLILSetVar):
            return HLILAssign(
                HLILVar(HLILVariable(instr.var.name, None)),
                self._convert_expr(instr.value)
            )

        # Store to register
        elif isinstance(instr, MLILStoreReg):
            var = HLILVariable(kind=VariableKind.REG, index=instr.index)
            return HLILAssign(HLILVar(var), self._convert_expr(instr.value))

        # Store to global
        elif isinstance(instr, MLILStoreGlobal):
            var = HLILVariable(kind=VariableKind.GLOBAL, index=instr.index)
            return HLILAssign(HLILVar(var), self._convert_expr(instr.value))

        # Function calls (convert to expression statements)
        elif isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
            call_expr = self._convert_expr(instr)
            # Save call expression for REG[0] substitution
            self.last_call = call_expr
            return HLILExprStmt(call_expr)

        # Control flow handled by _reconstruct_control_flow
        elif isinstance(instr, (MLILIf, MLILGoto)):
            return None

        # Expression statement (catch-all for other expressions)
        elif isinstance(instr, MediumLevelILExpr):
            return HLILExprStmt(self._convert_expr(instr))

        return None

    def _convert_expr(self, expr: MediumLevelILExpr) -> HLILExpression:
        # Variable reference
        if isinstance(expr, MLILVar):
            return HLILVar(HLILVariable(expr.var.name, None))

        # Constant
        elif isinstance(expr, MLILConst):
            return HLILConst(expr.value)

        # Binary operation
        elif isinstance(expr, MLILBinaryOp):
            op = _BINARY_OP_MAP.get(expr.operation, '?')
            return HLILBinaryOp(
                op,
                self._convert_expr(expr.lhs),
                self._convert_expr(expr.rhs)
            )

        # Address-of operator (output parameters) - must come before MLILUnaryOp
        elif isinstance(expr, MLILAddressOf):
            # Convert &var to addr_of(var) intrinsic call
            operand = self._convert_expr(expr.operand)
            return HLILCall('addr_of', [operand])

        # Unary operation
        elif isinstance(expr, MLILUnaryOp):
            # Convert logical NOT to explicit comparison with 0
            if expr.operation in (MediumLevelILOperation.MLIL_LOGICAL_NOT, MediumLevelILOperation.MLIL_TEST_ZERO):
                operand = self._convert_expr(expr.operand)
                return HLILBinaryOp('==', operand, HLILConst(0))

            op = _UNARY_OP_MAP.get(expr.operation, '?')
            return HLILUnaryOp(
                op,
                self._convert_expr(expr.operand)
            )

        # Register load
        elif isinstance(expr, MLILLoadReg):
            # Special case: REG[0] is used for return values
            if expr.index == 0 and self.last_call is not None:
                call_expr = self.last_call
                self.last_call = None
                return call_expr

            var = HLILVariable(kind=VariableKind.REG, index=expr.index)
            return HLILVar(var)

        # Global variable load
        elif isinstance(expr, MLILLoadGlobal):
            var = HLILVariable(kind=VariableKind.GLOBAL, index=expr.index)
            return HLILVar(var)

        # Function call
        elif isinstance(expr, MLILCall):
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILCall(str(expr.target), args)

        # System call
        elif isinstance(expr, MLILSyscall):
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILSyscall(expr.subsystem, expr.cmd, args)

        # Script call (external module call)
        elif isinstance(expr, MLILCallScript):
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILExternCall(f'{expr.module}:{expr.func}', args)

        # Fallback: create a constant with string representation
        else:
            return HLILConst(f'<{type(expr).__name__}>')


def convert_mlil_to_hlil(mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
    converter = MLILToHLILConverter(mlil_func)
    return converter.convert()

'''
MLIL to HLIL Converter

Converts unstructured MLIL (with goto/label) to structured HLIL (with if/while/switch).
'''

from typing import Dict, List, Set, Optional
from ir.mlil.mlil import (
    MediumLevelILFunction,
    MediumLevelILBasicBlock,
    MediumLevelILInstruction,
    MediumLevelILExpr,
    MediumLevelILStatement,
    MLILVariable,
    MLILGoto,
    MLILIf,
    MLILRet,
    MLILSetVar,
    MLILVar,
    MLILConst,
    MLILBinaryOp,
    MLILUnaryOp,
    MLILCall,
    MLILSyscall,
    MLILCallScript,
    MLILLoadReg,
    MLILStoreReg,
    MLILLoadGlobal,
    MLILStoreGlobal,
    MLILDebug,
    MLILNop,
)
from .hlil import (
    HighLevelILFunction, HLILBlock, HLILStatement, HLILExpression,
    HLILVariable, HLILVar, HLILConst, HLILBinaryOp, HLILUnaryOp, HLILCall,
    HLILAssign, HLILExprStmt, HLILReturn, HLILIf, HLILWhile, HLILBreak, HLILContinue,
    HLILComment
)


class MLILToHLILConverter:
    '''Convert MLIL function to HLIL function'''

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = HighLevelILFunction(mlil_func.name, mlil_func.start_addr)

        # Control flow graph
        self.block_successors: Dict[int, List[int]] = {}  # block_index -> [successor_indices]
        self.visited_blocks: Set[int] = set()

        # Track last function call for return value substitution
        self.last_call: Optional[HLILExpression] = None  # Track last call expression for REG[0] substitution

    def _uses_reg0(self, instr: MediumLevelILInstruction) -> bool:
        '''Check if instruction uses REG[0]'''
        if isinstance(instr, MLILLoadReg) and instr.index == 0:
            return True

        # Recursively check expressions
        if isinstance(instr, MLILBinaryOp):
            return self._uses_reg0(instr.lhs) or self._uses_reg0(instr.rhs)

        elif isinstance(instr, MLILUnaryOp):
            return self._uses_reg0(instr.operand)

        elif isinstance(instr, (MLILSetVar, MLILStoreGlobal, MLILStoreReg)):
            return self._uses_reg0(instr.value)

        elif isinstance(instr, (MLILIf, MLILRet)):
            if hasattr(instr, 'condition'):
                return self._uses_reg0(instr.condition)
            elif hasattr(instr, 'value') and instr.value:
                return self._uses_reg0(instr.value)

        elif isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
            # Check arguments
            if hasattr(instr, 'args'):
                return any(self._uses_reg0(arg) for arg in instr.args)

        return False

    def convert(self) -> HighLevelILFunction:
        '''Convert MLIL function to HLIL function'''
        # Phase 1: Convert variables
        self._convert_variables()

        # Phase 2: Build control flow graph
        self._build_cfg()

        # Phase 3: Reconstruct structured control flow
        if self.mlil_func.basic_blocks:
            self._reconstruct_control_flow(0, self.hlil_func.body)

        return self.hlil_func

    def _convert_variables(self):
        '''Convert MLIL variables to HLIL variables (only used ones)'''
        # Collect used variables
        used_vars = set()

        def collect_used_vars(expr):
            '''Recursively collect variable names from expression'''
            if isinstance(expr, MLILVar):
                used_vars.add(expr.var.name)
            elif isinstance(expr, MLILBinaryOp):
                collect_used_vars(expr.lhs)
                collect_used_vars(expr.rhs)
            elif isinstance(expr, MLILUnaryOp):
                collect_used_vars(expr.operand)
            elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in expr.args:
                    collect_used_vars(arg)

        # Scan all instructions
        for block in self.mlil_func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar):
                    used_vars.add(inst.var.name)  # Var is defined (may be used later)
                    collect_used_vars(inst.value)
                elif isinstance(inst, MLILIf):
                    collect_used_vars(inst.condition)
                elif isinstance(inst, MLILRet):
                    if inst.value:
                        collect_used_vars(inst.value)
                elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
                    for arg in inst.args:
                        collect_used_vars(arg)
                elif isinstance(inst, MLILStoreGlobal):
                    collect_used_vars(inst.value)
                elif isinstance(inst, MLILStoreReg):
                    collect_used_vars(inst.value)

        # Convert only used variables
        for mlil_var in self.mlil_func.variables.values():
            # Skip unused variables
            if mlil_var.name not in used_vars:
                continue

            # Convert type hint
            type_hint = None
            if hasattr(mlil_var, 'type') and mlil_var.type:
                type_hint = str(mlil_var.type)

            hlil_var = HLILVariable(mlil_var.name, type_hint)

            # Determine if parameter based on naming convention (arg0, arg1, etc.)
            is_parameter = mlil_var.name.startswith('arg')

            # Add as parameter or local variable
            if is_parameter:
                self.hlil_func.parameters.append(hlil_var)
            else:
                self.hlil_func.variables.append(hlil_var)

    def _build_cfg(self):
        '''Build control flow graph from MLIL basic blocks'''
        for i, block in enumerate(self.mlil_func.basic_blocks):
            successors = []

            # Check last instruction for control flow
            if block.instructions:
                last_instr = block.instructions[-1]

                if isinstance(last_instr, MLILIf):
                    # Conditional branch: has true and false successors
                    if last_instr.true_target is not None:
                        successors.append(last_instr.true_target.index)
                    if last_instr.false_target is not None:
                        successors.append(last_instr.false_target.index)

                elif isinstance(last_instr, MLILGoto):
                    # Unconditional jump
                    if last_instr.target is not None:
                        successors.append(last_instr.target.index)

                elif not isinstance(last_instr, MLILRet):
                    # Fall-through to next block
                    if i + 1 < len(self.mlil_func.basic_blocks):
                        successors.append(i + 1)

            self.block_successors[i] = successors

    def _find_merge_block(self, block1_idx: int, block2_idx: int) -> Optional[int]:
        '''Find the merge block where two branches converge

        Args:
            block1_idx: First branch block index
            block2_idx: Second branch block index

        Returns:
            Merge block index, or None if no merge block found
        '''
        # Check direct successors first
        succ1 = set(self.block_successors.get(block1_idx, []))
        succ2 = set(self.block_successors.get(block2_idx, []))

        # If there's a common direct successor, that's the merge block
        common = succ1 & succ2
        if common:
            return min(common)

        # Check if one branch target is reachable from the other
        # This handles: if (cond) goto merge; else goto path_to_merge;
        def is_reachable(from_idx: int, to_idx: int, max_depth: int = 10) -> bool:
            '''Check if to_idx is reachable from from_idx'''
            if from_idx == to_idx:
                return True
            visited = set()
            queue = [(from_idx, 0)]
            while queue:
                idx, depth = queue.pop(0)
                if idx == to_idx:
                    return True
                if idx in visited or depth >= max_depth:
                    continue
                visited.add(idx)
                for succ in self.block_successors.get(idx, []):
                    if succ not in visited:
                        queue.append((succ, depth + 1))
            return False

        # If block1 is reachable from block2, block1 is the merge
        if is_reachable(block2_idx, block1_idx):
            return block1_idx
        # If block2 is reachable from block1, block2 is the merge
        if is_reachable(block1_idx, block2_idx):
            return block2_idx

        # Otherwise, do limited BFS (max 3 levels)
        def get_reachable(start_idx: int, max_depth: int = 3) -> Dict[int, int]:
            '''Returns {block_idx: depth}'''
            reachable = {}
            queue = [(start_idx, 0)]
            visited = set()

            while queue:
                idx, depth = queue.pop(0)
                if idx in visited or depth > max_depth:
                    continue
                visited.add(idx)
                reachable[idx] = depth

                for succ in self.block_successors.get(idx, []):
                    if succ not in visited:
                        queue.append((succ, depth + 1))

            return reachable

        reachable1 = get_reachable(block1_idx)
        reachable2 = get_reachable(block2_idx)

        # Find common blocks
        common_blocks = set(reachable1.keys()) & set(reachable2.keys())
        if not common_blocks:
            return None

        # Return the closest common block (minimum combined depth)
        return min(common_blocks, key=lambda idx: reachable1[idx] + reachable2[idx])

    def _is_passthrough_block(self, block_idx: int) -> bool:
        '''Check if a block is a passthrough (only goto, no real statements)

        Args:
            block_idx: Block index to check

        Returns:
            True if the block only contains goto/debug, False otherwise
        '''
        if block_idx >= len(self.mlil_func.basic_blocks):
            return False

        block = self.mlil_func.basic_blocks[block_idx]
        if not block.instructions:
            return False

        # Check all instructions except the last one
        for instr in block.instructions[:-1]:
            # Allow only debug/nop instructions
            if not isinstance(instr, (MLILDebug, MLILNop)):
                return False

        # Last instruction must be a goto
        last_instr = block.instructions[-1]
        return isinstance(last_instr, MLILGoto)

    def _reconstruct_control_flow(self, block_idx: int, target_block: HLILBlock, stop_at: Optional[int] = None) -> Optional[int]:
        '''Recursively reconstruct structured control flow

        Args:
            block_idx: Current MLIL basic block index
            target_block: HLIL block to add statements to
            stop_at: Stop processing at this block index (for handling merge blocks)

        Returns:
            Next block index to process, or None if control flow ends
        '''
        # Skip passthrough blocks
        original_idx = block_idx
        while self._is_passthrough_block(block_idx):
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

        # Stop if we reached the merge block
        if stop_at is not None and block_idx == stop_at:
            return block_idx

        self.visited_blocks.add(block_idx)
        mlil_block = self.mlil_func.basic_blocks[block_idx]

        # Convert all instructions except the last (terminator)
        instructions = mlil_block.instructions[:-1] if mlil_block.instructions else []
        for i, instr in enumerate(instructions):
            # Check if this is a call and next instruction uses REG[0]
            if isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
                # Check next instruction (or terminator)
                next_instr = None
                if i + 1 < len(instructions):
                    next_instr = instructions[i + 1]
                elif mlil_block.instructions:
                    next_instr = mlil_block.instructions[-1]  # terminator

                should_skip = False
                if next_instr and self._uses_reg0(next_instr):
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

                # Process branches (stop at merge block)
                true_block = HLILBlock()
                false_block = HLILBlock()

                saved_visited = self.visited_blocks.copy()

                if true_target_idx is not None:
                    self.visited_blocks = saved_visited.copy()
                    self._reconstruct_control_flow(true_target_idx, true_block, stop_at=merge_block_idx)

                if false_target_idx is not None:
                    self.visited_blocks = saved_visited.copy()
                    self._reconstruct_control_flow(false_target_idx, false_block, stop_at=merge_block_idx)

                self.visited_blocks = saved_visited

                # Add if statement
                if_stmt = HLILIf(condition, true_block, false_block if false_block.statements else None)
                target_block.add_statement(if_stmt)

                # Continue with merge block
                if merge_block_idx is not None:
                    # Don't process merge block if it's our stop point (belongs to parent)
                    if stop_at is not None and merge_block_idx == stop_at:
                        return merge_block_idx
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
        '''Convert a single MLIL instruction to HLIL statement'''
        # Convert debug information to comments
        if isinstance(instr, MLILDebug):
            return HLILComment(f'{instr.debug_type}({instr.value})')

        # Skip nop instructions
        elif isinstance(instr, MLILNop):
            return None

        # Return statement
        if isinstance(instr, MLILRet):
            if hasattr(instr, 'value') and instr.value:
                return HLILReturn(self._convert_expr(instr.value))
            return HLILReturn()

        # Assignment
        elif isinstance(instr, MLILSetVar):
            var_name = instr.var.name
            type_hint = str(instr.var.type) if hasattr(instr.var, 'type') and instr.var.type else None
            return HLILAssign(
                HLILVar(HLILVariable(var_name, type_hint)),
                self._convert_expr(instr.value)
            )

        # Store to register (treat as assignment to special variable)
        elif isinstance(instr, MLILStoreReg):
            reg_name = f'REG[{instr.index}]'
            return HLILAssign(
                HLILVar(HLILVariable(reg_name)),
                self._convert_expr(instr.value)
            )

        # Function calls (convert to expression statements)
        elif isinstance(instr, (MLILCall, MLILSyscall, MLILCallScript)):
            call_expr = self._convert_expr(instr)
            # Save call expression for REG[0] substitution
            self.last_call = call_expr
            return HLILExprStmt(call_expr)

        # Conditional jump -> skip for now (needs control flow restructuring)
        elif isinstance(instr, MLILIf):
            return None  # TODO: Implement control flow restructuring

        # Unconditional jump -> skip for now
        elif isinstance(instr, MLILGoto):
            return None  # TODO: Implement control flow restructuring

        # Expression statement (catch-all for other expressions)
        elif isinstance(instr, MediumLevelILExpr):
            return HLILExprStmt(self._convert_expr(instr))

        return None

    def _convert_expr(self, expr: MediumLevelILExpr) -> HLILExpression:
        '''Convert MLIL expression to HLIL expression'''
        # Variable reference
        if isinstance(expr, MLILVar):
            var_name = expr.var.name
            type_hint = str(expr.var.type) if hasattr(expr.var, 'type') and expr.var.type else None
            return HLILVar(HLILVariable(var_name, type_hint))

        # Constant
        elif isinstance(expr, MLILConst):
            return HLILConst(expr.value)

        # Binary operation
        elif isinstance(expr, MLILBinaryOp):
            # Map MLIL operation to operator string
            from ir.mlil.mlil import MediumLevelILOperation
            op_map = {
                MediumLevelILOperation.MLIL_ADD: '+',
                MediumLevelILOperation.MLIL_SUB: '-',
                MediumLevelILOperation.MLIL_MUL: '*',
                MediumLevelILOperation.MLIL_DIV: '/',
                MediumLevelILOperation.MLIL_MOD: '%',
                MediumLevelILOperation.MLIL_AND: '&',
                MediumLevelILOperation.MLIL_OR: '|',
                MediumLevelILOperation.MLIL_XOR: '^',
                MediumLevelILOperation.MLIL_SHL: '<<',
                MediumLevelILOperation.MLIL_SHR: '>>',
                MediumLevelILOperation.MLIL_LOGICAL_AND: '&&',
                MediumLevelILOperation.MLIL_LOGICAL_OR: '||',
                MediumLevelILOperation.MLIL_EQ: '==',
                MediumLevelILOperation.MLIL_NE: '!=',
                MediumLevelILOperation.MLIL_LT: '<',
                MediumLevelILOperation.MLIL_LE: '<=',
                MediumLevelILOperation.MLIL_GT: '>',
                MediumLevelILOperation.MLIL_GE: '>=',
            }
            op = op_map.get(expr.operation, '?')
            return HLILBinaryOp(
                op,
                self._convert_expr(expr.lhs),
                self._convert_expr(expr.rhs)
            )

        # Unary operation
        elif isinstance(expr, MLILUnaryOp):
            # Map MLIL operation to operator string
            from ir.mlil.mlil import MediumLevelILOperation

            # Convert logical NOT to explicit comparison with 0
            if expr.operation in (MediumLevelILOperation.MLIL_LOGICAL_NOT, MediumLevelILOperation.MLIL_TEST_ZERO):
                operand = self._convert_expr(expr.operand)
                return HLILBinaryOp('==', operand, HLILConst(0))

            # Other unary operations
            op_map = {
                MediumLevelILOperation.MLIL_NEG: '-',
            }
            op = op_map.get(expr.operation, '?')
            return HLILUnaryOp(
                op,
                self._convert_expr(expr.operand)
            )

        # Register load
        elif isinstance(expr, MLILLoadReg):
            # Special case: REG[0] is used for return values
            # If last_call exists, substitute it instead of REG[0]
            if expr.index == 0 and self.last_call is not None:
                call_expr = self.last_call
                self.last_call = None  # Clear after use
                return call_expr

            reg_name = f'REG[{expr.index}]'
            return HLILVar(HLILVariable(reg_name))

        # Global variable load
        elif isinstance(expr, MLILLoadGlobal):
            if hasattr(expr, 'offset'):
                global_name = f'global_{expr.offset}'
            else:
                global_name = 'global'
            return HLILVar(HLILVariable(global_name))

        # Function call
        elif isinstance(expr, MLILCall):
            args = [self._convert_expr(arg) for arg in expr.args]
            func_name = expr.target if hasattr(expr, 'target') else str(expr)
            return HLILCall(str(func_name), args)

        # System call
        elif isinstance(expr, MLILSyscall):
            from .hlil import HLILSyscall
            args = [self._convert_expr(arg) for arg in expr.args]
            subsystem = expr.subsystem if hasattr(expr, 'subsystem') else 'sys'
            cmd = expr.cmd if hasattr(expr, 'cmd') else str(expr.syscall_id)
            return HLILSyscall(subsystem, cmd, args)

        # Script call
        elif isinstance(expr, MLILCallScript):
            args = [self._convert_expr(arg) for arg in expr.args]
            func_name = f'{expr.module}.{expr.func}'
            return HLILCall(func_name, args)

        # Fallback: create a constant with string representation
        else:
            return HLILConst(f'<{type(expr).__name__}>')


def convert_mlil_to_hlil(mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
    '''Convert MLIL function to HLIL function

    Args:
        mlil_func: MLIL function to convert

    Returns:
        Converted HLIL function
    '''
    converter = MLILToHLILConverter(mlil_func)
    return converter.convert()

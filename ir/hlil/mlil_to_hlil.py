'''MLIL to HLIL Converter - goto/label to if/while/switch'''

from typing import Dict, List, Set, Optional

from ir.mlil.mlil import *
from .hlil import *


class MLILToHLILConverter:

    def __init__(self, mlil_func: MediumLevelILFunction):
        self.mlil_func = mlil_func
        self.hlil_func = HighLevelILFunction(mlil_func.name, mlil_func.start_addr)

        self.block_successors: Dict[int, List[int]] = {}
        self.visited_blocks: Set[int] = set()
        self.last_call: Optional[HLILExpression] = None

    def _uses_reg0(self, instr: MediumLevelILInstruction) -> bool:
        if isinstance(instr, MLILLoadReg) and instr.index == 0:
            return True

        if isinstance(instr, MLILBinaryOp):
            return self._uses_reg0(instr.lhs) or self._uses_reg0(instr.rhs)

        elif isinstance(instr, MLILUnaryOp):
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
        used_vars = set()

        def collect_used_vars(expr):
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

        # Collect parameters and variables separately
        parameters = []
        variables = []

        for mlil_var in self.mlil_func.variables.values():
            # Skip unused variables
            if mlil_var.name not in used_vars:
                continue

            # Determine if parameter based on naming convention (arg0, arg1, etc.)
            is_parameter = mlil_var.name.startswith('arg')

            # Type information will be added by separate passes
            hlil_var = HLILVariable(mlil_var.name, None)

            # Add as parameter or local variable
            if is_parameter:
                # Extract parameter index for sorting
                try:
                    param_index = int(mlil_var.name[3:])  # arg1 -> 1
                    parameters.append((param_index, hlil_var))
                except ValueError:
                    variables.append(hlil_var)
            else:
                variables.append(hlil_var)

        # Sort parameters by index and add to function
        parameters.sort(key=lambda x: x[0])
        self.hlil_func.parameters.extend([var for _, var in parameters])
        self.hlil_func.variables.extend(variables)

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

    def _find_merge_block(self, block1_idx: int, block2_idx: int) -> Optional[int]:
        '''Find merge block where branches converge'''
        succ1 = set(self.block_successors.get(block1_idx, []))
        succ2 = set(self.block_successors.get(block2_idx, []))

        common = succ1 & succ2
        if common:
            return min(common)

        def is_reachable(from_idx: int, to_idx: int, max_depth: int = 10) -> bool:
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

        if is_reachable(block2_idx, block1_idx):
            return block1_idx
        if is_reachable(block1_idx, block2_idx):
            return block2_idx

        def get_reachable(start_idx: int, max_depth: int = 3) -> Dict[int, int]:
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
        '''Check if block is passthrough (only goto)'''
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
        '''Reconstruct structured control flow from goto/label'''
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
            var_name = instr.var.name
            # Type information not available on non-SSA MLIL variables
            return HLILAssign(
                HLILVar(HLILVariable(var_name, None)),
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
        # Variable reference
        if isinstance(expr, MLILVar):
            var_name = expr.var.name
            # Type information not available on non-SSA MLIL variables
            return HLILVar(HLILVariable(var_name, None))

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

        # Address-of operator (output parameters) - must come before MLILUnaryOp
        elif isinstance(expr, MLILAddressOf):
            # Convert &var to addr_of(var) intrinsic call
            operand = self._convert_expr(expr.operand)
            return HLILCall('addr_of', [operand])

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
            func_name = str(expr.target)
            return HLILCall(func_name, args)

        # System call
        elif isinstance(expr, MLILSyscall):
            from .hlil import HLILSyscall
            args = [self._convert_expr(arg) for arg in expr.args]
            return HLILSyscall(expr.subsystem, expr.cmd, args)

        # Script call (external module call)
        elif isinstance(expr, MLILCallScript):
            from .hlil import HLILExternCall
            args = [self._convert_expr(arg) for arg in expr.args]
            target = f'{expr.module}:{expr.func}'
            return HLILExternCall(target, args)

        # Fallback: create a constant with string representation
        else:
            return HLILConst(f'<{type(expr).__name__}>')


def convert_mlil_to_hlil(mlil_func: MediumLevelILFunction) -> HighLevelILFunction:
    converter = MLILToHLILConverter(mlil_func)
    return converter.convert()

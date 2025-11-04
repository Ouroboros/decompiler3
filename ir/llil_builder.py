'''
LLIL v2 Builder - Layered architecture for convenience
'''

from typing import Union, Optional, List
from .llil import *
from .llil import (
    LowLevelILBinaryOp, LowLevelILOperation,
    LowLevelILSub, LowLevelILDiv, LowLevelILEq, LowLevelILNe,
    LowLevelILLt, LowLevelILLe, LowLevelILGt, LowLevelILGe,
    LowLevelILStackPush, LowLevelILStackStore, LowLevelILStackPop,
    LowLevelILStackAddr, LowLevelILSpAdd,
    WORD_SIZE
)


class LowLevelILBuilder:
    '''Mid-level builder with convenience methods'''

    def __init__(self, function: LowLevelILFunction):
        self.function = function
        self.current_block: Optional[LowLevelILBasicBlock] = None
        self.current_sp: int = 0  # Track current stack pointer state (for block sp_in/sp_out)
        self.frame_base_sp: Optional[int] = None  # Stack pointer at function entry (for frame-relative access)
        self.vstack: List[LowLevelILInstruction] = []  # Virtual stack for expression tracking

    def set_current_block(self, block: LowLevelILBasicBlock, sp: Optional[int] = None):
        '''Set the current basic block for instruction insertion

        Args:
            block: The block to set as current
            sp: Optional stack pointer value. If None, uses function's num_params on first block,
                or continues from previous block's sp

        Raises:
            RuntimeError: If block has not been added to function
        '''
        # Verify block has been added to function
        if block not in self.function.basic_blocks:
            raise RuntimeError(f'Block {block} has not been added to function. Call function.add_basic_block() first.')

        # Save previous block's sp_out if we have a current block
        if self.current_block is not None:
            self.current_block.sp_out = self.current_sp

        self.current_block = block

        # Set new block's sp_in and current sp
        if sp is not None:
            self.current_sp = sp
        elif self.frame_base_sp is None:
            # First block: use function's parameter count as initial sp
            self.current_sp = self.function.num_params
        # else: continue from previous block's sp

        block.sp_in = self.current_sp

        # Save frame base sp on first block (function entry)
        # New scheme: fp = 0 (points to first parameter)
        if self.frame_base_sp is None:
            self.frame_base_sp = 0
            self.function.frame_base_sp = 0

    def get_block_by_addr(self, addr: int) -> Optional[LowLevelILBasicBlock]:
        '''Get block by start address'''
        return self.function.get_block_by_addr(addr)

    def get_block_by_label(self, label: str) -> Optional[LowLevelILBasicBlock]:
        '''Get block by label name'''
        return self.function.get_block_by_label(label)

    def add_instruction(self, instr: LowLevelILInstruction):
        '''Add instruction to current block and update stack pointer tracking'''
        if self.current_block is None:
            raise RuntimeError('No current basic block set')
        self.current_block.add_instruction(instr)

        # Update stack pointer based on instruction type
        if isinstance(instr, LowLevelILSpAdd):
            self.current_sp += instr.delta

    # === Virtual Stack Management ===

    def _to_expr(self, value: Union[LowLevelILInstruction, int, float, str]) -> LowLevelILInstruction:
        '''Convert value to expression'''
        if isinstance(value, LowLevelILInstruction):
            return value
        elif isinstance(value, int):
            return self.const_int(value)
        elif isinstance(value, float):
            return self.const_float(value)
        elif isinstance(value, str):
            return self.const_str(value)
        else:
            raise TypeError(f'Cannot convert {type(value)} to expression')

    def push(self, value: Union[LowLevelILInstruction, int, float, str], size: int = 4) -> LowLevelILInstruction:
        '''Push value onto stack using StackPush instruction

        Generates LowLevelILStackPush and maintains sp.
        Returns the expression that was pushed.
        '''
        expr = self._to_expr(value)
        # Generate StackPush instruction (semantic: STACK[sp] = value; sp++)
        push_instr = LowLevelILStackPush(expr, size)
        push_instr.slot_index = self.current_sp  # Record slot being written to
        self.add_instruction(push_instr)
        # Maintain sp in builder (sp maintained here, not in add_instruction)
        self.current_sp += 1
        # Track on vstack for expression tracking
        self.vstack.append(expr)
        return expr

    def pop(self, size: int = 4) -> LowLevelILInstruction:
        '''Pop value from stack using StackPop expression

        Returns LowLevelILStackPop expression (value expression with side effect).
        Maintains sp in builder (sp maintained here, not in add_instruction).
        '''
        # Pop from virtual stack for tracking
        if self.vstack:
            self.vstack.pop()

        # Create StackPop expression (semantic: sp--; return STACK[sp])
        pop_expr = LowLevelILStackPop(size)
        # Maintain sp in builder
        self.current_sp -= 1
        pop_expr.slot_index = self.current_sp  # Record slot being read from

        return pop_expr

    # === Legacy Stack Operations (kept for compatibility) ===

    def stack_push(self, value: Union[LowLevelILInstruction, int, str], size: int = 4):
        '''STACK[sp++] = value (legacy, use push() instead)'''
        self.push(value, size)

    def stack_pop(self, size: int = 4) -> LowLevelILStackLoad:
        '''STACK[--sp] (legacy, use pop() instead)'''
        return self.pop(size)

    def stack_load(self, offset: int, size: int = 4) -> LowLevelILStackLoad:
        '''STACK[sp + offset] (no sp change)'''
        return LowLevelILStackLoad(offset, size)

    def stack_store(self, value: Union[LowLevelILInstruction, int, str], offset: int, size: int = 4):
        '''STACK[sp + offset] = value (no sp change)'''
        self.add_instruction(LowLevelILStackStore(value, offset, size))

    def frame_load(self, offset: int, size: int = 4) -> 'LowLevelILFrameLoad':
        '''STACK[frame + offset] - Frame-relative load (for function parameters/locals)

        Args:
            offset: Byte offset relative to frame base (function entry sp)
            size: Size in bytes (default 4)

        Returns:
            FrameLoad expression
        '''
        return LowLevelILFrameLoad(offset, size)

    def frame_store(self, value: Union[LowLevelILInstruction, int, str], offset: int, size: int = 4):
        '''STACK[frame + offset] = value - Frame-relative store (for function parameters/locals)

        Args:
            value: Value to store
            offset: Byte offset relative to frame base (function entry sp)
            size: Size in bytes (default 4)
        '''
        self.add_instruction(LowLevelILFrameStore(value, offset, size))

    def load_frame(self, offset: int):
        '''Load from frame + offset and push to stack

        Use this for accessing function parameters (frame-relative addressing).

        Args:
            offset: Byte offset relative to frame base
        '''
        frame_val = self.frame_load(offset)
        self.stack_push(frame_val)

    def load_stack(self, offset: int):
        '''Load from sp + offset and push to stack

        Automatically determines whether to use sp-relative or fp-relative addressing:
        - If accessing parameter area (STACK[0..num_params-1]), uses fp
        - Otherwise uses sp

        Args:
            offset: Byte offset relative to current sp
        '''
        # Calculate word offset
        word_offset = offset // WORD_SIZE
        # Calculate absolute stack position
        absolute_pos = self.current_sp + word_offset

        # Check if accessing parameter area
        # New scheme: fp = 0, parameters at STACK[0..num_params-1]
        num_params = self.function.num_params if hasattr(self.function, 'num_params') else 0
        if absolute_pos >= 0 and absolute_pos < num_params:
            # Accessing parameters - use fp-relative
            # absolute_pos = fp + fp_offset, and fp = 0, so fp_offset = absolute_pos
            fp_offset = absolute_pos * WORD_SIZE  # Convert to bytes
            self.load_frame(fp_offset)
        else:
            # Accessing within current stack frame (temporaries/locals) - use sp-relative
            stack_val = self.stack_load(offset)
            self.stack_push(stack_val)

    def push_stack_addr(self, offset: int):
        '''Push the address of stack location (sp + offset)

        This is used for PUSH_STACK_OFFSET instruction which pushes the address
        of a stack slot. The slot index is computed at build time using current_sp
        and remains constant regardless of subsequent sp changes.

        Args:
            offset: Byte offset relative to current sp (at build time)
        '''
        # Convert byte offset to word offset
        word_offset = offset // WORD_SIZE
        # Calculate absolute slot index using current sp
        slot_index = self.current_sp + word_offset
        # Create stack address with absolute slot index
        stack_addr = LowLevelILStackAddr(slot_index)
        self.stack_push(stack_addr)

    def sp_add(self, delta: int):
        '''Adjust stack pointer: sp += delta

        Only for shrinking stack (delta < 0). Use push() to grow stack.

        Args:
            delta: Number of words to add (must be negative)
        '''
        assert delta < 0, f'sp_add only for shrinking stack, got delta={delta}. Use push() to grow stack.'
        self.add_instruction(LowLevelILSpAdd(delta))

        # Synchronize vstack: pop items when shrinking stack
        for _ in range(-delta):
            if self.vstack:
                self.vstack.pop()

    # === Register Operations ===

    def reg_store(self, reg_index: int, value: Union[LowLevelILInstruction, int], size: int = 4):
        '''R[index] = value'''
        self.add_instruction(LowLevelILRegStore(reg_index, value, size))

    def reg_load(self, reg_index: int, size: int = 4) -> LowLevelILRegLoad:
        '''R[index]'''
        return LowLevelILRegLoad(reg_index, size)

    # === Constants ===

    def const_int(self, value: int, size: int = 4, is_hex: bool = False) -> LowLevelILConst:
        '''Integer constant

        Args:
            value: Integer value
            size: Size in bytes (default 4)
            is_hex: If True, display as hex; if False, use auto detection (default)
        '''
        return LowLevelILConst(value, size, is_hex)

    def const_float(self, value: float, size: int = 4) -> LowLevelILConst:
        '''Float constant (size: 4 for float, 8 for double)'''
        return LowLevelILConst(value, size, False)

    def const_str(self, value: str) -> LowLevelILConst:
        '''String constant'''
        return LowLevelILConst(value, 0, False)

    # === Binary Operations ===

    def _binary_op(self, op_class, lhs = None, rhs = None, *, push: bool = True, size: int = 4) -> LowLevelILInstruction:
        '''Generic binary operation handler

        Stack operation order (for implicit mode):
          rhs = stack_pop();   // First pop gets right operand (top of stack)
          lhs = stack_pop();   // Second pop gets left operand (below it)
          result = (lhs OP rhs);

        Args:
            op_class: The operation class (e.g., LowLevelILAdd)
            lhs: Left operand (None = pop from vstack)
            rhs: Right operand (None = pop from vstack)
            push: Whether to push result back to vstack
            size: Operation size

        Returns:
            The operation expression
        '''
        # Get operands - both must be None or both must be provided
        if lhs is None and rhs is None:
            # Implicit mode: pop both from vstack
            rhs = self.pop(size)  # First pop gets right operand (top of stack)
            lhs = self.pop(size)  # Second pop gets left operand (below it)
        elif lhs is not None and rhs is not None:
            # Explicit mode: both provided
            lhs = self._to_expr(lhs)
            rhs = self._to_expr(rhs)
        else:
            raise ValueError('Binary operation requires both operands or neither (lhs and rhs must both be None or both be provided)')

        # Create operation with operands
        op = op_class(lhs, rhs, size)

        # Binary operations are expressions, not statements
        # Only add as instruction if we're pushing (making it a statement via StackPush)
        if push:
            # Use push() to properly set slot_index and maintain sp
            self.push(op, size)
        else:
            # If not pushing, add the operation itself (e.g., for comparisons in branches)
            self.add_instruction(op)

        return op

    def add(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''ADD operation - computes lhs + rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILAdd, lhs, rhs, push = push, size = size)

    def sub(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''SUB operation - computes lhs - rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILSub, lhs, rhs, push = push, size = size)

    def mul(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''MUL operation - computes lhs * rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILMul, lhs, rhs, push = push, size = size)

    def div(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''DIV operation - computes lhs / rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILDiv, lhs, rhs, push = push, size = size)

    # === Comparison Operations ===

    def eq(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''EQ operation - computes lhs == rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILEq, lhs, rhs, push = push, size = size)

    def ne(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''NE operation - computes lhs != rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILNe, lhs, rhs, push = push, size = size)

    def lt(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''LT operation - computes lhs < rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILLt, lhs, rhs, push = push, size = size)

    def le(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''LE operation - computes lhs <= rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILLe, lhs, rhs, push = push, size = size)

    def gt(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''GT operation - computes lhs > rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILGt, lhs, rhs, push = push, size = size)

    def ge(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''GE operation - computes lhs >= rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILGe, lhs, rhs, push = push, size = size)

    # === Control Flow ===

    def jmp(self, target: Union[str, LowLevelILBasicBlock]):
        '''Unconditional jump - target can be label or block'''
        if isinstance(target, str):
            target_block = self.get_block_by_label(target)
            if target_block is None:
                raise ValueError(f'Undefined label: {target}')
            target = target_block
        self.add_instruction(LowLevelILJmp(target))

    def branch_if(self, condition: LowLevelILInstruction,
                  true_target: Union[str, LowLevelILBasicBlock],
                  false_target: Union[str, LowLevelILBasicBlock]):
        '''Conditional branch - targets can be labels or blocks

        Args:
            condition: Condition expression
            true_target: Block to jump to if condition is true
            false_target: Block to jump to if condition is false
        '''
        # Resolve true target
        if isinstance(true_target, str):
            true_block = self.get_block_by_label(true_target)
            if true_block is None:
                raise ValueError(f'Undefined label: {true_target}')
            true_target = true_block

        # Resolve false target
        if isinstance(false_target, str):
            false_block = self.get_block_by_label(false_target)
            if false_block is None:
                raise ValueError(f'Undefined label: {false_target}')
            false_target = false_block

        self.add_instruction(LowLevelILIf(condition, true_target, false_target))

    def call(self, target: Union[str, LowLevelILInstruction],
             return_target: Optional[Union[str, LowLevelILBasicBlock]] = None,
             stack_cleanup: Optional[int] = None):
        '''Function call

        Args:
            target: Function name or address
            return_target: Block or label to return to after call (resolved in build_cfg)
            stack_cleanup: Number of stack slots to pop after call (for callee cleanup)
                          If None, no automatic cleanup is performed
        '''
        self.add_instruction(LowLevelILCall(target, return_target))
        # In Falcom VM, callee cleans up the stack (including func_id, ret_addr, and arguments)
        if stack_cleanup is not None:
            self.current_sp -= stack_cleanup

    def ret(self):
        '''Return'''
        self.add_instruction(LowLevelILRet())

    # === Special ===

    def label(self, name: str):
        '''Label - inserts a label instruction at current position

        Note: Block labels should be set via LowLevelILBasicBlock constructor.
        This method only adds a visual label instruction for display purposes.
        '''
        if self.current_block is None:
            raise RuntimeError('No current block to label')
        self.add_instruction(LowLevelILLabelInstr(name))

    def debug_line(self, line_no: int):
        '''Debug line number'''
        self.add_instruction(LowLevelILDebug('line', line_no))

    def syscall(self, subsystem: int, cmd: int, argc: int):
        '''System call

        Args:
            subsystem: System call category (e.g., 6 for audio, graphics, etc.)
            cmd: Command ID within the subsystem
            argc: Number of arguments for this syscall
        '''
        self.add_instruction(LowLevelILSyscall(subsystem, cmd, argc))


class LLILFormatter:
    '''Formatting layer for beautiful output'''

    @staticmethod
    def indent_lines(lines: List[str], indent: str) -> List[str]:
        '''Add indentation to multiple lines

        Args:
            lines: List of lines to indent
            indent: Indentation string to prepend

        Returns:
            List of indented lines
        '''
        return [indent + line for line in lines]

    @staticmethod
    def format_instruction(instr: LowLevelILInstruction) -> str:
        '''Format a single instruction - can be customized per instruction type

        Returns a single line for simple instructions.
        For multi-line instructions, use format_instruction_expanded().
        '''
        # For now, use the instruction's __str__ method
        # This can be extended with custom formatting logic for specific instruction types
        return str(instr)

    @staticmethod
    def _format_binary_op_expanded(binary_op: LowLevelILBinaryOp) -> List[str]:
        '''Format binary operation with expanded pseudo-code

        Stack operation order:
          rhs = stack_pop();   // First pop gets right operand (top of stack)
          lhs = stack_pop();   // Second pop gets left operand (below it)
          result = (lhs OP rhs);
          stack_push(result);
        '''
        # Map operation types to their expression strings
        expr_map = {
            LowLevelILOperation.LLIL_ADD: 'lhs + rhs',
            LowLevelILOperation.LLIL_SUB: 'lhs - rhs',
            LowLevelILOperation.LLIL_MUL: 'lhs * rhs',
            LowLevelILOperation.LLIL_DIV: 'lhs / rhs',
            LowLevelILOperation.LLIL_EQ: '(lhs == rhs) ? 1 : 0',
            LowLevelILOperation.LLIL_NE: '(lhs != rhs) ? 1 : 0',
            LowLevelILOperation.LLIL_LT: '(lhs < rhs) ? 1 : 0',
            LowLevelILOperation.LLIL_LE: '(lhs <= rhs) ? 1 : 0',
            LowLevelILOperation.LLIL_GT: '(lhs > rhs) ? 1 : 0',
            LowLevelILOperation.LLIL_GE: '(lhs >= rhs) ? 1 : 0',
        }

        expr = expr_map[binary_op.operation]

        # Get slot indices from operands if they are StackPop
        rhs_line = 'rhs = STACK[--sp]'
        lhs_line = 'lhs = STACK[--sp]'

        if isinstance(binary_op.rhs, LowLevelILStackPop) and binary_op.rhs.slot_index is not None:
            rhs_line += f' ; [{binary_op.rhs.slot_index}]'

        if isinstance(binary_op.lhs, LowLevelILStackPop) and binary_op.lhs.slot_index is not None:
            lhs_line += f' ; [{binary_op.lhs.slot_index}]'

        return [
            f'; expand {binary_op.operation_name}',
            rhs_line,  # First pop gets right operand (top of stack)
            lhs_line,  # Second pop gets left operand (below it)
            f'STACK[sp++] = {expr}'
        ]

    @staticmethod
    def format_instruction_expanded(instr: LowLevelILInstruction) -> List[str]:
        '''Format instruction with expanded stack operations (multi-line)

        Returns a list of lines showing explicit stack behavior.
        For binary operations like EQ, MUL, ADD, this shows:
        - Pop operations to get operands
        - The actual operation
        - Push operation for result
        '''
        # StackPush containing a binary operation: expand the binary op
        if isinstance(instr, LowLevelILStackPush) and isinstance(instr.value, LowLevelILBinaryOp):
            return LLILFormatter._format_binary_op_expanded(instr.value)

        # Binary operations: pop 2, compute, push 1
        if isinstance(instr, LowLevelILBinaryOp):
            return LLILFormatter._format_binary_op_expanded(instr)

        # StackStore with StackPop: expand to avoid ambiguity
        if isinstance(instr, LowLevelILStackStore) and isinstance(instr.value, LowLevelILStackPop):
            word_offset = instr.offset // WORD_SIZE
            if word_offset == 0:
                target = 'STACK[sp]'
            elif word_offset > 0:
                target = f'STACK[sp + {word_offset}]'
            else:
                target = f'STACK[sp - {-word_offset}]'

            return [
                '; expand POP_TO',
                'temp = STACK[--sp]',
                f'{target} = temp'
            ]

        # For non-binary operations, return single line
        return [str(instr)]

    @staticmethod
    def format_instruction_sequence(instructions: List[LowLevelILInstruction], indent: str = '  ') -> list[str]:
        '''Format sequence of instructions - returns list of lines

        Args:
            instructions: List of instructions to format
            indent: Indentation string to prepend to each line (default: '  ')

        Returns:
            List of formatted lines with indentation
        '''
        result = []

        for instr in instructions:
            # Use expanded format for multi-line instructions
            expanded = LLILFormatter.format_instruction_expanded(instr)
            if len(expanded) > 1:
                # Multi-line instruction
                result.extend(LLILFormatter.indent_lines(expanded, indent))
            else:
                # Single line instruction - add slot comment if applicable
                line = expanded[0]
                # Add slot comment for StackPush and StackPop
                if isinstance(instr, (LowLevelILStackPush, LowLevelILStackPop)) and instr.slot_index is not None:
                    line = f'{line} ; [{instr.slot_index}]'

                result.append(f'{indent}{line}')

        return result

    @staticmethod
    def format_llil_function(func: LowLevelILFunction) -> list[str]:
        assert isinstance(func, LowLevelILFunction)

        '''Format entire LLIL function with beautiful output - returns list of lines'''
        result = [
            f'; ---------- {func.name} ----------',
        ]

        for block in func.basic_blocks:
            # Block header: {block_N}(addr), label, [sp = N, fp = M]

            block_info = [
                f'block_{block.index}(0x{block.start:04X})',
                block.label,
            ]

            # Show sp and fp (fp only on first block)
            if block.index == 0 and func.frame_base_sp is not None:
                block_info.append(f'[sp = {block.sp_in}, fp = {func.frame_base_sp}]')
            else:
                block_info.append(f'[sp = {block.sp_in}]')

            result.append(', '.join(block_info))

            # Skip LowLevelILLabelInstr if present (redundant with block.label)
            if block.instructions and isinstance(block.instructions[0], LowLevelILLabelInstr):
                instructions_to_format = block.instructions[1:]
            else:
                instructions_to_format = block.instructions

            # Format instructions - now returns list
            indent = '  '
            result.extend(LLILFormatter.format_instruction_sequence(instructions_to_format, indent))
            result.append('')

        return result

    @staticmethod
    def to_dot(func: LowLevelILFunction) -> str:
        '''Generate Graphviz DOT format for CFG visualization

        Args:
            func: LowLevelILFunction to visualize

        Returns:
            DOT format string that can be rendered with:
            - Graphviz: dot -Tpng output.dot -o output.png
            - Online: https://dreampuf.github.io/GraphvizOnline/

        Example:
            from ir.llil_builder import LLILFormatter
            dot = LLILFormatter.to_dot(func)
            with open('cfg.dot', 'w') as f:
                f.write(dot)
        '''
        lines = []
        lines.append(f'digraph "{func.name}" {{')
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, fontname="Courier New", fontsize=10];')
        lines.append('    edge [fontname="Courier New", fontsize=9];')
        lines.append('')

        # Add nodes (basic blocks)
        for block in func.basic_blocks:
            label_parts = []

            # Block header: block_N(0xADDR), label, [sp = N, fp = M]
            # Use same format as format_llil_function
            header_parts = [
                f'block_{block.index}(0x{block.start:X})',
                block.label,
            ]

            # Show sp and fp (fp only on first block)
            if block.index == 0 and func.frame_base_sp is not None:
                header_parts.append(f'[sp = {block.sp_in}, fp = {func.frame_base_sp}]')
            else:
                header_parts.append(f'[sp = {block.sp_in}]')

            header = ', '.join(header_parts) + '\\l'
            label_parts.append(header)
            label_parts.append('-' * 40 + '\\l')

            # Format instructions using expand format (same as format_llil_function)
            # Skip LowLevelILLabelInstr if present
            if block.instructions and isinstance(block.instructions[0], LowLevelILLabelInstr):
                instructions_to_format = block.instructions[1:]
            else:
                instructions_to_format = block.instructions

            # Use format_instruction_sequence to get expanded format
            formatted_lines = LLILFormatter.format_instruction_sequence(instructions_to_format, '')
            for line in formatted_lines:
                # Escape for DOT format
                escaped = line.replace('\\', '\\\\').replace('"', '\\"')
                label_parts.append(escaped + '\\l')

            label = ''.join(label_parts)

            # Node styling
            if block.index == 0:
                # Entry block
                lines.append(f'    {block.block_name} [label="{label}", style=filled, fillcolor=lightgreen];')
            elif block.has_terminal and isinstance(block.instructions[-1], LowLevelILRet):
                # Exit block
                lines.append(f'    {block.block_name} [label="{label}", style=filled, fillcolor=lightblue];')
            else:
                lines.append(f'    {block.block_name} [label="{label}"];')

        lines.append('')

        # Add edges
        for block in func.basic_blocks:
            if not block.outgoing_edges:
                continue

            last_instr = block.instructions[-1] if block.instructions else None

            for target in block.outgoing_edges:
                # Determine edge label and style
                edge_label = ''
                edge_style = ''

                if isinstance(last_instr, LowLevelILIf):
                    # Conditional branch
                    if target == last_instr.true_target:
                        edge_label = 'true'
                        edge_style = ', color=green'
                    elif last_instr.false_target and target == last_instr.false_target:
                        edge_label = 'false'
                        edge_style = ', color=red'
                    else:
                        edge_label = 'fall-through'
                        edge_style = ', style=dashed'
                elif isinstance(last_instr, LowLevelILGoto):
                    edge_label = 'goto'
                    edge_style = ', color=blue'
                else:
                    # Fall-through
                    edge_label = 'fall-through'
                    edge_style = ', style=dashed'

                if edge_label:
                    lines.append(f'    {block.block_name} -> {target.block_name} [label="{edge_label}"{edge_style}];')
                else:
                    lines.append(f'    {block.block_name} -> {target.block_name}{edge_style};')

        lines.append('}')
        return '\n'.join(lines)

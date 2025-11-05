'''
LLIL v2 Builder - Layered architecture for convenience
'''

from typing import Union, Optional, List
from .llil import *


class LowLevelILBuilder:
    '''Mid-level builder with convenience methods'''

    def __init__(self, function: LowLevelILFunction):
        self.function = function
        self.current_block: Optional[LowLevelILBasicBlock] = None
        self.__current_sp: int = 0  # Track current stack pointer state (for block sp_in/sp_out) - PRIVATE
        self.frame_base_sp: Optional[int] = None  # Stack pointer at function entry (for frame-relative access)
        self.__vstack: List[LowLevelILExpr] = []  # Virtual stack for expression tracking - PRIVATE (ONLY stores LowLevelILExpr)

    # === Stack Pointer Management (Public Interface) ===

    def sp_get(self) -> int:
        '''Get current stack pointer value'''
        return self.__current_sp

    def __sp_set(self, value: int):
        '''Set shadow SP to absolute value (does NOT emit IL) - PRIVATE

        Use cases (INTERNAL ONLY):
        1. Initialize block SP state (set_current_block)

        WARNING: This does NOT emit SpAdd IL. Only used internally.
        To modify SP with IL emission, use emit_sp_add().
        '''
        self.__current_sp = value

    def __sp_adjust(self, delta: int):
        '''Adjust shadow SP by delta (does NOT emit IL) - PRIVATE

        This is ONLY called by add_instruction() when it detects SpAdd IL.
        It syncs the shadow SP to reflect the IL that was just added.

        WARNING: NEVER call this directly! Use emit_sp_add() instead.
        '''
        self.__current_sp += delta

    def __cleanup_stack(self, argc: int):
        '''Clean up stack and vstack without emitting IL - PRIVATE

        Adjusts shadow SP and pops from vstack.
        Used for callee cleanup convention where runtime handles stack cleanup.

        Args:
            argc: Number of stack slots to clean up
        '''
        if argc > 0:
            self.__sp_adjust(-argc)
            # Pop from vstack (will raise if underflow - indicates bug in caller)
            for _ in range(argc):
                self.vstack_pop()

    def emit_sp_add(self, delta: int) -> LowLevelILSpAdd:
        '''Emit SpAdd IL and sync shadow sp (single entry point for SP changes)

        This is the ONLY method that should be used to modify SP.
        Ensures all SP changes are represented in the IL.

        Args:
            delta: Number of words to add to sp (can be positive or negative)

        Returns:
            The emitted LowLevelILSpAdd instruction
        '''
        sp_add = LowLevelILSpAdd(delta)
        self.add_instruction(sp_add)
        # Note: add_instruction will handle the sp update via its existing logic
        return sp_add

    # === Virtual Stack Management (Public Interface) ===

    def vstack_push(self, expr: LowLevelILExpr):
        '''Push expression to vstack (only accepts LowLevelILExpr)'''
        if not isinstance(expr, LowLevelILExpr):
            raise TypeError(f'vstack only accepts LowLevelILExpr, got {type(expr).__name__}')
        self.__vstack.append(expr)

    def vstack_pop(self) -> LowLevelILExpr:
        '''Pop expression from vstack (returns LowLevelILExpr)'''
        if not self.__vstack:
            raise RuntimeError('Vstack underflow: attempting to pop from empty vstack')
        return self.__vstack.pop()

    def vstack_peek(self) -> LowLevelILExpr:
        '''Peek at top of vstack without popping (returns LowLevelILExpr)'''
        if not self.__vstack:
            raise RuntimeError('Vstack empty: cannot peek')
        return self.__vstack[-1]

    def vstack_size(self) -> int:
        '''Get current vstack size'''
        return len(self.__vstack)

    def set_current_block(self, block: LowLevelILBasicBlock):
        '''Set the current basic block for instruction insertion

        Args:
            block: The block to set as current

        Raises:
            RuntimeError: If block has not been added to function
        '''
        # Verify block has been added to function
        if block not in self.function.basic_blocks:
            raise RuntimeError(f'Block {block} has not been added to function. Call function.add_basic_block() first.')

        # Save previous block's sp_out if we have a current block
        if self.current_block is not None:
            self.current_block.sp_out = self.sp_get()

        self.current_block = block

        # Set new block's sp_in and current sp
        if self.frame_base_sp is None:
            # First block: use function's parameter count as initial sp
            self.__sp_set(self.function.num_params)
        # else: continue from previous block's sp

        block.sp_in = self.sp_get()

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
            self.__sp_adjust(instr.delta)

    # === Virtual Stack Management ===

    def _to_expr(self, value: Union[LowLevelILExpr, int, float, str]) -> LowLevelILExpr:
        '''Convert value to expression (always returns LowLevelILExpr)'''
        if isinstance(value, LowLevelILExpr):
            return value
        elif isinstance(value, LowLevelILInstruction):
            # Should not happen - only Expr should be passed
            raise TypeError(f'Expected LowLevelILExpr, got {type(value).__name__}. Statements cannot be used as expressions.')
        elif isinstance(value, int):
            return self.const_int(value)
        elif isinstance(value, float):
            return self.const_float(value)
        elif isinstance(value, str):
            return self.const_str(value)
        else:
            raise TypeError(f'Cannot convert {type(value)} to expression')

    def push(self, value: Union[LowLevelILExpr, int, float, str], size: int = 4) -> LowLevelILExpr:
        '''Push value onto stack (SPEC-compliant: StackStore + SpAdd)

        Generates:
          1. StackStore(sp+0, value)
          2. SpAdd(+1)

        Args:
            value: Expression or primitive value to push (must be LowLevelILExpr or int/float/str)
            size: Size in bytes

        Returns:
            The expression that was pushed (LowLevelILExpr)
        '''
        expr = self._to_expr(value)
        # 1. StackStore(sp+0, value)
        self.add_instruction(LowLevelILStackStore(expr, offset=0, size=size))
        # 2. SpAdd(+1)
        self.emit_sp_add(1)
        # Track on vstack for expression tracking
        self.vstack_push(expr)
        return expr

    def pop(self, size: int = 4) -> LowLevelILExpr:
        '''Pop value from stack

        Returns the expression from vstack (will raise if empty).
        '''
        self.__sp_adjust(-1)
        return self.vstack_pop()

    # === Legacy Stack Operations (kept for compatibility) ===

    def stack_push(self, value: Union[LowLevelILExpr, int, str], size: int = 4):
        '''STACK[sp++] = value (legacy, use push() instead)'''
        self.push(value, size)

    def stack_pop(self, size: int = 4) -> LowLevelILExpr:
        '''STACK[--sp] (legacy, use pop() instead)'''
        return self.pop(size)

    def stack_load(self, offset: int, size: int = 4) -> LowLevelILStackLoad:
        '''STACK[sp + offset] (no sp change) - returns expression'''
        return LowLevelILStackLoad(offset, size)

    def stack_store(self, value: Union[LowLevelILExpr, int, str], offset: int, size: int = 4):
        '''STACK[sp + offset] = value (no sp change)'''
        expr = self._to_expr(value)
        self.add_instruction(LowLevelILStackStore(expr, offset, size))

    def frame_load(self, offset: int, size: int = 4) -> 'LowLevelILFrameLoad':
        '''STACK[frame + offset] - Frame-relative load (for function parameters/locals)

        Args:
            offset: Byte offset relative to frame base (function entry sp)
            size: Size in bytes (default 4)

        Returns:
            FrameLoad expression
        '''
        return LowLevelILFrameLoad(offset, size)

    def frame_store(self, value: Union[LowLevelILExpr, int, str], offset: int, size: int = 4):
        '''STACK[frame + offset] = value - Frame-relative store (for function parameters/locals)

        Args:
            value: Value to store (expression or primitive)
            offset: Byte offset relative to frame base (function entry sp)
            size: Size in bytes (default 4)
        '''
        expr = self._to_expr(value)
        self.add_instruction(LowLevelILFrameStore(expr, offset, size))

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
        absolute_pos = self.sp_get() + word_offset

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
        slot_index = self.sp_get() + word_offset
        # Create stack address with absolute slot index
        stack_addr = LowLevelILStackAddr(slot_index)
        self.stack_push(stack_addr)

    # REMOVED: sp_add() - use emit_sp_add() instead
    # def sp_add(self, delta: int):
    #     '''DEPRECATED: Use emit_sp_add() instead'''

    # === Register Operations ===

    def reg_store(self, reg_index: int, value: Union[LowLevelILExpr, int], size: int = 4):
        '''R[index] = value (store expression to register)'''
        expr = self._to_expr(value)
        self.add_instruction(LowLevelILRegStore(reg_index, expr, size))

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

    def const_raw(self, value: int, size: int = 4) -> LowLevelILConst:
        '''Raw constant (type-less, displayed as hex)

        Args:
            value: Raw value
            size: Size in bytes (default 4)
        '''
        return LowLevelILConst(value, size, is_hex = False, is_raw = True)

    # === Binary Operations ===

    def _binary_op(self, op_class, lhs = None, rhs = None, *, push: bool = True, size: int = 4) -> LowLevelILExpr:
        '''Generic binary operation handler

        Stack operation order (for implicit mode):
          rhs = stack_pop();   // First pop gets right operand (top of stack)
          lhs = stack_pop();   // Second pop gets left operand (below it)
          result = (lhs OP rhs);

        Args:
            op_class: The operation class (e.g., LowLevelILAdd)
            lhs: Left operand (None = pop from vstack) - must be expr or primitive
            rhs: Right operand (None = pop from vstack) - must be expr or primitive
            push: Whether to push result back to vstack
            size: Operation size

        Returns:
            The operation expression (LowLevelILExpr)
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

    # === Bitwise Operations ===

    def bitwise_and(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''Bitwise AND operation - computes lhs & rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILAnd, lhs, rhs, push = push, size = size)

    def bitwise_or(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''Bitwise OR operation - computes lhs | rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILOr, lhs, rhs, push = push, size = size)

    # === Logical Operations ===

    def logical_and(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''Logical AND operation - computes lhs && rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILLogicalAnd, lhs, rhs, push = push, size = size)

    def logical_or(self, lhs = None, rhs = None, *, push: bool = True, size: int = 4):
        '''Logical OR operation - computes lhs || rhs (pops rhs first, then lhs)'''
        return self._binary_op(LowLevelILLogicalOr, lhs, rhs, push = push, size = size)

    # === Unary Operations ===

    def neg(self, operand = None, *, push: bool = True, size: int = 4):
        '''NEG operation - arithmetic negation -x (pops operand if not provided)'''
        if operand is None:
            operand = self.pop(size)
        else:
            operand = self._to_expr(operand)
        op = LowLevelILNeg(operand, size)
        if push:
            self.push(op, size)
        else:
            self.add_instruction(op)
        return op

    def logical_not(self, operand = None, *, push: bool = True, size: int = 4):
        '''NOT operation - logical NOT !x (pops operand if not provided)'''
        if operand is None:
            operand = self.pop(size)
        else:
            operand = self._to_expr(operand)
        op = LowLevelILNot(operand, size)
        if push:
            self.push(op, size)
        else:
            self.add_instruction(op)
        return op

    def test_zero(self, operand = None, *, push: bool = True, size: int = 4):
        '''TEST_ZERO operation - test if x == 0 (pops operand if not provided)'''
        if operand is None:
            operand = self.pop(size)
        else:
            operand = self._to_expr(operand)
        op = LowLevelILTestZero(operand, size)
        if push:
            self.push(op, size)
        else:
            self.add_instruction(op)
        return op

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
             argc: Optional[int] = None):
        '''Function call (terminal instruction)

        Args:
            target: Function name or address
            return_target: Block or label to return to after call (resolved in build_cfg)
            argc: Number of stack slots to clean up (includes func_id + ret_addr + args)
        '''
        # If argc provided, clean up stack (without emitting IL)
        # This is for Falcom VM callee cleanup convention
        if argc is not None:
            self.__cleanup_stack(argc)

        self.add_instruction(LowLevelILCall(target, return_target))

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

    # Map operation to expression template
    __expr_templates = {
        LowLevelILOperation.LLIL_ADD: '{lhs} + {rhs}',
        LowLevelILOperation.LLIL_SUB: '{lhs} - {rhs}',
        LowLevelILOperation.LLIL_MUL: '{lhs} * {rhs}',
        LowLevelILOperation.LLIL_DIV: '{lhs} / {rhs}',
        LowLevelILOperation.LLIL_EQ: '({lhs} == {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_NE: '({lhs} != {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_LT: '({lhs} < {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_LE: '({lhs} <= {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_GT: '({lhs} > {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_GE: '({lhs} >= {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_AND: '{lhs} & {rhs}',
        LowLevelILOperation.LLIL_OR: '{lhs} | {rhs}',
        LowLevelILOperation.LLIL_LOGICAL_AND: '({lhs} && {rhs}) ? 1 : 0',
        LowLevelILOperation.LLIL_LOGICAL_OR: '({lhs} || {rhs}) ? 1 : 0',
    }

    @staticmethod
    def _format_binary_op_expanded(binary_op: LowLevelILBinaryOp) -> List[str]:
        '''Format binary operation with expanded pseudo-code'''

        template = LLILFormatter.__expr_templates[binary_op.operation]
        expr = template.format(lhs = 'lhs', rhs = 'rhs')

        lines = []
        lines.append(f'rhs = STACK[--sp]  ; {binary_op.rhs}')
        lines.append(f'lhs = STACK[--sp]  ; {binary_op.lhs}')
        lines.append(f'STACK[sp] = {expr}')
        lines.append('sp++')

        return lines

    @staticmethod
    def _format_unary_op_expanded(unary_op: 'LowLevelILUnaryOp') -> List[str]:
        '''Format unary operation with expanded pseudo-code'''
        from .llil import LowLevelILUnaryOp

        # Map operation to expression template
        expr_templates = {
            LowLevelILOperation.LLIL_NEG: '-{operand}',
            LowLevelILOperation.LLIL_NOT: '!{operand} ? 1 : 0',
            LowLevelILOperation.LLIL_TEST_ZERO: '({operand} == 0) ? 1 : 0',
        }

        template = expr_templates.get(unary_op.operation, f'UNARY_OP({unary_op.operation})({{operand}})')
        expr = template.format(operand = 'operand')

        lines = []
        lines.append(f'operand = STACK[--sp]  ; {unary_op.operand}')
        lines.append(f'STACK[sp] = {expr}')
        lines.append('sp++')

        return lines

    @staticmethod
    def format_instruction_expanded(instr: LowLevelILInstruction) -> List[str]:
        '''Format instruction with expanded stack operations (multi-line)

        Returns a list of lines showing explicit stack behavior.
        For binary operations like EQ, MUL, ADD, this shows:
        - Pop operations to get operands
        - The actual operation
        - Push operation for result
        '''

        # StackStore containing a binary operation: expand the binary op
        if isinstance(instr, LowLevelILStackStore) and isinstance(instr.value, LowLevelILBinaryOp):
            return LLILFormatter._format_binary_op_expanded(instr.value)

        # StackStore containing a unary operation: expand the unary op
        if isinstance(instr, LowLevelILStackStore) and isinstance(instr.value, LowLevelILUnaryOp):
            return LLILFormatter._format_unary_op_expanded(instr.value)

        # Binary operations: pop 2, compute, push 1
        if isinstance(instr, LowLevelILBinaryOp):
            return LLILFormatter._format_binary_op_expanded(instr)

        # Unary operations: pop 1, compute, push 1
        if isinstance(instr, LowLevelILUnaryOp):
            return LLILFormatter._format_unary_op_expanded(instr)

        # StackStore: show store operation
        # if isinstance(instr, LowLevelILStackStore):
        #     word_offset = instr.offset // WORD_SIZE
        #     if word_offset == 0:
        #         target = 'STACK[sp]'
        #     elif word_offset > 0:
        #         target = f'STACK[sp + {word_offset}]'
        #     else:
        #         target = f'STACK[sp - {-word_offset}]'

        #     return [f'{target} = {instr.value}']

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
                if isinstance(instr, (LowLevelILStackStore, LowLevelILStackLoad)) and hasattr(instr, 'slot_index') and instr.slot_index is not None:
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

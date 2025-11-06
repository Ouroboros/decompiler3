'''
Falcom VM Builder - High-level builder with Falcom VM patterns
'''

from typing import Union
from ir.llil import *
from ir.llil_builder import *
from .constants import FalcomConstants
from .llil_ext import *


class FalcomVMBuilder(LowLevelILBuilder):
    '''High-level builder with Falcom VM patterns

    Usage:
        builder = FalcomVMBuilder()
        builder.create_function('DOF_ON', 0x1FFDB6, num_params=2)

        # Create blocks (auto-added to function)
        entry = builder.create_basic_block(0x1FFDB6, 'DOF_ON')
        loc_ret = builder.create_basic_block(0x1FFDCB, 'loc_ret')

        # Build instructions
        builder.set_current_block(entry)
        builder.push_func_id()
        builder.push_ret_addr('loc_ret')
        builder.push_int(1)
        builder.call('some_func')

        builder.set_current_block(loc_ret)
        builder.ret()

        return builder.finalize()  # Returns function after validation
    '''

    def __init__(self):
        '''Create builder without function (call create_function first)'''
        # Initialize with a temporary None function
        # Parent class expects a function, but we'll replace it in create_function
        super().__init__(None)
        self.sp_before_call = None  # Track sp before call for automatic cleanup
        self.return_target_block = None  # Track return target basic block for next call
        self.caller_frame_instr = None  # Track PUSH_CALLER_FRAME instruction for call_module
        self._finalized = False

    def create_function(self, name: str, start_addr: int = 0, num_params: int = 0):
        '''Create function inside builder

        Args:
            name: Function name
            start_addr: Function start address
            num_params: Number of parameters

        Example:
            builder = FalcomVMBuilder()
            builder.create_function('DOF_ON', 0x1FFDB6, 2)
            # ... build instructions ...
            return builder.finalize()
        '''
        if self.function is not None:
            raise RuntimeError('Function already created')

        # Create and set the function
        self.function = LowLevelILFunction(name, start_addr, num_params)

    def add_instruction(self, instr):
        '''Override to handle Falcom-specific instructions'''
        # Handle Falcom-specific instructions BEFORE calling parent
        if isinstance(instr, LowLevelILPushCallerFrame):
            # PUSH_CALLER_FRAME pushes 4 values (occupies 4 stack slots)
            instr.slot_index = self.sp_get()  # Record starting slot
            # No sp adjustment here - push_caller_frame will emit SpAdd explicitly
        elif isinstance(instr, LowLevelILCallModule):
            # CALL_MODULE cleans up args + 4 caller frame values
            # No sp adjustment here - call_module will emit SpAdd explicitly
            pass

        # Call parent to add instruction to block and handle SpAdd
        super().add_instruction(instr)

    def create_basic_block(self, start: int, label: str = None) -> LowLevelILBasicBlock:
        '''Create basic block and automatically add to function

        Args:
            start: Block start address
            label: Optional label name (if None, uses default loc_{start:X})

        Returns:
            The created basic block

        Example:
            entry = builder.create_basic_block(0x243C5, 'AV_04_0017')
            loc_ret = builder.create_basic_block(0x243E3, 'loc_243E3')
        '''
        if self.function is None:
            raise RuntimeError('No function created. Call create_function() first.')

        # Get next block index
        index = len(self.function.basic_blocks)
        # Create block
        block = LowLevelILBasicBlock(start, index, label=label)
        # Automatically add to function
        self.function.add_basic_block(block)
        return block

    def finalize(self) -> 'LowLevelILFunction':
        '''Finalize builder and return function

        Returns:
            The constructed LowLevelILFunction

        Raises:
            RuntimeError: If function not created, already finalized, or pending call sequences
        '''

        if self.sp_get() != 0:
            raise RuntimeError(f'Stack is not empty at the end of the function. Current sp: {self.sp_get()}')

        self.function.build_cfg()

        if self.function is None:
            raise RuntimeError('No function created. Call create_function() first.')

        if self._finalized:
            raise RuntimeError('Builder already finalized')

        # Check for pending call setup
        if self.sp_before_call is not None:
            raise RuntimeError(
                f'Function ended with pending call setup at sp={self.sp_before_call}. '
                f'Incomplete call sequence detected (push_func_id without call).'
            )

        if self.return_target_block is not None:
            raise RuntimeError(
                f'Function ended with pending return target "{self.return_target_block.label}". '
                f'Incomplete call sequence detected (push_ret_addr without call).'
            )

        self._finalized = True
        return self.function

    # === Falcom Specific Constants ===

    def push_func_id(self):
        '''Push current function ID - marks the start of call setup'''
        # Verify no pending call setup
        if self.sp_before_call is not None:
            raise RuntimeError(
                f'Previous call setup at sp={self.sp_before_call} not completed. '
                f'Did you forget to call()?'
            )
        # Save sp before we start pushing for the call
        self.sp_before_call = self.sp_get()
        self.stack_push(FalcomConstants.current_func_id())

    def push_ret_addr(self, target: LowLevelILBasicBlock):
        '''Push return address and remember it for call instruction

        Args:
            target: Basic block reference for return target
        '''
        # Verify push_func_id was called first

        if not isinstance(target, LowLevelILBasicBlock):
            raise RuntimeError(f'target must be a LowLevelILBasicBlock, got {type(target)}')

        if self.sp_before_call is None:
            raise RuntimeError(
                'push_ret_addr called without push_func_id. '
                'Call push_func_id() first to set up the call.'
            )
        # Verify no previous return target pending
        if self.return_target_block is not None:
            raise RuntimeError(
                f'Previous return target {self.return_target_block.label} not consumed. '
                f'Did you forget to call()?'
            )

        # Remember block for next call
        self.return_target_block = target
        self.stack_push(FalcomConstants.ret_addr_block(target))

    def push_caller_frame(self, return_target: LowLevelILBasicBlock):
        '''PUSH_CALLER_FRAME operation - save caller frame for module call

        VM behavior: Pushes 4 values onto stack atomically to save caller frame:
          1. funcIndex (current function ID)
          2. retAddr (return address label/block)
          3. currScript (current script index)
          4. 0xF0000000 (context marker)

        This saves the current call frame so a module call can return properly.

        Args:
            return_target: Basic block reference for return target
        '''
        # Verify no pending call setup
        if self.sp_before_call is not None:
            raise RuntimeError(
                f'Previous call setup at sp={self.sp_before_call} not completed. '
                f'Did you forget to call()?'
            )

        # Save sp before we start pushing for the call
        self.sp_before_call = self.sp_get()

        # Prepare the 4 values
        func_id = FalcomConstants.current_func_id()
        ret_addr = FalcomConstants.ret_addr_block(return_target)
        script = FalcomConstants.current_script()
        context_marker = self.const_raw(0xF0000000)

        # Remember block for next call
        self.return_target_block = return_target

        # Create and add the atomic PUSH_CALLER_FRAME instruction
        # This occupies 4 stack slots
        push_frame_instr = LowLevelILPushCallerFrame(func_id, ret_addr, script, context_marker)
        self.add_instruction(push_frame_instr)

        # Emit SpAdd to represent the 4 values pushed
        self.emit_sp_add(4, hidden_for_formatter = False)

        # Save reference for call_module to use
        self.caller_frame_instr = push_frame_instr

    def call(self, target):
        '''Falcom VM call - automatically cleans up stack (callee cleanup convention)'''
        # Verify call setup was done
        if self.return_target_block is None:
            raise RuntimeError(
                'No return target set. Did you forget to call push_ret_addr()?'
            )
        if self.sp_before_call is None:
            raise RuntimeError(
                'No call setup found. Did you forget to call push_func_id()?'
            )

        # Get return block
        return_block = self.return_target_block

        # Calculate argc: total values pushed = sp_get() - sp_before_call
        # This includes: func_id (1) + ret_addr (1) + args (N)
        argc = self.sp_get() - self.sp_before_call

        # Verify we have at least func_id and ret_addr
        if argc < 2:
            raise RuntimeError(
                f'Expected at least 2 values (func_id, ret_addr), got {argc}'
            )

        # Verify sp alignment: restored sp must match target block's entry sp
        # Only check if block has been visited (has instructions or sp_in was explicitly set)
        restored_sp = self.sp_before_call
        if return_block.instructions:
            # Block has been built, verify sp matches
            if return_block.sp_in != restored_sp:
                raise RuntimeError(
                    f'Stack pointer mismatch when connecting to {return_block.block_name}: '
                    f'call restores sp to {restored_sp}, but target block has sp_in={return_block.sp_in}. '
                    f'This indicates inconsistent stack management.'
                )

        # Call with return target and argc for stack cleanup
        # argc will adjust shadow SP to balance the stack
        super().call(target, return_target = return_block, argc = argc)

        # Clean up state
        self.return_target_block = None
        self.sp_before_call = None

    def call_module(self, module: str, func: str, arg_count: int):
        '''CALL_MODULE operation - call a module function

        Args:
            module: Module name (e.g., 'system')
            func: Function name (e.g., 'OnTalkBegin')
            arg_count: Number of arguments on vstack

        Note: Assumes push_caller_frame() was already called.
        Pops arg_count arguments from vstack and creates CALL_MODULE IL.
        The IL includes the caller_frame and arguments, representing the complete call operation.
        Automatically cleans up arg_count + 4 caller frame values from stack.
        '''
        # Verify call setup was done
        if self.return_target_block is None:
            raise RuntimeError(
                'No return target set. Did you forget to call push_caller_frame()?'
            )
        if self.sp_before_call is None:
            raise RuntimeError(
                'No call setup found. Did you forget to call push_caller_frame()?'
            )
        if self.caller_frame_instr is None:
            raise RuntimeError(
                'No caller frame instruction found. Did you forget to call push_caller_frame()?'
            )

        # Get return block
        return_block = self.return_target_block

        # Pop arguments from vstack (in reverse order: last arg first)
        args = []
        for _ in range(arg_count):
            if self.vstack_size() == 0:
                raise RuntimeError(
                    f'Vstack underflow: trying to pop {arg_count} args but only {len(args)} available'
                )
            args.append(self.vstack_pop())
        # Reverse to get correct order (first arg first)
        args.reverse()

        # Create and add CALL_MODULE instruction
        # This represents the call that cleans up args + 4 caller frame values
        call_instr = LowLevelILCallModule(module, func, self.caller_frame_instr, args, return_block)
        self.add_instruction(call_instr)

        # Emit SpAdd to represent stack cleanup (arg_count + 4 caller frame values)
        cleanup_count = arg_count + 4
        self.emit_sp_add(-cleanup_count, hidden_for_formatter = False)

        # Verify stack is balanced
        if self.sp_get() != self.sp_before_call:
            raise RuntimeError(
                f'Stack imbalance in call_module: after cleaning up {arg_count} args + 4 caller frame, '
                f'current_sp={self.sp_get()} but expected sp_before_call={self.sp_before_call}'
            )

        # Add terminal instruction to jump to return block
        self.jmp(return_block)

        # Clean up state
        self.return_target_block = None
        self.sp_before_call = None
        self.caller_frame_instr = None

    # === VM Operations ===

    def push_int(self, value: int, is_hex: bool = False):
        '''PUSH_INT operation

        Args:
            value: Integer value to push
            is_hex: If True, display as hex; if False, display as decimal (default)
        '''
        self.stack_push(self.const_int(value, is_hex = is_hex))

    def push_str(self, value: str):
        '''PUSH_STR operation'''
        self.stack_push(self.const_str(value))

    def push_raw(self, value: int):
        '''PUSH_RAW operation - push raw 4-byte value without type info

        Args:
            value: Raw 32-bit value to push (displayed as hex)
        '''
        self.stack_push(self.const_raw(value))

    def set_reg(self, reg_index: int):
        '''SET_REG operation'''
        # Pop from stack using StackPop expression
        stack_val = self.pop()
        self.reg_store(reg_index, stack_val)

    def get_reg(self, reg_index: int):
        '''GET_REG operation'''
        reg_val = self.reg_load(reg_index)
        self.stack_push(reg_val)

    def pop_to(self, offset: int):
        '''POP_TO operation - pop and store to STACK[sp + offset]

        Args:
            offset: Byte offset relative to sp (after pop)

        Note: After pop, sp is decremented, so STACK[sp + offset] refers
              to the new sp position. For example, POP_TO(-WORD_SIZE) stores the
              popped value to STACK[sp - 1] where sp is the post-pop value.
        '''
        val = self.pop()
        self.add_instruction(LowLevelILStackStore(val, offset, WORD_SIZE))

    def pop_jmp_zero(self, true_target, false_target):
        '''POP_JMP_ZERO operation - branch if popped value is zero

        Args:
            true_target: Block to jump to if value is zero
            false_target: Block to jump to if value is not zero
        '''
        # Pop from stack using StackPop expression
        cond = self.pop()
        # Create EQ(cond, 0) without adding as instruction
        # This is just used as the branch condition expression
        zero = self.const_int(0)
        is_zero = LowLevelILEq(cond, zero)
        # Create If with both targets explicitly specified
        self.add_instruction(LowLevelILIf(is_zero, true_target, false_target))

    def pop_jmp_not_zero(self, true_target, false_target):
        '''POP_JMP_NOT_ZERO operation - branch if popped value is not zero

        Args:
            true_target: Block to jump to if value is not zero
            false_target: Block to jump to if value is zero
        '''
        # NOT_ZERO is the inverse of ZERO, so swap the targets
        self.pop_jmp_zero(false_target, true_target)

    def pop_bytes(self, num_bytes: int, *, hidden_for_formatter: bool = False):
        '''POP operation - discard N bytes from stack

        VM instruction: POP(size)

        Args:
            num_bytes: Number of bytes to pop (e.g., 4 to pop 1 word, 16 to pop 4 words)
            hidden_for_formatter: If True, hide the SpAdd in formatted output (default: False)
        '''
        # Convert bytes to words
        if num_bytes % WORD_SIZE != 0:
            raise ValueError(f'num_bytes ({num_bytes}) must be a multiple of WORD_SIZE ({WORD_SIZE})')

        num_words = num_bytes // WORD_SIZE
        self.emit_sp_add(-num_words, hidden_for_formatter = hidden_for_formatter)

    def pop_n(self, count: int, *, hidden_for_formatter: bool = False):
        '''POP_N operation - discard N slots from stack

        VM instruction: POP_N(count)

        Args:
            count: Number of slots to pop (e.g., 1 to pop 1 slot)
            hidden_for_formatter: If True, hide the SpAdd in formatted output (default: False)
        '''

        if count <= 0:
            raise ValueError(f'count ({count}) must be positive')

        self.emit_sp_add(-count, hidden_for_formatter = hidden_for_formatter)

    def load_global(self, index: int):
        '''LOAD_GLOBAL operation - push global variable onto stack

        Args:
            index: Global variable array index
        '''
        global_val = LowLevelILGlobalLoad(index)
        self.stack_push(global_val)

    def set_global(self, index: int):
        '''SET_GLOBAL operation - pop from stack and store to global

        Args:
            index: Global variable array index
        '''
        val = self.pop(hidden_for_formatter = True)
        self.add_instruction(LowLevelILGlobalStore(index, val))

class FalcomLLILFormatter(LLILFormatter):
    @classmethod
    def _format_global_store_expanded(cls, global_store: 'LowLevelILGlobalStore') -> List[str]:
        '''Format global store with expanded pseudo-code'''
        return [
            f'GLOBAL[{global_store.index}] = STACK[--sp] ; {global_store.value}',
        ]

    @classmethod
    def format_instruction_expanded(cls, instr: LowLevelILInstruction) -> List[str]:
        if isinstance(instr, LowLevelILGlobalStore):
            return cls._format_global_store_expanded(instr)

        return super().format_instruction_expanded(instr)

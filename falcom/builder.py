'''
Falcom VM Builder - High-level builder with Falcom VM patterns
'''

from typing import Union, List
from ir.llil import LowLevelILEq, LowLevelILIf, LowLevelILFunction, LowLevelILBasicBlock
from ir.llil_builder import LowLevelILBuilder
from .constants import FalcomConstants


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
        self.return_target_label = None  # Track return address label for next call
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
        if self.return_target_label is not None:
            raise RuntimeError(
                f'Function ended with pending return target "{self.return_target_label}". '
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
        self.sp_before_call = self.current_sp
        self.stack_push(FalcomConstants.current_func_id())

    def push_ret_addr(self, target: Union[str, LowLevelILBasicBlock]):
        '''Push return address and remember it for call instruction

        Args:
            target: Either a label string or a basic block reference
        '''
        # Verify push_func_id was called first
        if self.sp_before_call is None:
            raise RuntimeError(
                'push_ret_addr called without push_func_id. '
                'Call push_func_id() first to set up the call.'
            )
        # Verify no previous return target pending
        if self.return_target_label is not None:
            raise RuntimeError(
                f'Previous return target {self.return_target_label} not consumed. '
                f'Did you forget to call()?'
            )

        # Handle both string label and block reference
        if isinstance(target, str):
            self.return_target_label = target  # Remember label for next call
            self.stack_push(FalcomConstants.ret_addr(target))
        else:
            # target is a LowLevelILBasicBlock
            self.return_target_label = target.label  # Remember label for next call
            self.stack_push(FalcomConstants.ret_addr_block(target))

    def call(self, target):
        '''Falcom VM call - automatically cleans up stack (callee cleanup convention)'''
        # Verify call setup was done
        if self.return_target_label is None:
            raise RuntimeError(
                'No return target set. Did you forget to call push_ret_addr()?'
            )
        if self.sp_before_call is None:
            raise RuntimeError(
                'No call setup found. Did you forget to call push_func_id()?'
            )

        # Verify return target label can be resolved to a basic block
        return_block = self.get_block_by_label(self.return_target_label)
        if return_block is None:
            raise RuntimeError(
                f'Return target label "{self.return_target_label}" cannot be resolved to a basic block. '
                f'Make sure the block with this label has been created.'
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

        # Call with return target (pass block, not label string)
        super().call(target, return_target = return_block)

        # Clean up state
        self.return_target_label = None
        self.current_sp = self.sp_before_call
        self.sp_before_call = None

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
        from ir.llil import LowLevelILStackStore, WORD_SIZE
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
        # Pop from stack using StackPop expression
        cond = self.pop()
        # Create If with both targets explicitly specified
        self.add_instruction(LowLevelILIf(cond, true_target, false_target))

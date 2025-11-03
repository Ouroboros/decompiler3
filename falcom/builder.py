'''
Falcom VM Builder - High-level builder with Falcom VM patterns
'''

from typing import Union, List
from ir.llil import LowLevelILEq, LowLevelILIf
from ir.llil_builder import LowLevelILBuilder
from .constants import FalcomConstants


class FalcomVMBuilder(LowLevelILBuilder):
    '''High-level builder with Falcom VM patterns'''

    def __init__(self, function):
        super().__init__(function)
        self.sp_before_call = None  # Track sp before call for automatic cleanup
        self.return_target_label = None  # Track return address label for next call

    # === Falcom Specific Constants ===

    def push_func_id(self):
        '''Push current function ID - marks the start of call setup'''
        # Save sp before we start pushing for the call
        self.sp_before_call = self.current_sp
        self.stack_push(FalcomConstants.current_func_id())

    def push_ret_addr(self, label: str):
        '''Push return address and remember it for call instruction'''
        self.return_target_label = label  # Remember for next call
        self.stack_push(FalcomConstants.ret_addr(label))

    def call(self, target):
        '''Falcom VM call - automatically cleans up stack (callee cleanup convention)'''
        # Call with return target
        super().call(target, return_target = self.return_target_label)
        self.return_target_label = None  # Clear after use
        # Falcom VM: callee cleans up all arguments, ret_addr, and func_id
        if self.sp_before_call is not None:
            self.current_sp = self.sp_before_call
            self.sp_before_call = None

    # === Falcom Call Patterns ===

    def falcom_call_simple(self, func_name: str, args: List[Union[int, str]], ret_label: str):
        '''Standard Falcom call pattern'''
        # Push call context
        self.push_func_id()
        self.push_ret_addr(ret_label)

        # Push arguments
        for arg in args:
            if isinstance(arg, int):
                self.stack_push(self.const_int(arg))
            elif isinstance(arg, str):
                self.stack_push(self.const_str(arg))
            else:
                self.stack_push(arg)

        # Call function
        self.call(func_name)

        # Return label
        self.label(ret_label)

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

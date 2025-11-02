'''
Falcom VM Builder - High-level builder with Falcom VM patterns
'''

from typing import Union, List
from ir.llil import LowLevelILEq, LowLevelILBranch
from ir.llil_builder import LowLevelILBuilder
from .constants import FalcomConstants


class FalcomVMBuilder(LowLevelILBuilder):
    '''High-level builder with Falcom VM patterns'''

    def __init__(self, function):
        super().__init__(function)
        self.sp_before_call = None  # Track sp before call for automatic cleanup

    # === Falcom Specific Constants ===

    def push_func_id(self):
        '''Push current function ID - marks the start of call setup'''
        # Save sp before we start pushing for the call
        self.sp_before_call = self.current_sp
        self.stack_push(FalcomConstants.current_func_id())

    def push_ret_addr(self, label: str):
        '''Push return address'''
        self.stack_push(FalcomConstants.ret_addr(label))

    def call(self, target):
        '''Falcom VM call - automatically cleans up stack (callee cleanup convention)'''
        super().call(target)
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

    def load_stack(self, offset: int):
        '''LOAD_STACK operation - loads from sp + offset and pushes result

        Automatically determines whether to use sp-relative or fp-relative addressing:
        - If accessing below frame base (parameters), uses fp
        - Otherwise uses sp
        '''
        from ir.llil import WORD_SIZE

        # Calculate word offset
        word_offset = offset // WORD_SIZE
        # Calculate absolute stack position
        absolute_pos = self.current_sp + word_offset

        # Check if accessing parameter area (below frame base)
        if self.frame_base_sp is not None and absolute_pos < self.frame_base_sp:
            # Accessing below frame (parameters) - use fp-relative
            # Convert to fp-relative offset: we want fp + offset to reach absolute_pos
            # absolute_pos = fp + fp_offset, so fp_offset = absolute_pos - fp
            fp_offset = (absolute_pos - self.frame_base_sp) * WORD_SIZE  # Convert back to bytes
            self.load_frame(fp_offset)
        else:
            # Accessing within current stack frame - use sp-relative
            stack_val = self.stack_load(offset)
            self.stack_push(stack_val)

    def load_frame(self, offset: int):
        '''LOAD_FRAME operation - loads from frame + offset and pushes result

        Use this for accessing function parameters (frame-relative addressing).
        '''
        frame_val = self.frame_load(offset)
        self.stack_push(frame_val)

    def set_reg(self, reg_index: int):
        '''SET_REG operation'''
        # Pop from stack using StackPop expression
        stack_val = self.pop()
        self.reg_store(reg_index, stack_val)

    def get_reg(self, reg_index: int):
        '''GET_REG operation'''
        reg_val = self.reg_load(reg_index)
        self.stack_push(reg_val)

    def pop_jmp_zero(self, target):
        '''POP_JMP_ZERO operation - branch if popped value is zero'''
        # Pop from stack using StackPop expression
        cond = self.pop()
        # Create EQ(cond, 0) without adding as instruction
        # This is just used as the branch condition expression
        zero = self.const_int(0)
        is_zero = LowLevelILEq(cond, zero)
        # Branch if is_zero is true
        self.add_instruction(LowLevelILBranch(is_zero, target))

    def pop_jmp_not_zero(self, target):
        '''POP_JMP_NOT_ZERO operation'''
        # Pop from stack using StackPop expression
        cond = self.pop()
        # Branch if condition != 0
        self.branch_if(cond, target)

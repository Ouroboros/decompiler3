"""
Falcom VM Builder - High-level builder with Falcom VM patterns
"""

from typing import Union, List
from ir.llil import FalcomConstants, LowLevelILVspAdd, LowLevelILStackLoad, LowLevelILBranch
from ir.llil_builder import LowLevelILBuilder


class FalcomVMBuilder(LowLevelILBuilder):
    """High-level builder with Falcom VM patterns"""

    # === Falcom Specific Constants ===

    def push_func_id(self):
        """Push current function ID"""
        self.stack_push(FalcomConstants.current_func_id())

    def push_ret_addr(self, label: str):
        """Push return address"""
        self.stack_push(FalcomConstants.ret_addr(label))

    # === Falcom Call Patterns ===

    def falcom_call_simple(self, func_name: str, args: List[Union[int, str]], ret_label: str):
        """Standard Falcom call pattern"""
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

    # === Common VM Operations ===

    def vm_push_int(self, value: int):
        """VM PUSH_INT operation"""
        self.stack_push(self.const_int(value))

    def vm_push_str(self, value: str):
        """VM PUSH_STR operation"""
        self.stack_push(self.const_str(value))

    def vm_load_stack(self, offset: int):
        """VM LOAD_STACK operation"""
        stack_val = self.stack_load(offset)
        self.stack_push(stack_val)

    def vm_set_reg(self, reg_index: int):
        """VM SET_REG operation"""
        stack_val = self.stack_pop()
        self.reg_store(reg_index, stack_val)

    def vm_get_reg(self, reg_index: int):
        """VM GET_REG operation"""
        reg_val = self.reg_load(reg_index)
        self.stack_push(reg_val)

    def vm_pop_jmp_zero(self, target):
        """VM POP_JMP_ZERO operation"""
        self.add_instruction(LowLevelILVspAdd(-1))  # Pop
        cond = LowLevelILStackLoad(0)  # Get condition
        self.branch_zero(target)  # Use parent's branch_zero which handles label lookup

    def vm_pop_jmp_not_zero(self, target):
        """VM POP_JMP_NOT_ZERO operation"""
        self.add_instruction(LowLevelILVspAdd(-1))  # Pop
        cond = LowLevelILStackLoad(0)  # Get condition
        self.branch_nonzero(target)  # Use parent's branch_nonzero which handles label lookup

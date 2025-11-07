'''
Falcom-specific LLIL Extensions

Defines Falcom VM-specific instructions that extend the generic LLIL.
'''

from enum import IntEnum
from typing import Optional, Union, List, TYPE_CHECKING
from ir.llil import *
from .constants import *


class LowLevelILFalcomOperation(IntEnum):
    '''Falcom-specific LLIL operations

    Independent enum that starts from LLIL_USER_DEFINED to avoid conflicts.
    '''
    LLIL_PUSH_CALLER_FRAME = LowLevelILOperation.LLIL_PUSH_CALLER_FRAME  # Push caller frame (4 values)
    LLIL_CALL_MODULE = LowLevelILOperation.LLIL_CALL_MODULE              # Call module function
    LLIL_GLOBAL_LOAD = LowLevelILOperation.LLIL_USER_DEFINED     # Load from global variable array
    LLIL_GLOBAL_STORE = LowLevelILOperation.LLIL_USER_DEFINED + 1  # Store to global variable array


class LowLevelILPushCallerFrame(LowLevelILStatement):
    '''PUSH_CALLER_FRAME - Push caller frame to save call context (statement)

    Pushes 4 values onto stack for Falcom VM call convention:
      1. funcIndex (current function ID)
      2. retAddr (return address label/block)
      3. currScript (current script index)
      4. context_marker (0xF0000000)

    This is an atomic operation that occupies 4 stack slots.
    Saves the caller frame so the module call can return properly.
    '''

    def __init__(
        self,
        func_id: 'LowLevelILConstFuncId',
        ret_addr: Union['LowLevelILConstRetAddr', 'LowLevelILConstRetAddrBlock'],
        script_ptr: 'LowLevelILConstScript',
        context_marker: LowLevelILConst
    ):
        super().__init__(LowLevelILFalcomOperation.LLIL_PUSH_CALLER_FRAME)
        self.func_id          = func_id                     # FalcomConstants.current_func_id()
        self.ret_addr         = ret_addr                    # FalcomConstants.ret_addr() or ret_addr_block()
        self.script_ptr       = script_ptr                  # FalcomConstants.current_script()
        self.context_marker   = context_marker              # Raw value (0xF0000000)
        self.slot_index       : Optional[int] = None        # Will be set by builder to track stack position

    def __str__(self) -> str:
        return f'push_caller_frame({self.ret_addr})'


class LowLevelILCallModule(LowLevelILCall):
    '''CALL_MODULE - Call a module function with automatic stack cleanup (statement)

    Calls a module function and cleans up arguments + caller frame from stack.
    All operands (caller frame + arguments) are popped from vstack.

    Stack layout (bottom to top):
      - caller frame (4 values): funcIndex, retAddr, currScript, context_marker
      - arg0, arg1, ..., argN (arguments)
    '''

    def __init__(
        self,
        module: str,
        func: str,
        caller_frame: LowLevelILPushCallerFrame,
        args: List['LowLevelILExpr'],
        return_target: 'LowLevelILBasicBlock'
    ):
        target = f'{module}.{func}'
        super().__init__(target, return_target)
        self.operation    = LowLevelILFalcomOperation.LLIL_CALL_MODULE
        self.module       = module          # Module name (e.g., 'system')
        self.func         = func            # Function name (e.g., 'OnTalkBegin')
        self.caller_frame = caller_frame    # Caller frame (popped from vstack)
        self.args         = args            # Arguments (popped from vstack)
        self.arg_count    = len(args)       # Number of arguments

    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.args)
        return f'call_module(`{self.module}.{self.func}`, [{args_str}]) -> {self.return_target.label}'


class LowLevelILGlobalLoad(LowLevelILExpr):
    '''LOAD_GLOBAL - Load from global variable array (expression)

    Reads from VM's global variable array and pushes to stack.
    '''

    def __init__(self, index: int):
        super().__init__(LowLevelILFalcomOperation.LLIL_GLOBAL_LOAD)
        self.index = index

    def __str__(self) -> str:
        return f'GLOBAL[{self.index}]'


class LowLevelILGlobalStore(LowLevelILStatement):
    '''SET_GLOBAL - Store to global variable array (statement)

    Pops value from stack and writes to VM's global variable array.
    '''

    def __init__(self, index: int, value: LowLevelILExpr):
        super().__init__(LowLevelILFalcomOperation.LLIL_GLOBAL_STORE)
        self.index = index
        self.value = value  # Expression popped from stack

    def __str__(self) -> str:
        return f'GLOBAL[{self.index}] = {self.value}'

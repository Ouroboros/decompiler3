'''
Falcom-specific LLIL Extensions

Defines Falcom VM-specific instructions that extend the generic LLIL.
'''

from enum import IntEnum
from typing import Optional
from ir.llil import LowLevelILInstruction, LowLevelILOperation


class LowLevelILFalcomOperation(IntEnum):
    '''Falcom-specific LLIL operations

    Independent enum that starts from LLIL_USER_DEFINED to avoid conflicts.
    '''
    LLIL_GLOBAL_LOAD = LowLevelILOperation.LLIL_USER_DEFINED     # Load from global variable array
    LLIL_GLOBAL_STORE = LowLevelILOperation.LLIL_USER_DEFINED + 1  # Store to global variable array


class LowLevelILGlobalLoad(LowLevelILInstruction):
    '''LOAD_GLOBAL - Load from global variable array (value expression)

    Reads from VM's global variable array and pushes to stack.
    '''

    def __init__(self, index: int):
        super().__init__(LowLevelILFalcomOperation.LLIL_GLOBAL_LOAD)
        self.index = index

    def __str__(self) -> str:
        return f'GLOBAL[{self.index}]'


class LowLevelILGlobalStore(LowLevelILInstruction):
    '''SET_GLOBAL - Store to global variable array (statement)

    Pops value from stack and writes to VM's global variable array.
    '''

    def __init__(self, index: int, value: LowLevelILInstruction):
        super().__init__(LowLevelILFalcomOperation.LLIL_GLOBAL_STORE)
        self.index = index
        self.value = value  # Expression popped from stack

    def __str__(self) -> str:
        return f'GLOBAL[{self.index}] = {self.value}'

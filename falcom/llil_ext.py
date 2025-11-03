'''
Falcom-specific LLIL Extensions

Defines Falcom VM-specific instructions that extend the generic LLIL.
'''

from enum import IntEnum
from typing import Optional
from ir.llil import LowLevelILInstruction


class LowLevelILFalcomOperation(IntEnum):
    '''Falcom-specific LLIL operations

    Start from 1000 to avoid conflicts with generic LLIL operations.
    '''
    LLIL_GLOBAL_LOAD = 1000   # Load from global variable array
    LLIL_GLOBAL_STORE = 1001  # Store to global variable array


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

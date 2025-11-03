'''
Falcom VM specific constants and types
'''

from typing import TYPE_CHECKING
from ir.llil import LowLevelILConst, WORD_SIZE

if TYPE_CHECKING:
    from ir.llil import LowLevelILBasicBlock


class LowLevelILConstFuncId(LowLevelILConst):
    '''Falcom VM function ID constant'''

    def __init__(self):
        # Special constant with no actual value
        super().__init__(None, WORD_SIZE, False)

    def __str__(self) -> str:
        return '<func_id>'


class LowLevelILConstRetAddr(LowLevelILConst):
    '''Falcom VM return address constant (label-based)'''

    def __init__(self, label: str):
        # Store label as the value
        super().__init__(label, WORD_SIZE * 2, False)
        self.label = label

    def __str__(self) -> str:
        return f'<&{self.label}>'


class LowLevelILConstRetAddrBlock(LowLevelILConst):
    '''Falcom VM return address constant (block-based)'''

    def __init__(self, block: 'LowLevelILBasicBlock'):
        # Store block reference as the value
        super().__init__(block, WORD_SIZE * 2, False)
        self.block = block

    def __str__(self) -> str:
        return f'<&{self.block.label}>'


class FalcomConstants:
    '''Falcom VM specific constants factory'''

    @staticmethod
    def current_func_id():
        return LowLevelILConstFuncId()

    @staticmethod
    def ret_addr(label: str):
        return LowLevelILConstRetAddr(label)

    @staticmethod
    def ret_addr_block(block: 'LowLevelILBasicBlock'):
        return LowLevelILConstRetAddrBlock(block)

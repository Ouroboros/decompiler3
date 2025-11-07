'''
IL Options - Generic options for all IL levels

This module provides option classes that can be used across all IL levels
(LLIL, MLIL, HLIL, etc.) to control formatting, processing, and other
metadata without affecting IR semantics.
'''

from common import *


class ILOptions(StrictBase):
    '''
        Attributes:
        hidden_for_formatter: If True, formatter skips this instruction
    '''

    hidden_for_formatter: bool

    def __init__(self):
        self.hidden_for_formatter = False

'''
Medium Level IL core structures.
'''

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import auto
from typing import Dict, Iterator, List, Optional

from common import *
from ir.core import ILInstruction, ILOptions


class MediumLevelILOperation(IntEnum2):
    '''Atomic MLIL operations. Mostly mirrors LLIL but without stack semantics.'''

    # Constants
    MLIL_CONST_INT        = auto()
    MLIL_CONST_FLOAT      = auto()
    MLIL_CONST_STR        = auto()

    # Variables
    MLIL_VAR              = auto()
    MLIL_SET_VAR          = auto()
    MLIL_PHI              = auto()

    # Arithmetic / logical
    MLIL_ADD              = auto()
    MLIL_SUB              = auto()
    MLIL_MUL              = auto()
    MLIL_DIV              = auto()
    MLIL_MOD              = auto()
    MLIL_AND              = auto()
    MLIL_OR               = auto()
    MLIL_NOT              = auto()
    MLIL_XOR              = auto()
    MLIL_SHL              = auto()
    MLIL_SHR              = auto()

    # Comparisons
    MLIL_CMP_EQ           = auto()
    MLIL_CMP_NE           = auto()
    MLIL_CMP_LT           = auto()
    MLIL_CMP_LE           = auto()
    MLIL_CMP_GT           = auto()
    MLIL_CMP_GE           = auto()

    # Control flow
    MLIL_GOTO             = auto()
    MLIL_IF               = auto()
    MLIL_RET              = auto()

    # Calls / syscalls / intrinsics
    MLIL_CALL             = auto()
    MLIL_INTRINSIC        = auto()
    MLIL_SYSCALL          = auto()


class MediumLevelILInstruction(ILInstruction, ABC):
    '''Base class for all MLIL instructions.'''

    def __init__(self, operation: MediumLevelILOperation):
        super().__init__()
        self.operation = operation
        self.address = 0
        self.inst_index = -1  # inherit from LLIL instruction index when translated
        self.options = ILOptions()

    @property
    def operation_name(self) -> str:
        return self.operation.name.replace('MLIL_', '')

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class MediumLevelILExpr(MediumLevelILInstruction):
    '''Expression nodes produce values.'''

    def __str__(self) -> str:
        return self.operation_name


class MediumLevelILStatement(MediumLevelILInstruction):
    '''Statements have side effects only.'''

    def __str__(self) -> str:
        return self.operation_name


class MLILVar:
    '''Logical variable (SSA root).'''

    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    def __str__(self) -> str:
        return self.name


class MLILVarVersion:
    '''Specific assignment/version of a variable (SSA).'''

    def __init__(self, var: MLILVar, version: int):
        self.var = var
        self.version = version

    def __str__(self) -> str:
        return f'{self.var}_{self.version}'


class MediumLevelILBasicBlock:
    '''Block of MLIL statements with CFG edges.'''

    def __init__(self, index: int):
        self.index = index
        self.instructions: List[MediumLevelILStatement] = []
        self.incoming_edges: List['MediumLevelILBasicBlock'] = []
        self.outgoing_edges: List['MediumLevelILBasicBlock'] = []

    def add_instruction(self, inst: MediumLevelILStatement):
        self.instructions.append(inst)

    def add_outgoing_edge(self, block: 'MediumLevelILBasicBlock'):
        if block not in self.outgoing_edges:
            self.outgoing_edges.append(block)
        if self not in block.incoming_edges:
            block.incoming_edges.append(self)

    @property
    def block_name(self) -> str:
        return f'mlil_block_{self.index}'

    def __str__(self) -> str:
        body = '\n'.join(f'    {inst}' for inst in self.instructions)
        return f'{self.block_name}:\n{body}\n'


class MediumLevelILFunction:
    '''Container for MLIL basic blocks and variables.'''

    def __init__(self, name: str, start_addr: int = 0):
        self.name = name
        self.start_addr = start_addr
        self.basic_blocks: List[MediumLevelILBasicBlock] = []
        self.variables: Dict[str, MLILVar] = {}
        self._inst_block_map: Dict[int, MediumLevelILBasicBlock] = {}

    def create_block(self) -> MediumLevelILBasicBlock:
        block = MediumLevelILBasicBlock(len(self.basic_blocks))
        self.basic_blocks.append(block)
        return block

    def register_instruction(self, block: MediumLevelILBasicBlock, inst: MediumLevelILInstruction):
        if inst.inst_index == -1:
            raise RuntimeError('MLIL instruction must inherit inst_index from LLIL translation.')
        self._inst_block_map[inst.inst_index] = block

    def get_block_for_instruction(self, inst_index: int) -> Optional[MediumLevelILBasicBlock]:
        return self._inst_block_map.get(inst_index)

    def iter_blocks(self) -> Iterator[MediumLevelILBasicBlock]:
        return iter(self.basic_blocks)

    def __str__(self) -> str:
        result = [f'; ---- MLIL Function {self.name} ----']
        for block in self.basic_blocks:
            result.append(str(block))
        return '\n'.join(result)

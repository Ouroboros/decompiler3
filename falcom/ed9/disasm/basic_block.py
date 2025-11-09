"""
Basic Block representation
"""

from common import *
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .instruction import Instruction


class BranchKind(IntEnum2):
    """Type of control flow edge"""
    UNCONDITIONAL   = 0     # JMP, CALL return
    TRUE            = 1     # Conditional branch taken
    FALSE           = 2     # Conditional branch not taken


@dataclass
class BasicBlock:
    """Basic block in control flow graph"""
    offset          : int                                                  # Starting offset
    name            : str                 = ''                              # Block label (e.g., 'loc_1234')
    instructions    : list['Instruction'] = field(default_factory = list)
    branches        : list['BasicBlock']  = field(default_factory = list)   # All successors
    true_branches   : list['BasicBlock']  = field(default_factory = list)   # True branch targets
    false_branches  : list['BasicBlock']  = field(default_factory = list)   # False branch targets

    def __post_init__(self):
        if not self.name:
            self.name = f'loc_{self.offset:X}'

    @property
    def end_offset(self) -> int:
        """End offset (exclusive)"""
        if self.instructions:
            last = self.instructions[-1]
            return last.offset + last.size
        return self.offset

    @property
    def terminal(self) -> 'Instruction | None':
        """Terminal instruction (last instruction in block)"""
        return self.instructions[-1] if self.instructions else None

    def add_branch(self, target: 'BasicBlock', kind: BranchKind | None = None) -> 'BasicBlock':
        """Add a branch to target block"""
        if target not in self.branches:
            self.branches.append(target)

        if kind == BranchKind.TRUE and target not in self.true_branches:
            self.true_branches.append(target)

        elif kind == BranchKind.FALSE and target not in self.false_branches:
            self.false_branches.append(target)

        return target

    def insert_branch(self, target: 'BasicBlock') -> 'BasicBlock':
        """Insert branch at beginning of branches list"""
        if target not in self.branches:
            self.branches.insert(0, target)

        return target

    def __str__(self) -> str:
        indent = default_endian()
        lines = [f'BasicBlock {self.name} @ 0x{self.offset:X}']
        for inst in self.instructions:
            lines.append(f'{indent}{inst.offset:08X}: {inst}')

        if self.branches:
            branch_names = ', '.join(b.name for b in self.branches)
            lines.append(f'{indent}→ {branch_names}')

        return '\n'.join(lines)

'''Dead Code Elimination Pass'''

from ir.pipeline import Pass
from ..hlil import (
    HighLevelILFunction,
    HLILBlock,
    HLILIf,
    HLILWhile,
    HLILSwitch,
    HLILReturn,
    HLILBreak,
    HLILContinue,
)


class DeadCodeEliminationPass(Pass):
    '''Remove unreachable code after return/break/continue'''

    def run(self, func: HighLevelILFunction) -> HighLevelILFunction:
        self._remove_unreachable(func.body)
        return func

    def _remove_unreachable(self, block: HLILBlock):
        if not block or not block.statements:
            return

        for stmt in block.statements:
            if isinstance(stmt, HLILIf):
                self._remove_unreachable(stmt.true_block)
                self._remove_unreachable(stmt.false_block)

            elif isinstance(stmt, HLILWhile):
                self._remove_unreachable(stmt.body)

            elif isinstance(stmt, HLILSwitch):
                for case in stmt.cases:
                    self._remove_unreachable(case.body)

        for i, stmt in enumerate(block.statements):
            if isinstance(stmt, (HLILReturn, HLILBreak, HLILContinue)):
                if i + 1 < len(block.statements):
                    block.statements = block.statements[:i + 1]
                break

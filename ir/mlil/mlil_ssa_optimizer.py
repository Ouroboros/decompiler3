'''MLIL SSA Optimizer - orchestrates SSA optimization passes'''

from typing import Optional

from ir.pipeline import PassObserver
from .mlil import MediumLevelILFunction
from .passes import (
    NNFPass,
    SCCP,
    ConstantFolder,
    ConstantPropagationPass,
    CopyPropagationPass,
    ExpressionSimplificationPass,
    ConditionSimplificationPass,
    ExpressionInliningPass,
    SSADeadCodeEliminationPass,
    DeadPhiSourceEliminationPass,
)


class SSAOptimizer:
    '''Orchestrates SSA optimization passes'''

    def __init__(self, function: MediumLevelILFunction, folder: type[ConstantFolder] = ConstantFolder,
                 observer: Optional[PassObserver] = None):
        self.function = function
        self.folder = folder
        # Reports SCCP and every iterated pass as '<pass>#<iteration>'
        self.observer = observer

    def _observe(self, name: str):
        if self.observer is not None:
            self.observer(name, self.function)

    def optimize(self) -> MediumLevelILFunction:
        '''Run all optimization passes'''
        # Run SCCP first for aggressive constant propagation and unreachable code elimination
        sccp = SCCP(self.function, folder = self.folder)
        sccp.run()
        self._observe(SCCP.__name__)

        # Passes to run iteratively until fixpoint
        passes = [
            ConstantPropagationPass(),
            ExpressionSimplificationPass(),
            ConditionSimplificationPass(),
            NNFPass(),
            CopyPropagationPass(),
            ExpressionInliningPass(),
            SSADeadCodeEliminationPass(sccp_replaced_vars = sccp.replaced_vars),
            DeadPhiSourceEliminationPass(sccp_replaced_vars = sccp.replaced_vars),
        ]

        # Iterative optimization until fixpoint
        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            snapshot = self._snapshot()

            for p in passes:
                p.run(self.function)
                self._observe(f'{p.name}#{iteration}')

            if self._snapshot() == snapshot:
                break

        return self.function

    def _snapshot(self) -> str:
        '''Create snapshot of function state for change detection'''
        lines = []
        for block in self.function.basic_blocks:
            for inst in block.instructions:
                lines.append(repr(inst))
        return '\n'.join(lines)

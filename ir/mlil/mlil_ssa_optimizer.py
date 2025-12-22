'''MLIL SSA Optimizer - orchestrates SSA optimization passes'''

from .mlil import MediumLevelILFunction
from .passes import (
    NNFPass,
    SCCP,
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

    def __init__(self, function: MediumLevelILFunction):
        self.function = function

    def optimize(self) -> MediumLevelILFunction:
        '''Run all optimization passes'''
        # Run SCCP first for aggressive constant propagation and unreachable code elimination
        sccp = SCCP(self.function)
        sccp.run()

        # Passes to run iteratively until fixpoint
        passes = [
            ConstantPropagationPass(),
            ExpressionSimplificationPass(),
            ConditionSimplificationPass(),
            NNFPass(),
            CopyPropagationPass(),
            ExpressionInliningPass(),
            SSADeadCodeEliminationPass(),
            DeadPhiSourceEliminationPass(),
        ]

        # Iterative optimization until fixpoint
        max_iterations = 10
        for _ in range(max_iterations):
            snapshot = self._snapshot()

            for p in passes:
                p.run(self.function)

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

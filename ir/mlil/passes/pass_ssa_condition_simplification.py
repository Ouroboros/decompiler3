'''Condition Simplification Pass

Simplify conditionals:
- (expr == 0) → !expr
- (expr != 0) → expr
- !(a == b) → a != b, !(a != b) → a == b; ordered comparisons are false for a NaN operand,
  so !(a < b) is not a >= b and stays negated
'''

from typing import Optional
from ir.pipeline import Pass
from ..mlil import (
    MediumLevelILFunction,
    MediumLevelILInstruction,
    MLILConst,
    MLILLogicalNot,
    MLILEq,
    MLILNe,
    MLILLt,
    MLILLe,
    MLILGt,
    MLILGe,
)
from ..mlil_ssa import MLILIf


class ConditionSimplificationPass(Pass):
    '''Simplify conditional expressions'''

    def run(self, func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Apply condition simplification'''
        for block in func.basic_blocks:
            for i, inst in enumerate(block.instructions):
                if not isinstance(inst, MLILIf):
                    continue

                condition = inst.condition
                new_condition = None

                if isinstance(condition, MLILLogicalNot):
                    inverted = self._invert_comparison(condition.operand)
                    if inverted is not None:
                        new_condition = inverted

                elif isinstance(condition, MLILEq):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        inverted = self._invert_comparison(condition.lhs)
                        if inverted is not None:
                            new_condition = inverted

                        else:
                            new_condition = MLILLogicalNot(condition.lhs)

                elif isinstance(condition, MLILNe):
                    if isinstance(condition.rhs, MLILConst) and condition.rhs.value == 0:
                        if isinstance(condition.lhs, (MLILEq, MLILNe, MLILLt, MLILLe, MLILGt, MLILGe)):
                            new_condition = condition.lhs

                if new_condition is not None:
                    block.instructions[i] = MLILIf(new_condition, inst.true_target, inst.false_target, address = inst.address).copy_metadata_from(inst)

        return func

    def _invert_comparison(self, expr: MediumLevelILInstruction) -> Optional[MediumLevelILInstruction]:
        '''Invert == / !=; None for anything else, ordered comparisons included (NaN)'''
        if isinstance(expr, MLILEq):
            return MLILNe(expr.lhs, expr.rhs)

        elif isinstance(expr, MLILNe):
            return MLILEq(expr.lhs, expr.rhs)

        else:
            return None

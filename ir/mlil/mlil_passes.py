'''MLIL Passes - Pass-based MLIL processing'''

from typing import Optional

from ir.pipeline import *
from ir.llil import *
from .mlil import *
from .llil_to_mlil import *
from .mlil_ssa import *
from .mlil_ssa_optimizer import *
from .mlil_type_inference import *
from .mlil_types import *


class LLILToMLILPass(Pass):
    '''LLIL to MLIL conversion pass (base)'''

    def __init__(self, translator_class: type = None):
        self.translator_class = translator_class or LLILToMLILTranslator

    def run(self, llil_func: LowLevelILFunction) -> MediumLevelILFunction:
        '''Convert LLIL function to MLIL'''
        translator = self.translator_class()
        return translator.translate(llil_func)


class SSAConversionPass(Pass):
    '''Convert MLIL to SSA form'''

    def run(self, mlil_func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Convert to SSA form (in-place)'''
        convert_to_ssa(mlil_func)
        return mlil_func


class SSAOptimizationPass(Pass):
    '''Run SSA-based optimizations'''

    def run(self, mlil_func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Optimize SSA form'''
        optimizer = SSAOptimizer(mlil_func)
        return optimizer.optimize()


class TypeInferencePass(Pass):
    '''Infer types for SSA variables'''

    def __init__(self, signature_db: Optional[FunctionSignatureDB] = None):
        self.signature_db = signature_db

    def run(self, mlil_func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Infer types and store in var_types'''
        from .mlil_ssa import MLILVariableSSA

        ssa_var_types = infer_types(mlil_func, self.signature_db)

        # Map SSA types back to base variables
        base_types = {}
        for ssa_var, typ in ssa_var_types.items():
            base_name = ssa_var.base_var.name

            if base_name in base_types:
                base_types[base_name] = unify_types(base_types[base_name], typ)

            else:
                base_types[base_name] = typ

        mlil_func.var_types = base_types
        return mlil_func


class SSADeconstructionPass(Pass):
    '''Convert SSA back to non-SSA form'''

    def run(self, mlil_func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Deconstruct SSA form (in-place)'''
        convert_from_ssa(mlil_func)
        return mlil_func


class DeadCodeEliminationPass(Pass):
    '''Remove unused variable assignments'''

    def run(self, mlil_func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Eliminate dead code'''
        from .mlil import MLILVar, MLILBinaryOp, MLILUnaryOp, MLILSetVar
        from .mlil import MLILIf, MLILRet, MLILCall, MLILSyscall, MLILCallScript
        from .mlil import MLILStoreGlobal, MLILStoreReg

        var_uses = {}

        def count_uses(expr):
            if isinstance(expr, MLILVar):
                var_name = expr.var.name
                var_uses[var_name] = var_uses.get(var_name, 0) + 1

            elif isinstance(expr, MLILBinaryOp):
                count_uses(expr.lhs)
                count_uses(expr.rhs)

            elif isinstance(expr, MLILUnaryOp):
                count_uses(expr.operand)

            elif isinstance(expr, (MLILCall, MLILSyscall, MLILCallScript)):
                for arg in expr.args:
                    count_uses(arg)

        # Scan all instructions to count uses
        for block in mlil_func.basic_blocks:
            for inst in block.instructions:
                if isinstance(inst, MLILSetVar):
                    count_uses(inst.value)

                elif isinstance(inst, MLILIf):
                    count_uses(inst.condition)

                elif isinstance(inst, MLILRet):
                    if inst.value:
                        count_uses(inst.value)

                elif isinstance(inst, (MLILCall, MLILSyscall, MLILCallScript)):
                    for arg in inst.args:
                        count_uses(arg)

                elif isinstance(inst, MLILStoreGlobal):
                    count_uses(inst.value)

                elif isinstance(inst, MLILStoreReg):
                    count_uses(inst.value)

        # Remove unused assignments
        for block in mlil_func.basic_blocks:
            new_instructions = []
            for inst in block.instructions:
                if not isinstance(inst, MLILSetVar):
                    new_instructions.append(inst)
                    continue

                var_name = inst.var.name
                if var_uses.get(var_name, 0) > 0:
                    new_instructions.append(inst)

            block.instructions = new_instructions

        return mlil_func

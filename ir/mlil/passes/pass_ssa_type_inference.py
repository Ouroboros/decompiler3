'''Type inference pass'''

from typing import Optional

from ir.pipeline import Pass
from ..mlil import MediumLevelILFunction
from ..mlil_type_inference import infer_types, FunctionSignatureDB
from ..mlil_types import unify_types


class TypeInferencePass(Pass):
    '''Infer types for SSA variables'''

    def __init__(self, signature_db: Optional[FunctionSignatureDB] = None):
        self.signature_db = signature_db

    def run(self, mlil_func: MediumLevelILFunction) -> MediumLevelILFunction:
        '''Infer types and store in var_types'''
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

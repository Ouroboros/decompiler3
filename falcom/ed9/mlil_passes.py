'''Falcom ED9 MLIL Passes'''

from ir.pipeline import *
from ir.llil import *
from ir.mlil import *
from ir.mlil.mlil_passes import *
from .mlil_translator import *
from .type_signatures import *


class ED9LLILToMLILPass(LLILToMLILPass):
    '''ED9-specific LLIL to MLIL conversion pass'''

    def __init__(self):
        super().__init__(translator_class=FalcomLLILToMLILTranslator)


class ED9TypeInferencePass(TypeInferencePass):
    '''ED9-specific type inference pass'''

    def __init__(self, parser=None):
        '''Initialize with optional parser for signature extraction'''
        signature_db = ED9TypeSignatures(parser)
        super().__init__(signature_db=signature_db)

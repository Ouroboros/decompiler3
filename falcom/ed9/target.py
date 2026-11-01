'''ED9 (Kuro no Kiseki) as an IR target (ir/core/target.py)'''

from ir.core import StorageModel, Target

from .backend import ED9Hints
from .calls import ED9Calls
from .constant_folder import ED9ConstantFolder
from .metadata import ED9ModuleMetadata
from .vm_values import WORD_SIZE


# VMContext register file (falcom/ed9/spec/vm_spec.md section 8)
REGISTER_COUNT = 16


class ED9Storage(StorageModel):
    global_name     = 'GLOBALS'
    register_name   = 'REGS'
    register_count  = REGISTER_COUNT


class ED9Target(Target):
    name        = 'ed9'
    word_size   = WORD_SIZE
    values      = ED9ConstantFolder
    storage     = ED9Storage
    calls       = ED9Calls
    hints       = ED9Hints
    metadata    = ED9ModuleMetadata

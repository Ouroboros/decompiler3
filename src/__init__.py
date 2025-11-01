"""
Decompiler3: BinaryNinja-style IR System with TypeScript Support

A three-layer IR system (LLIL/MLIL/HLIL + SSA) with bidirectional TypeScript compilation.
"""

try:
    from .pipeline.decompiler import DecompilerPipeline
    from .ir.llil import *
    from .ir.mlil import *
    from .ir.hlil import *
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = ["DecompilerPipeline"]
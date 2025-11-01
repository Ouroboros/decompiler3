"""
BinaryNinja风格的三层IR系统与双向TypeScript编译管道
"""

__version__ = "0.1.0"
__author__ = "Claude"

# 导出新的BinaryNinja风格IR系统
from .ir.common import BaseILInstruction, ILRegister
from .ir.llil import LowLevelILFunction
from .ir.mlil import MediumLevelILFunction
from .ir.hlil import HighLevelILFunction
from .ir.lifter import DecompilerPipeline

__all__ = [
    "BaseILInstruction", "ILRegister",
    "LowLevelILFunction", "MediumLevelILFunction", "HighLevelILFunction",
    "DecompilerPipeline",
]
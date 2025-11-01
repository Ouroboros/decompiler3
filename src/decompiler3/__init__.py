"""
BinaryNinja风格的三层IR系统与双向TypeScript编译管道
"""

__version__ = "0.1.0"
__author__ = "Claude"

# 仅导出确实存在且能正常工作的核心模块
from .ir.base import OperationType, IRFunction, IRBasicBlock, IRExpression, IRVariable
from .ir.hlil import HLILExpression, HLILConstant, HLILVariable, HLILBinaryOp

__all__ = [
    "OperationType", "IRFunction", "IRBasicBlock", "IRExpression", "IRVariable",
    "HLILExpression", "HLILConstant", "HLILVariable", "HLILBinaryOp",
]
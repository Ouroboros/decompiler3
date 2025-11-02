"""
Falcom VM specific constants and types
"""

from ir.llil import LowLevelILInstruction, LowLevelILOperation


class LowLevelILConstFuncId(LowLevelILInstruction):
    """Falcom VM function ID constant"""

    def __init__(self):
        super().__init__(LowLevelILOperation.LLIL_CONST, 4)

    def __str__(self) -> str:
        return "<func_id>"


class LowLevelILConstRetAddr(LowLevelILInstruction):
    """Falcom VM return address constant"""

    def __init__(self, label: str):
        super().__init__(LowLevelILOperation.LLIL_CONST, 8)
        self.label = label

    def __str__(self) -> str:
        return f"<&{self.label}>"


class FalcomConstants:
    """Falcom VM specific constants factory"""

    @staticmethod
    def current_func_id():
        return LowLevelILConstFuncId()

    @staticmethod
    def ret_addr(label: str):
        return LowLevelILConstRetAddr(label)

#!/usr/bin/env python3
"""
Falcom VM Builder - External VM-specific builder

This demonstrates how to create custom builders for specific VMs
without modifying the core IR system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Union
from decompiler3.ir.llil import LowLevelILBuilder, LowLevelILFunction, LowLevelILCall, LowLevelILRet, LowLevelILLabel
from decompiler3.ir.common import ILRegister, TypedOperand, OperandType


class LowLevelILBuilderExtended(LowLevelILBuilder):
    """Extended builder with Binary Ninja stack operations and typed operands"""

    # Stack operations (Binary Ninja style with typed operands)
    def push(self, src: Union['LowLevelILInstruction', TypedOperand], size: int = 4):
        from decompiler3.ir.llil import LowLevelILPush
        return LowLevelILPush(src, size)

    def pop(self, size: int = 4):
        from decompiler3.ir.llil import LowLevelILPop
        return LowLevelILPop(size)

    # Convenience methods for typed operands
    def push_int(self, value: int, size: int = 4):
        """Push integer literal"""
        return self.push(TypedOperand(OperandType.INT, value), size)

    def push_str(self, value: str):
        """Push string literal"""
        return self.push(TypedOperand(OperandType.STR, value), 0)

    def push_float(self, value: float, size: int = 8):
        """Push float literal"""
        return self.push(TypedOperand(OperandType.FLOAT, value), size)

    def push_current_func_id(self):
        """Push current function ID"""
        return self.push(TypedOperand(OperandType.FUNC_ID, None), 4)

    def push_ret_addr(self, label: str):
        """Push return address label"""
        return self.push(TypedOperand(OperandType.RET_ADDR, label), 8)

    def label(self, name: str):
        from decompiler3.ir.llil import LowLevelILLabel
        return LowLevelILLabel(name)

    # Override call to support string function names
    def call_func(self, func_name: str, arguments: Optional[List] = None):
        """Call function by name (VM style)"""
        from decompiler3.ir.llil import LowLevelILCall
        return LowLevelILCall(func_name, arguments)


class FalcomVMBuilder(LowLevelILBuilderExtended):
    """Falcom VM specific builder with game engine patterns"""

    def __init__(self, function: LowLevelILFunction):
        super().__init__(function)
        self._current_func_name = function.name

    # Falcom VM specific instructions
    def DEBUG_SET_LINENO(self, line_no: int) -> LowLevelILLabel:
        """Debug line number marker (often skipped in IR)"""
        return self.label(f"debug_line_{line_no}")

    def map_event_box_set_enable(self, enable: int = None) -> LowLevelILCall:
        """Falcom-specific: Enable/disable event box"""
        if enable is not None:
            self.add_instruction(self.push_int(enable))
        return self.call_func('map_event_box_set_enable')

    def avoice_play(self, voice_id: int = None) -> LowLevelILCall:
        """Falcom-specific: Play voice audio"""
        if voice_id is not None:
            self.add_instruction(self.push_int(voice_id))
        return self.call_func('avoice_play')

    def RETURN(self, value: Union[int, 'LowLevelILInstruction'] = None) -> LowLevelILRet:
        """Falcom VM style return"""
        if value is not None:
            if isinstance(value, int):
                self.add_instruction(self.push_int(value))
                reg_result = ILRegister("reg0", 0, 4)
                self.add_instruction(self.set_reg(reg_result, self.pop()))
            else:
                reg_result = ILRegister("reg0", 0, 4)
                self.add_instruction(self.set_reg(reg_result, value))
        return self.ret()

    # High-level Falcom patterns
    def falcom_function_prologue(self) -> None:
        """Standard Falcom function entry pattern"""
        self.add_instruction(self.push_current_func_id())

    def falcom_function_call(self, func_name: str, ret_label: str, *args) -> None:
        """Standard Falcom function call pattern"""
        # Push function context
        self.add_instruction(self.push_current_func_id())
        self.add_instruction(self.push_ret_addr(ret_label))

        # Push arguments
        for arg in args:
            if isinstance(arg, int):
                self.add_instruction(self.push_int(arg))
            elif isinstance(arg, str):
                self.add_instruction(self.push_str(arg))
            else:
                self.add_instruction(self.push(arg))

        # Call function
        self.add_instruction(self.call_func(func_name))
        self.add_instruction(self.label(ret_label))

    def battle_start(self, battle_id: int, formation: str = None) -> None:
        """Falcom-specific: Start battle sequence"""
        args = [battle_id]
        if formation:
            args.append(formation)
        self.falcom_function_call('battle_start', f'battle_end_{battle_id}', *args)

    def item_give(self, item_id: int, quantity: int = 1) -> None:
        """Falcom-specific: Give item to player"""
        self.falcom_function_call('item_give', f'item_given_{item_id}', item_id, quantity)

    def npc_talk(self, npc_id: int, message: str) -> None:
        """Falcom-specific: NPC dialogue"""
        self.falcom_function_call('npc_talk', f'npc_talk_end_{npc_id}', npc_id, message)

    def bgm_play(self, bgm_id: int, fade_time: int = 1000) -> None:
        """Falcom-specific: Play background music"""
        self.falcom_function_call('bgm_play', f'bgm_started_{bgm_id}', bgm_id, fade_time)

    def cutscene_play(self, cutscene_id: str) -> None:
        """Falcom-specific: Play cutscene"""
        self.falcom_function_call('cutscene_play', f'cutscene_end_{cutscene_id}', cutscene_id)


# Factory function for creating Falcom VM builders
def create_falcom_builder(function: LowLevelILFunction) -> FalcomVMBuilder:
    """Factory to create Falcom VM builder"""
    return FalcomVMBuilder(function)


if __name__ == "__main__":
    # Simple test
    func = LowLevelILFunction("test", 0x1000)
    builder = create_falcom_builder(func)

    print("✅ FalcomVMBuilder created successfully!")
    print(f"   Builder type: {type(builder).__name__}")
    print(f"   Available methods: battle_start, item_give, npc_talk, bgm_play, cutscene_play")
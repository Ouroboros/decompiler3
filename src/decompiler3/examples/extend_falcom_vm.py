#!/usr/bin/env python3
"""
实际示例：扩展Falcom VM指令集

添加游戏特定指令：DRAW_SPRITE, PLAY_SOUND, SAVE_GAME等
"""

import sys
import os

from decompiler3.ir.base import OperationType
from decompiler3.target.capability import TargetCapability, InstructionCapability, DataType, AddressingMode
from decompiler3.target.instruction_selection import InstructionSelector, InstructionPattern, MachineInstruction
from decompiler3.builtin.registry import BuiltinFunction, BuiltinSignature, BuiltinMapping, SideEffect, register_builtin

# 1. 首先扩展IR操作类型（概念上）
# 在实际项目中，需要修改 src/ir/base.py 的 OperationType
class GameOperationType:
    """游戏特定操作类型（扩展概念）"""
    DRAW_SPRITE = "DRAW_SPRITE"
    PLAY_SOUND = "PLAY_SOUND"
    SAVE_GAME = "SAVE_GAME"
    LOAD_GAME = "LOAD_GAME"
    SHOW_DIALOG = "SHOW_DIALOG"
    SET_FLAG = "SET_FLAG"
    GET_FLAG = "GET_FLAG"

# 2. 扩展Falcom VM能力
class ExtendedFalcomVMCapability(TargetCapability):
    """扩展的Falcom VM，支持游戏特定指令"""

    def __init__(self):
        super().__init__("falcom_vm_extended")
        self.pointer_size = 4
        self.word_size = 4
        self.is_stack_machine = True

        # 继承基础能力
        self._add_basic_vm_instructions()
        # 添加游戏扩展指令
        self._add_game_instructions()

    def _add_basic_vm_instructions(self):
        """添加基础VM指令"""
        for op in [OperationType.ADD, OperationType.SUB, OperationType.MUL]:
            self.add_instruction_capability(InstructionCapability(
                op, [DataType.INT32], [], latency=1, throughput=1
            ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.CONST, [DataType.INT32], [AddressingMode.IMMEDIATE]
        ))

    def _add_game_instructions(self):
        """添加游戏特定指令"""
        # 图形指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.CALL,  # 使用CALL代表DRAW_SPRITE
            [DataType.INT32, DataType.INT32, DataType.POINTER],
            [AddressingMode.IMMEDIATE],
            latency=10, throughput=1, has_side_effects=True
        ))

        # 音频指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.CALL,  # 音频调用
            [DataType.POINTER, DataType.INT32],
            [AddressingMode.IMMEDIATE],
            latency=5, throughput=1, has_side_effects=True
        ))

# 3. 扩展指令选择器
class ExtendedFalcomInstructionSelector(InstructionSelector):
    """扩展的Falcom指令选择器"""

    def __init__(self):
        # 使用基础能力初始化
        from decompiler3.target.capability import get_target_capability
        base_capability = get_target_capability("falcom_vm")
        super().__init__(base_capability)

        # 添加游戏特定模式
        self._add_game_patterns()

    def _add_game_patterns(self):
        """添加游戏指令模式"""
        from decompiler3.ir.mlil import MLILBuiltinCall

        # 精灵绘制模式
        draw_sprite_pattern = InstructionPattern(
            "falcom_draw_sprite",
            lambda expr: (isinstance(expr, MLILBuiltinCall) and
                         expr.builtin_name == "draw_sprite"),
            ["DRAW_SPRITE"],
            cost=10
        )
        self.patterns.append(draw_sprite_pattern)

        # 音效播放模式
        play_sound_pattern = InstructionPattern(
            "falcom_play_sound",
            lambda expr: (isinstance(expr, MLILBuiltinCall) and
                         expr.builtin_name == "play_sound"),
            ["PLAY_SE"],
            cost=5
        )
        self.patterns.append(play_sound_pattern)

    def _select_game_instruction(self, expr) -> list[MachineInstruction]:
        """选择游戏特定指令"""
        from decompiler3.ir.mlil import MLILBuiltinCall

        if isinstance(expr, MLILBuiltinCall):
            if expr.builtin_name == "draw_sprite":
                # DRAW_SPRITE sprite_id, x, y
                return [MachineInstruction("DRAW_SPRITE",
                       [str(arg) for arg in expr.arguments[:3]])]

            elif expr.builtin_name == "play_sound":
                # PLAY_SE sound_id, volume
                return [MachineInstruction("PLAY_SE",
                       [str(arg) for arg in expr.arguments[:2]])]

            elif expr.builtin_name == "save_game":
                # SAVE_GAME slot_id
                return [MachineInstruction("SAVE_GAME",
                       [str(expr.arguments[0]) if expr.arguments else "0"])]

        return []

# 4. 扩展字节码编码器
class ExtendedFalcomAssembler:
    """扩展的Falcom字节码编码器"""

    def __init__(self):
        # 基础指令编码
        self.base_opcodes = {
            "CONST": 0x01,
            "ADD": 0x02,
            "SUB": 0x03,
            "MUL": 0x04,
            "CALL": 0x11,
            "RET": 0x10,
        }

        # 游戏扩展指令编码
        self.game_opcodes = {
            "DRAW_SPRITE": 0x40,   # 绘制精灵
            "PLAY_SE": 0x41,       # 播放音效
            "PLAY_BGM": 0x42,      # 播放背景音乐
            "SAVE_GAME": 0x43,     # 保存游戏
            "LOAD_GAME": 0x44,     # 加载游戏
            "SHOW_MSG": 0x45,      # 显示消息
            "SET_FLAG": 0x46,      # 设置标志
            "GET_FLAG": 0x47,      # 获取标志
            "JUMP_IF": 0x48,       # 条件跳转
            "FADE_IN": 0x49,       # 淡入效果
            "FADE_OUT": 0x4A,      # 淡出效果
        }

        self.opcodes = {**self.base_opcodes, **self.game_opcodes}

    def assemble(self, instructions: list[MachineInstruction]) -> bytes:
        """汇编扩展指令到字节码"""
        bytecode = bytearray()

        for instr in instructions:
            if instr.opcode.endswith(":"):
                continue  # 跳过标签

            opcode = self.opcodes.get(instr.opcode, 0xFF)
            bytecode.append(opcode)

            # 编码操作数
            if instr.opcode == "DRAW_SPRITE":
                # DRAW_SPRITE sprite_id(2), x(2), y(2)
                sprite_id = int(instr.operands[0]) if len(instr.operands) > 0 else 0
                x = int(instr.operands[1]) if len(instr.operands) > 1 else 0
                y = int(instr.operands[2]) if len(instr.operands) > 2 else 0

                bytecode.extend(sprite_id.to_bytes(2, 'little'))
                bytecode.extend(x.to_bytes(2, 'little'))
                bytecode.extend(y.to_bytes(2, 'little'))

            elif instr.opcode == "PLAY_SE":
                # PLAY_SE sound_id(2), volume(1)
                sound_id = int(instr.operands[0]) if len(instr.operands) > 0 else 0
                volume = int(instr.operands[1]) if len(instr.operands) > 1 else 100

                bytecode.extend(sound_id.to_bytes(2, 'little'))
                bytecode.append(volume)

            elif instr.opcode == "SAVE_GAME":
                # SAVE_GAME slot_id(1)
                slot_id = int(instr.operands[0]) if len(instr.operands) > 0 else 0
                bytecode.append(slot_id)

            elif instr.opcode == "SHOW_MSG":
                # SHOW_MSG text_id(2)
                text_id = int(instr.operands[0]) if len(instr.operands) > 0 else 0
                bytecode.extend(text_id.to_bytes(2, 'little'))

            elif instr.opcode in ["CONST", "ADD", "SUB", "MUL"]:
                # 基础指令编码
                for operand in instr.operands:
                    if operand.isdigit():
                        value = int(operand)
                        bytecode.extend(value.to_bytes(4, 'little'))

        return bytes(bytecode)

# 5. 注册游戏Built-in函数
def register_game_builtins():
    """注册游戏相关的Built-in函数"""

    # 绘制精灵
    draw_sprite_builtin = BuiltinFunction(BuiltinSignature(
        name="draw_sprite",
        parameters=["number", "number", "number"],  # sprite_id, x, y
        return_type=None,
        side_effects=[SideEffect.IO],
        description="Draw sprite at specified position",
        category="graphics"
    ))
    draw_sprite_builtin.add_target_mapping("falcom_vm_extended", BuiltinMapping(
        direct_opcode="DRAW_SPRITE"
    ))
    register_builtin(draw_sprite_builtin)

    # 播放音效
    play_sound_builtin = BuiltinFunction(BuiltinSignature(
        name="play_sound",
        parameters=["number", "number"],  # sound_id, volume
        return_type=None,
        side_effects=[SideEffect.IO],
        description="Play sound effect",
        category="audio"
    ))
    play_sound_builtin.add_target_mapping("falcom_vm_extended", BuiltinMapping(
        direct_opcode="PLAY_SE"
    ))
    register_builtin(play_sound_builtin)

    # 保存游戏
    save_game_builtin = BuiltinFunction(BuiltinSignature(
        name="save_game",
        parameters=["number"],  # slot_id
        return_type="boolean",
        side_effects=[SideEffect.IO, SideEffect.SYSTEM],
        description="Save game to specified slot",
        category="system"
    ))
    save_game_builtin.add_target_mapping("falcom_vm_extended", BuiltinMapping(
        direct_opcode="SAVE_GAME"
    ))
    register_builtin(save_game_builtin)

# 6. 完整演示
def demonstrate_extended_vm():
    """演示扩展VM的完整功能"""
    print("扩展Falcom VM演示")
    print("=" * 30)

    # 注册游戏built-ins
    register_game_builtins()

    # 创建扩展能力
    extended_capability = ExtendedFalcomVMCapability()
    print(f"✓ 创建扩展VM能力: {extended_capability.name}")

    # 创建指令选择器
    selector = ExtendedFalcomInstructionSelector()
    print("✓ 创建扩展指令选择器")

    # 创建汇编器
    assembler = ExtendedFalcomAssembler()
    print("✓ 创建扩展汇编器")

    # 演示游戏脚本编译
    print("\n游戏脚本示例:")
    game_script = '''
    // 显示开场动画
    draw_sprite(1, 100, 200);  // 绘制角色精灵
    play_sound(10, 80);        // 播放音效
    save_game(1);              // 保存到槽位1
    '''
    print(game_script)

    # 模拟指令序列
    instructions = [
        MachineInstruction("DRAW_SPRITE", ["1", "100", "200"]),
        MachineInstruction("PLAY_SE", ["10", "80"]),
        MachineInstruction("SAVE_GAME", ["1"]),
    ]

    # 汇编到字节码
    bytecode = assembler.assemble(instructions)
    print(f"\n生成字节码: {' '.join(f'{b:02X}' for b in bytecode)}")

    # 解析字节码
    print("\n字节码解析:")
    i = 0
    while i < len(bytecode):
        opcode = bytecode[i]
        if opcode == 0x40:  # DRAW_SPRITE
            sprite_id = int.from_bytes(bytecode[i+1:i+3], 'little')
            x = int.from_bytes(bytecode[i+3:i+5], 'little')
            y = int.from_bytes(bytecode[i+5:i+7], 'little')
            print(f"  DRAW_SPRITE {sprite_id}, {x}, {y}")
            i += 7
        elif opcode == 0x41:  # PLAY_SE
            sound_id = int.from_bytes(bytecode[i+1:i+3], 'little')
            volume = bytecode[i+3]
            print(f"  PLAY_SE {sound_id}, {volume}")
            i += 4
        elif opcode == 0x43:  # SAVE_GAME
            slot_id = bytecode[i+1]
            print(f"  SAVE_GAME {slot_id}")
            i += 2
        else:
            print(f"  UNKNOWN {opcode:02X}")
            i += 1

    print("\n✅ 扩展VM演示完成!")

def main():
    demonstrate_extended_vm()

if __name__ == "__main__":
    main()
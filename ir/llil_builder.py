"""
LLIL v2 Builder - Layered architecture for convenience
"""

from typing import Union, Optional, List
from .llil import *


class LowLevelILBuilder:
    """Mid-level builder with convenience methods"""

    def __init__(self, function: LowLevelILFunction):
        self.function = function
        self.current_block: Optional[LowLevelILBasicBlock] = None

    def set_current_block(self, block: LowLevelILBasicBlock):
        """Set the current basic block for instruction insertion"""
        self.current_block = block

    def add_instruction(self, instr: LowLevelILInstruction):
        """Add instruction to current block"""
        if self.current_block is None:
            raise RuntimeError("No current basic block set")
        self.current_block.add_instruction(instr)

    # === Stack Operations (convenience methods) ===

    def stack_push(self, value: Union[LowLevelILInstruction, int, str], size: int = 4):
        """S[vsp++] = value"""
        self.add_instruction(LowLevelILStackStore(value, 0, size))
        self.add_instruction(LowLevelILVspAdd(1))

    def stack_pop(self, size: int = 4) -> LowLevelILStackLoad:
        """S[--vsp]"""
        self.add_instruction(LowLevelILVspAdd(-1))
        return LowLevelILStackLoad(0, size)

    def stack_load(self, offset: int, size: int = 4) -> LowLevelILStackLoad:
        """S[vsp + offset] (no vsp change)"""
        return LowLevelILStackLoad(offset, size)

    def stack_store(self, value: Union[LowLevelILInstruction, int, str], offset: int, size: int = 4):
        """S[vsp + offset] = value (no vsp change)"""
        self.add_instruction(LowLevelILStackStore(value, offset, size))

    # === Register Operations ===

    def reg_store(self, reg_index: int, value: Union[LowLevelILInstruction, int], size: int = 4):
        """R[index] = value"""
        self.add_instruction(LowLevelILRegStore(reg_index, value, size))

    def reg_load(self, reg_index: int, size: int = 4) -> LowLevelILRegLoad:
        """R[index]"""
        return LowLevelILRegLoad(reg_index, size)

    # === Constants ===

    def const_int(self, value: int, size: int = 4) -> LowLevelILConst:
        """Integer constant"""
        return LowLevelILConst(value, size)

    def const_str(self, value: str) -> LowLevelILConst:
        """String constant"""
        return LowLevelILConst(value, 0)

    # === Arithmetic Operations ===

    def add(self, size: int = 4):
        """Stack-based ADD"""
        self.add_instruction(LowLevelILAdd(size))

    def eq(self, size: int = 4):
        """Stack-based EQ"""
        self.add_instruction(LowLevelILEq(size))

    # === Control Flow ===

    def jmp(self, target: Union[str, int]):
        """Unconditional jump"""
        self.add_instruction(LowLevelILJmp(target))

    def branch_zero(self, target: Union[str, int]):
        """Branch if stack top is zero"""
        self.add_instruction(LowLevelILBranch("zero", target))

    def branch_nonzero(self, target: Union[str, int]):
        """Branch if stack top is nonzero"""
        self.add_instruction(LowLevelILBranch("nonzero", target))

    def call(self, target: Union[str, LowLevelILInstruction]):
        """Function call"""
        self.add_instruction(LowLevelILCall(target))

    def ret(self):
        """Return"""
        self.add_instruction(LowLevelILRet())

    # === Special ===

    def label(self, name: str):
        """Label"""
        self.add_instruction(LowLevelILLabel(name))

    def debug_line(self, line_no: int):
        """Debug line number"""
        self.add_instruction(LowLevelILDebug("line", line_no))

    def syscall(self, catalog: int, cmd: int, arg_count: int):
        """System call"""
        self.add_instruction(LowLevelILSyscall(catalog, cmd, arg_count))


class LLILFormatter:
    """Formatting layer for beautiful output"""

    @staticmethod
    def format_instruction_sequence(instructions: List[LowLevelILInstruction]) -> str:
        """Format sequence with pattern recognition"""
        result = []
        i = 0

        while i < len(instructions):
            instr = instructions[i]

            # Pattern: S[vsp] = value; vsp++ → S[vsp++] = value
            if (isinstance(instr, LowLevelILStackStore) and instr.offset == 0 and
                i + 1 < len(instructions) and
                isinstance(instructions[i + 1], LowLevelILVspAdd) and
                instructions[i + 1].delta == 1):

                result.append(f"S[vsp++] = {instr.value}")
                i += 2  # Skip both instructions

            # Pattern: vsp--; S[vsp] → S[--vsp]
            elif (isinstance(instr, LowLevelILVspAdd) and instr.delta == -1 and
                  i + 1 < len(instructions) and
                  isinstance(instructions[i + 1], LowLevelILStackLoad) and
                  instructions[i + 1].offset == 0):

                result.append("S[--vsp]")
                i += 2  # Skip both instructions

            else:
                result.append(str(instr))
                i += 1

        return "\n".join(f"  {line}" for line in result)

    @staticmethod
    def format_function(func: LowLevelILFunction) -> str:
        """Format entire function with beautiful output"""
        result = f"; ---------- {func.name} ----------\n"

        for block in func.basic_blocks:
            # Block header
            if block.instructions and isinstance(block.instructions[0], LowLevelILLabel):
                result += f"{block.instructions[0].name}: [vsp={block.vsp_in}]\n"
                instructions_to_format = block.instructions[1:]
            else:
                result += f"bb_{hex(block.start)}: [vsp={block.vsp_in}]\n"
                instructions_to_format = block.instructions

            # Format instructions
            result += LLILFormatter.format_instruction_sequence(instructions_to_format)
            result += "\n\n"

        return result
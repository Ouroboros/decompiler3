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
        self._label_map: dict[str, LowLevelILBasicBlock] = {}  # label name -> block

    def set_current_block(self, block: LowLevelILBasicBlock):
        """Set the current basic block for instruction insertion"""
        self.current_block = block

    def mark_label(self, name: str, block: LowLevelILBasicBlock):
        """Associate a label name with a block"""
        self._label_map[name] = block

    def get_block_by_label(self, label: str) -> Optional[LowLevelILBasicBlock]:
        """Get block by label name"""
        return self._label_map.get(label)

    def add_instruction(self, instr: LowLevelILInstruction):
        """Add instruction to current block"""
        if self.current_block is None:
            raise RuntimeError("No current basic block set")
        self.current_block.add_instruction(instr)

    # === Stack Operations (convenience methods) ===

    def stack_push(self, value: Union[LowLevelILInstruction, int, str], size: int = 4):
        """STACK[sp++] = value"""
        self.add_instruction(LowLevelILStackStore(value, 0, size))
        self.add_instruction(LowLevelILSpAdd(1))

    def stack_pop(self, size: int = 4) -> LowLevelILStackLoad:
        """STACK[--sp]"""
        self.add_instruction(LowLevelILSpAdd(-1))
        return LowLevelILStackLoad(0, size)

    def stack_load(self, offset: int, size: int = 4) -> LowLevelILStackLoad:
        """STACK[sp + offset] (no sp change)"""
        return LowLevelILStackLoad(offset, size)

    def stack_store(self, value: Union[LowLevelILInstruction, int, str], offset: int, size: int = 4):
        """STACK[sp + offset] = value (no sp change)"""
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

    def const_float(self, value: float, size: int = 4) -> LowLevelILConst:
        """Float constant (size: 4 for float, 8 for double)"""
        return LowLevelILConst(value, size)

    def const_str(self, value: str) -> LowLevelILConst:
        """String constant"""
        return LowLevelILConst(value, 0)

    # === Arithmetic Operations ===

    def add(self, size: int = 4):
        """Stack-based ADD"""
        self.add_instruction(LowLevelILAdd(size))

    def mul(self, size: int = 4):
        """Stack-based MUL"""
        self.add_instruction(LowLevelILMul(size))

    def eq(self, size: int = 4):
        """Stack-based EQ"""
        self.add_instruction(LowLevelILEq(size))

    # === Control Flow ===

    def jmp(self, target: Union[str, LowLevelILBasicBlock]):
        """Unconditional jump - target can be label or block"""
        if isinstance(target, str):
            target_block = self.get_block_by_label(target)
            if target_block is None:
                raise ValueError(f"Undefined label: {target}")
            target = target_block
        self.add_instruction(LowLevelILJmp(target))

    def branch_zero(self, target: Union[str, LowLevelILBasicBlock]):
        """Branch if stack top is zero - target can be label or block"""
        if isinstance(target, str):
            target_block = self.get_block_by_label(target)
            if target_block is None:
                raise ValueError(f"Undefined label: {target}")
            target = target_block
        self.add_instruction(LowLevelILBranch("zero", target))

    def branch_nonzero(self, target: Union[str, LowLevelILBasicBlock]):
        """Branch if stack top is nonzero - target can be label or block"""
        if isinstance(target, str):
            target_block = self.get_block_by_label(target)
            if target_block is None:
                raise ValueError(f"Undefined label: {target}")
            target = target_block
        self.add_instruction(LowLevelILBranch("nonzero", target))

    def call(self, target: Union[str, LowLevelILInstruction]):
        """Function call"""
        self.add_instruction(LowLevelILCall(target))

    def ret(self):
        """Return"""
        self.add_instruction(LowLevelILRet())

    # === Special ===

    def label(self, name: str):
        """Label - marks current block with this label name"""
        if self.current_block is None:
            raise RuntimeError("No current block to label")
        self.mark_label(name, self.current_block)
        self.add_instruction(LowLevelILLabelInstr(name))

    def debug_line(self, line_no: int):
        """Debug line number"""
        self.add_instruction(LowLevelILDebug("line", line_no))

    def syscall(self, catalog: int, cmd: int, arg_count: int):
        """System call"""
        self.add_instruction(LowLevelILSyscall(catalog, cmd, arg_count))


class LLILFormatter:
    """Formatting layer for beautiful output"""

    @staticmethod
    def format_instruction_sequence(instructions: List[LowLevelILInstruction]) -> list[str]:
        """Format sequence with pattern recognition - returns list of lines"""
        result = []
        i = 0

        while i < len(instructions):
            instr = instructions[i]

            # Pattern: STACK[sp] = value; sp++ → STACK[sp++] = value
            if (isinstance(instr, LowLevelILStackStore) and instr.offset == 0 and
                i + 1 < len(instructions) and
                isinstance(instructions[i + 1], LowLevelILSpAdd) and
                instructions[i + 1].delta == 1):

                result.append(f"  STACK[sp++] = {instr.value}")
                i += 2  # Skip both instructions

            # Pattern: sp--; STACK[sp] → STACK[--sp]
            elif (isinstance(instr, LowLevelILSpAdd) and instr.delta == -1 and
                  i + 1 < len(instructions) and
                  isinstance(instructions[i + 1], LowLevelILStackLoad) and
                  instructions[i + 1].offset == 0):

                result.append("  STACK[--sp]")
                i += 2  # Skip both instructions

            else:
                result.append(f"  {str(instr)}")
                i += 1

        return result

    @staticmethod
    def format_function(func: LowLevelILFunction) -> list[str]:
        """Format entire function with beautiful output - returns list of lines"""
        result = [
            f"; ---------- {func.name} ----------",
        ]

        for block in func.basic_blocks:
            # Block header: {block_N}  label_name: [sp=0]
            block_info = f"block_{block.index}"

            if block.instructions and isinstance(block.instructions[0], LowLevelILLabelInstr):
                label_name = block.instructions[0].name
                result.append(f"{block_info}  {label_name}: [sp={block.sp_in}]")
                instructions_to_format = block.instructions[1:]
            else:
                result.append(f"{block_info}: [sp={block.sp_in}]")
                instructions_to_format = block.instructions

            # Format instructions - now returns list
            result.extend(LLILFormatter.format_instruction_sequence(instructions_to_format))
            result.append("")

        return result

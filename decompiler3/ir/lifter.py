"""
New Lifter System for BinaryNinja-style IR

This module provides lifting between the three IR levels:
LLIL -> MLIL -> HLIL
"""

from typing import List, Optional, Dict, Any
from .llil import (
    LowLevelILFunction, LowLevelILInstruction, LowLevelILBasicBlock,
    LowLevelILAdd, LowLevelILSub, LowLevelILMul, LowLevelILConst,
    LowLevelILReg, LowLevelILSetReg, LowLevelILLoad, LowLevelILStore,
    LowLevelILJump, LowLevelILGoto, LowLevelILIf, LowLevelILCall, LowLevelILRet
)
from .mlil import (
    MediumLevelILFunction, MediumLevelILInstruction, MediumLevelILBasicBlock,
    MediumLevelILBuilder, MediumLevelILAdd, MediumLevelILSub, MediumLevelILMul,
    MediumLevelILConst, MediumLevelILVar, MediumLevelILSetVar,
    MediumLevelILJump, MediumLevelILGoto, MediumLevelILIf, MediumLevelILCall, MediumLevelILRet,
    Variable
)
from .hlil import (
    HighLevelILFunction, HighLevelILInstruction, HighLevelILBasicBlock,
    HighLevelILBuilder, HighLevelILAdd, HighLevelILSub, HighLevelILMul,
    HighLevelILConst, HighLevelILVar, HighLevelILAssign,
    HighLevelILIf, HighLevelILWhile, HighLevelILCall, HighLevelILRet
)
from .common import ILRegister, InstructionIndex


class LLILToMLILLifter:
    """Lifts LLIL to MLIL"""

    def __init__(self):
        self.register_to_var: Dict[str, Variable] = {}

    def lift_function(self, llil_function: LowLevelILFunction) -> MediumLevelILFunction:
        """Convert LLIL function to MLIL"""
        mlil_function = MediumLevelILFunction(llil_function.name, llil_function.address)

        # Convert basic blocks
        for llil_block in llil_function.basic_blocks:
            mlil_block = MediumLevelILBasicBlock(llil_block.start_address)
            mlil_function.add_basic_block(mlil_block)

            builder = MediumLevelILBuilder(mlil_function)
            builder.set_current_block(mlil_block)

            # Convert instructions
            for llil_instr in llil_block.instructions:
                mlil_instr = self._convert_llil_to_mlil(llil_instr, mlil_function)
                if mlil_instr:
                    builder.add_instruction(mlil_instr)

        return mlil_function

    def _convert_llil_to_mlil(self, llil_instr: LowLevelILInstruction, mlil_function: MediumLevelILFunction) -> Optional[MediumLevelILInstruction]:
        """Convert single LLIL instruction to MLIL"""

        if isinstance(llil_instr, LowLevelILConst):
            return MediumLevelILConst(llil_instr.constant, llil_instr.size)

        elif isinstance(llil_instr, LowLevelILAdd):
            left = self._convert_llil_to_mlil(llil_instr.left, mlil_function)
            right = self._convert_llil_to_mlil(llil_instr.right, mlil_function)
            if left and right:
                return MediumLevelILAdd(left, right, llil_instr.size)

        elif isinstance(llil_instr, LowLevelILSub):
            left = self._convert_llil_to_mlil(llil_instr.left, mlil_function)
            right = self._convert_llil_to_mlil(llil_instr.right, mlil_function)
            if left and right:
                return MediumLevelILSub(left, right, llil_instr.size)

        elif isinstance(llil_instr, LowLevelILMul):
            left = self._convert_llil_to_mlil(llil_instr.left, mlil_function)
            right = self._convert_llil_to_mlil(llil_instr.right, mlil_function)
            if left and right:
                return MediumLevelILMul(left, right, llil_instr.size)

        elif isinstance(llil_instr, LowLevelILReg):
            # Convert register to variable
            var = self._get_or_create_variable(llil_instr.src.name, mlil_function)
            return MediumLevelILVar(var)

        elif isinstance(llil_instr, LowLevelILSetReg):
            # Convert register assignment to variable assignment
            var = self._get_or_create_variable(llil_instr.dest.name, mlil_function)
            value = self._convert_llil_to_mlil(llil_instr.src, mlil_function)
            if value:
                return MediumLevelILSetVar(var, value)

        elif isinstance(llil_instr, LowLevelILJump):
            dest = self._convert_llil_to_mlil(llil_instr.dest, mlil_function)
            if dest:
                return MediumLevelILJump(dest)

        elif isinstance(llil_instr, LowLevelILGoto):
            return MediumLevelILGoto(llil_instr.dest)

        elif isinstance(llil_instr, LowLevelILIf):
            condition = self._convert_llil_to_mlil(llil_instr.condition, mlil_function)
            if condition:
                return MediumLevelILIf(condition, llil_instr.true, llil_instr.false)

        elif isinstance(llil_instr, LowLevelILCall):
            dest = self._convert_llil_to_mlil(llil_instr.dest, mlil_function)
            args = []
            for arg in llil_instr.arguments:
                mlil_arg = self._convert_llil_to_mlil(arg, mlil_function)
                if mlil_arg:
                    args.append(mlil_arg)
            if dest:
                return MediumLevelILCall(dest, args)

        elif isinstance(llil_instr, LowLevelILRet):
            if llil_instr.dest:
                value = self._convert_llil_to_mlil(llil_instr.dest, mlil_function)
                if value:
                    return MediumLevelILRet([value])
            return MediumLevelILRet()

        return None

    def _get_or_create_variable(self, reg_name: str, mlil_function: MediumLevelILFunction) -> Variable:
        """Get or create variable for register"""
        if reg_name not in self.register_to_var:
            var = mlil_function.create_variable(f"var_{reg_name}", "int", 4)
            self.register_to_var[reg_name] = var
        else:
            var = self.register_to_var[reg_name]
        return var


class MLILToHLILLifter:
    """Lifts MLIL to HLIL"""

    def lift_function(self, mlil_function: MediumLevelILFunction) -> HighLevelILFunction:
        """Convert MLIL function to HLIL"""
        hlil_function = HighLevelILFunction(mlil_function.name, mlil_function.address)

        # Copy variables
        for name, var in mlil_function.variables.items():
            hlil_function.create_variable(name, var.var_type, var.size)

        # Convert basic blocks
        for mlil_block in mlil_function.basic_blocks:
            hlil_block = HighLevelILBasicBlock(mlil_block.start_address)
            hlil_function.add_basic_block(hlil_block)

            builder = HighLevelILBuilder(hlil_function)
            builder.set_current_block(hlil_block)

            # Convert instructions
            for mlil_instr in mlil_block.instructions:
                hlil_instr = self._convert_mlil_to_hlil(mlil_instr, hlil_function)
                if hlil_instr:
                    builder.add_instruction(hlil_instr)

        return hlil_function

    def _convert_mlil_to_hlil(self, mlil_instr: MediumLevelILInstruction, hlil_function: HighLevelILFunction) -> Optional[HighLevelILInstruction]:
        """Convert single MLIL instruction to HLIL"""

        if isinstance(mlil_instr, MediumLevelILConst):
            return HighLevelILConst(mlil_instr.constant, mlil_instr.size)

        elif isinstance(mlil_instr, MediumLevelILAdd):
            left = self._convert_mlil_to_hlil(mlil_instr.left, hlil_function)
            right = self._convert_mlil_to_hlil(mlil_instr.right, hlil_function)
            if left and right:
                return HighLevelILAdd(left, right, mlil_instr.size)

        elif isinstance(mlil_instr, MediumLevelILSub):
            left = self._convert_mlil_to_hlil(mlil_instr.left, hlil_function)
            right = self._convert_mlil_to_hlil(mlil_instr.right, hlil_function)
            if left and right:
                return HighLevelILSub(left, right, mlil_instr.size)

        elif isinstance(mlil_instr, MediumLevelILMul):
            left = self._convert_mlil_to_hlil(mlil_instr.left, hlil_function)
            right = self._convert_mlil_to_hlil(mlil_instr.right, hlil_function)
            if left and right:
                return HighLevelILMul(left, right, mlil_instr.size)

        elif isinstance(mlil_instr, MediumLevelILVar):
            var = hlil_function.get_variable(mlil_instr.src.name)
            if var:
                return HighLevelILVar(var)

        elif isinstance(mlil_instr, MediumLevelILSetVar):
            var = hlil_function.get_variable(mlil_instr.dest.name)
            value = self._convert_mlil_to_hlil(mlil_instr.src, hlil_function)
            if var and value:
                return HighLevelILAssign(HighLevelILVar(var), value)

        elif isinstance(mlil_instr, MediumLevelILIf):
            condition = self._convert_mlil_to_hlil(mlil_instr.condition, hlil_function)
            if condition:
                # Simplified - just create basic if without structured control flow
                # In a real implementation, this would do control flow structuring
                return condition  # Placeholder

        elif isinstance(mlil_instr, MediumLevelILCall):
            dest = self._convert_mlil_to_hlil(mlil_instr.dest, hlil_function)
            args = []
            for arg in mlil_instr.arguments:
                hlil_arg = self._convert_mlil_to_hlil(arg, hlil_function)
                if hlil_arg:
                    args.append(hlil_arg)
            if dest:
                return HighLevelILCall(dest, args)

        elif isinstance(mlil_instr, MediumLevelILRet):
            if mlil_instr.src:
                values = []
                for src in mlil_instr.src:
                    hlil_val = self._convert_mlil_to_hlil(src, hlil_function)
                    if hlil_val:
                        values.append(hlil_val)
                return HighLevelILRet(values)
            return HighLevelILRet()

        return None


class DecompilerPipeline:
    """Complete decompilation pipeline using new IR system"""

    def __init__(self):
        self.llil_to_mlil = LLILToMLILLifter()
        self.mlil_to_hlil = MLILToHLILLifter()

    def decompile_function(self, llil_function: LowLevelILFunction) -> HighLevelILFunction:
        """Complete pipeline: LLIL -> MLIL -> HLIL"""
        print(f"🔄 Starting decompilation pipeline for {llil_function.name}")

        # Stage 1: LLIL -> MLIL
        print("   Stage 1: LLIL -> MLIL")
        mlil_function = self.llil_to_mlil.lift_function(llil_function)
        print(f"   ✅ MLIL: {len(mlil_function.basic_blocks)} blocks, {len(mlil_function.variables)} variables")

        # Stage 2: MLIL -> HLIL
        print("   Stage 2: MLIL -> HLIL")
        hlil_function = self.mlil_to_hlil.lift_function(mlil_function)
        print(f"   ✅ HLIL: {len(hlil_function.basic_blocks)} blocks, {len(hlil_function.variables)} variables")

        print(f"🎉 Decompilation complete!")
        return hlil_function

    def create_sample_llil_function(self) -> LowLevelILFunction:
        """Create a sample LLIL function for testing"""
        from .llil import LowLevelILBuilder

        # Create function
        function = LowLevelILFunction("sample_function", 0x1000)
        block = LowLevelILBasicBlock(0x1000)
        function.add_basic_block(block)

        # Create builder
        builder = LowLevelILBuilder(function)
        builder.set_current_block(block)

        # Create registers
        reg_a = ILRegister("eax", 0, 4)
        reg_b = ILRegister("ebx", 1, 4)

        # Build instructions: eax = 10 + 20; return eax
        const1 = builder.const(10)
        const2 = builder.const(20)
        add_result = builder.add(const1, const2)
        set_eax = builder.set_reg(reg_a, add_result)
        get_eax = builder.reg(reg_a)
        ret_stmt = builder.ret(get_eax)

        # Add to block
        instructions = [set_eax, ret_stmt]
        for instr in instructions:
            builder.add_instruction(instr)

        return function

    def create_fibonacci_llil_function(self) -> LowLevelILFunction:
        """Create a fibonacci LLIL function with control flow"""
        from .llil import LowLevelILBuilder

        # Create fibonacci function: int fibonacci(int n)
        function = LowLevelILFunction("fibonacci", 0x2000)

        # Create registers
        reg_n = ILRegister("eax", 0, 4)      # input parameter n
        reg_a = ILRegister("ebx", 1, 4)      # fibonacci(i-2)
        reg_b = ILRegister("ecx", 2, 4)      # fibonacci(i-1)
        reg_i = ILRegister("edx", 3, 4)      # loop counter
        reg_temp = ILRegister("esi", 4, 4)   # temporary

        # Block 0: Entry - check if n <= 1
        entry_block = LowLevelILBasicBlock(0x2000)
        function.add_basic_block(entry_block)
        builder = LowLevelILBuilder(function)
        builder.set_current_block(entry_block)

        # if (n <= 1) goto base_case else goto loop_init
        n_val = builder.reg(reg_n)
        const1 = builder.const(1)
        cmp_result = builder.cmp_sle(n_val, const1)  # n <= 1
        if_stmt = builder.if_stmt(cmp_result, InstructionIndex(1), InstructionIndex(2))
        builder.add_instruction(if_stmt)

        # Block 1: Base case - return n
        base_case_block = LowLevelILBasicBlock(0x2010)
        function.add_basic_block(base_case_block)
        builder.set_current_block(base_case_block)

        ret_n = builder.ret(n_val)
        builder.add_instruction(ret_n)

        # Block 2: Loop initialization
        loop_init_block = LowLevelILBasicBlock(0x2020)
        function.add_basic_block(loop_init_block)
        builder.set_current_block(loop_init_block)

        # a = 0, b = 1, i = 2
        const0 = builder.const(0)
        set_a = builder.set_reg(reg_a, const0)
        set_b = builder.set_reg(reg_b, const1)
        const2 = builder.const(2)
        set_i = builder.set_reg(reg_i, const2)
        builder.add_instruction(set_a)
        builder.add_instruction(set_b)
        builder.add_instruction(set_i)

        # goto loop_condition
        goto_cond = builder.goto(InstructionIndex(3))
        builder.add_instruction(goto_cond)

        # Block 3: Loop condition - while (i <= n)
        loop_cond_block = LowLevelILBasicBlock(0x2030)
        function.add_basic_block(loop_cond_block)
        builder.set_current_block(loop_cond_block)

        i_val = builder.reg(reg_i)
        loop_cmp = builder.cmp_sle(i_val, n_val)  # i <= n
        loop_if = builder.if_stmt(loop_cmp, InstructionIndex(4), InstructionIndex(5))
        builder.add_instruction(loop_if)

        # Block 4: Loop body
        loop_body_block = LowLevelILBasicBlock(0x2040)
        function.add_basic_block(loop_body_block)
        builder.set_current_block(loop_body_block)

        # temp = a + b
        a_val = builder.reg(reg_a)
        b_val = builder.reg(reg_b)
        add_ab = builder.add(a_val, b_val)
        set_temp = builder.set_reg(reg_temp, add_ab)
        builder.add_instruction(set_temp)

        # a = b
        set_a_b = builder.set_reg(reg_a, b_val)
        builder.add_instruction(set_a_b)

        # b = temp
        temp_val = builder.reg(reg_temp)
        set_b_temp = builder.set_reg(reg_b, temp_val)
        builder.add_instruction(set_b_temp)

        # i = i + 1
        inc_i = builder.add(i_val, const1)
        set_i_inc = builder.set_reg(reg_i, inc_i)
        builder.add_instruction(set_i_inc)

        # goto loop_condition
        goto_loop = builder.goto(InstructionIndex(3))
        builder.add_instruction(goto_loop)

        # Block 5: Return result
        return_block = LowLevelILBasicBlock(0x2050)
        function.add_basic_block(return_block)
        builder.set_current_block(return_block)

        # return b
        ret_b = builder.ret(b_val)
        builder.add_instruction(ret_b)

        return function
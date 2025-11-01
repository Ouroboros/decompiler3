"""
New Lifter System for BinaryNinja-style IR

This module provides lifting between the three IR levels:
LLIL -> MLIL -> HLIL
"""

from typing import List, Optional, Dict, Any
from .llil_new import (
    LowLevelILFunction, LowLevelILInstruction, LowLevelILBasicBlock,
    LowLevelILAdd, LowLevelILSub, LowLevelILMul, LowLevelILConst,
    LowLevelILReg, LowLevelILSetReg, LowLevelILLoad, LowLevelILStore,
    LowLevelILJump, LowLevelILGoto, LowLevelILIf, LowLevelILCall, LowLevelILRet
)
from .mlil_new import (
    MediumLevelILFunction, MediumLevelILInstruction, MediumLevelILBasicBlock,
    MediumLevelILBuilder, MediumLevelILAdd, MediumLevelILSub, MediumLevelILMul,
    MediumLevelILConst, MediumLevelILVar, MediumLevelILSetVar,
    MediumLevelILJump, MediumLevelILGoto, MediumLevelILIf, MediumLevelILCall, MediumLevelILRet,
    Variable
)
from .hlil_new import (
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


class NewDecompilerPipeline:
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
        from .llil_new import LowLevelILBuilder

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
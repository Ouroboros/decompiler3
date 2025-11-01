"""
Main compilation pipeline

Orchestrates the complete compilation process:
TypeScript → HLIL → MLIL → (legalize) → LLIL → instruction selection → bytecode
"""

from typing import List, Dict, Optional, Any, Union
from ..ir.base import IRFunction, IRContext, Architecture
from ..typescript.parser import TypeScriptParser
from ..target.legalization import legalize_for_target
from ..target.instruction_selection import select_instructions_for_target, MachineInstruction
from ..target.capability import get_target_capability


class CompilerPipeline:
    """Main compilation pipeline"""

    def __init__(self, target: str = "x86"):
        self.target = target
        self.target_capability = get_target_capability(target)

        if not self.target_capability:
            raise ValueError(f"Unsupported target: {target}")

        # Initialize components
        self.context = IRContext(Architecture(target))
        self.typescript_parser = TypeScriptParser()

    def compile_function(self, typescript_code: str, function_name: Optional[str] = None) -> bytes:
        """Compile TypeScript function to bytecode"""
        # Stage 1: TypeScript → HLIL
        hlil_function = self._parse_to_hlil(typescript_code, function_name)

        # Stage 2: HLIL → MLIL
        mlil_function = self._transform_to_mlil(hlil_function)

        # Stage 3: MLIL → Legalized LLIL
        llil_function = self._transform_and_legalize_to_llil(mlil_function)

        # Stage 4: LLIL → Machine Instructions
        machine_instructions = self._select_instructions(llil_function)

        # Stage 5: Machine Instructions → Bytecode
        bytecode = self._assemble_bytecode(machine_instructions)

        return bytecode

    def compile_from_typescript(self, typescript_code: str, metadata: Optional[Dict] = None) -> bytes:
        """Main compilation entry point"""
        if not typescript_code.strip():
            return b'\x10'  # Simple return instruction

        try:
            function_name = metadata.get("name") if metadata else None
            return self.compile_function(typescript_code, function_name)
        except Exception as e:
            # Return error bytecode
            return b'\xFF' + str(e).encode('utf-8')[:255]

    def _parse_to_hlil(self, typescript_code: str, function_name: Optional[str]) -> IRFunction:
        """Parse TypeScript to HLIL"""
        return self.typescript_parser.parse_function(typescript_code, function_name)

    def _transform_to_mlil(self, hlil_function: IRFunction) -> IRFunction:
        """Transform HLIL to MLIL"""
        from ..ir.mlil import MLILBuilder, MLILConstant, MLILReturn, MLILVariable, MLILBinaryOp
        from ..ir.hlil import HLILConstant, HLILReturn, HLILVariable, HLILBinaryOp
        from ..ir.base import IRBasicBlock

        mlil_function = IRFunction(hlil_function.name, hlil_function.address)
        mlil_function.parameters = hlil_function.parameters.copy()
        mlil_function.return_type = self._map_from_typescript_type(hlil_function.return_type)
        mlil_function.variables = hlil_function.variables.copy()

        # Convert basic blocks
        for i, block in enumerate(hlil_function.basic_blocks):
            mlil_block = IRBasicBlock(block.address)
            mlil_function.basic_blocks.append(mlil_block)

            builder = MLILBuilder(mlil_function)
            builder.set_current_block(mlil_block)

            # Convert HLIL instructions to MLIL
            for instruction in block.instructions:
                mlil_instruction = self._convert_hlil_to_mlil(instruction, builder, mlil_function)
                if mlil_instruction:
                    builder.add_instruction(mlil_instruction)

        return mlil_function

    def _transform_and_legalize_to_llil(self, mlil_function: IRFunction) -> IRFunction:
        """Transform MLIL to LLIL and legalize for target"""
        # First transform to basic LLIL
        llil_function = self._transform_to_llil(mlil_function)

        # Then legalize for target
        legalized_function = legalize_for_target(llil_function, self.target)

        return legalized_function

    def _transform_to_llil(self, mlil_function: IRFunction) -> IRFunction:
        """Transform MLIL to LLIL"""
        from ..ir.llil import LLILBuilder, LLILConstant, LLILReturn, LLILRegister, LLILBinaryOp
        from ..ir.mlil import MLILConstant, MLILReturn, MLILVariable, MLILBinaryOp
        from ..ir.base import IRBasicBlock

        llil_function = IRFunction(mlil_function.name, mlil_function.address)
        llil_function.parameters = mlil_function.parameters.copy()
        llil_function.return_type = mlil_function.return_type

        # Convert basic blocks
        for i, block in enumerate(mlil_function.basic_blocks):
            llil_block = IRBasicBlock(block.address)
            llil_function.basic_blocks.append(llil_block)

            builder = LLILBuilder(llil_function)
            builder.set_current_block(llil_block)

            # Convert MLIL instructions to LLIL
            for instruction in block.instructions:
                llil_instruction = self._convert_mlil_to_llil(instruction, builder, llil_function)
                if llil_instruction:
                    if isinstance(llil_instruction, list):
                        for instr in llil_instruction:
                            builder.add_instruction(instr)
                    else:
                        builder.add_instruction(llil_instruction)

        return llil_function

    def _select_instructions(self, llil_function: IRFunction) -> List[MachineInstruction]:
        """Select target instructions for LLIL"""
        return select_instructions_for_target(llil_function, self.target)

    def _assemble_bytecode(self, instructions: List[MachineInstruction]) -> bytes:
        """Assemble machine instructions to bytecode"""
        if self.target == "falcom_vm":
            return self._assemble_falcom_bytecode(instructions)
        elif self.target == "x86":
            return self._assemble_x86_bytecode(instructions)
        else:
            return self._assemble_generic_bytecode(instructions)

    def _assemble_falcom_bytecode(self, instructions: List[MachineInstruction]) -> bytes:
        """Assemble Falcom VM bytecode"""
        bytecode = bytearray()

        # Falcom VM opcode mapping
        opcode_map = {
            "CONST": 0x01,
            "ADD": 0x02,
            "SUB": 0x03,
            "MUL": 0x04,
            "DIV": 0x05,
            "MOD": 0x06,
            "AND": 0x07,
            "OR": 0x08,
            "XOR": 0x09,
            "EQ": 0x0A,
            "NE": 0x0B,
            "LT": 0x0C,
            "LE": 0x0D,
            "LOAD": 0x0E,
            "STORE": 0x0F,
            "CALL": 0x11,
            "FUNC_ENTRY": 0x12,
            "FUNC_EXIT": 0x13,
            "LOAD_LOCAL": 0x20,
            "STORE_LOCAL": 0x21,
        }

        for instruction in instructions:
            if instruction.opcode.endswith(":"):
                # Skip labels
                continue

            opcode = opcode_map.get(instruction.opcode, 0xFF)
            bytecode.append(opcode)

            # Encode operands
            for operand in instruction.operands:
                if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
                    # Integer operand
                    value = int(operand)
                    if -128 <= value <= 127:
                        bytecode.append(value & 0xFF)
                    else:
                        # Multi-byte integer
                        bytecode.extend(value.to_bytes(4, 'little', signed=True))
                else:
                    # String operand (function name, etc.)
                    string_bytes = operand.encode('utf-8')
                    bytecode.append(len(string_bytes))
                    bytecode.extend(string_bytes)

        # Add terminating instruction
        if not bytecode or bytecode[-1] != 0x13:  # FUNC_EXIT
            bytecode.append(0x13)

        return bytes(bytecode)

    def _assemble_x86_bytecode(self, instructions: List[MachineInstruction]) -> bytes:
        """Assemble x86 bytecode (simplified)"""
        # This is a very simplified x86 assembler
        # Real implementation would need a full x86 encoder
        bytecode = bytearray()

        for instruction in instructions:
            if instruction.opcode.endswith(":"):
                continue

            # Simple x86 instruction encoding
            if instruction.opcode == "mov":
                bytecode.extend([0x89, 0xC0])  # mov eax, eax (placeholder)
            elif instruction.opcode == "add":
                bytecode.extend([0x01, 0xC0])  # add eax, eax
            elif instruction.opcode == "sub":
                bytecode.extend([0x29, 0xC0])  # sub eax, eax
            elif instruction.opcode == "call":
                bytecode.extend([0xE8, 0x00, 0x00, 0x00, 0x00])  # call relative
            elif instruction.opcode == "ret":
                bytecode.append(0xC3)  # ret
            else:
                bytecode.extend([0x90])  # nop for unknown instructions

        return bytes(bytecode)

    def _assemble_generic_bytecode(self, instructions: List[MachineInstruction]) -> bytes:
        """Assemble generic bytecode"""
        # Simple generic bytecode format
        bytecode = bytearray()

        for instruction in instructions:
            if instruction.opcode.endswith(":"):
                continue

            # Encode instruction as: opcode_hash(1 byte) + operand_count(1 byte) + operands
            opcode_hash = hash(instruction.opcode) & 0xFF
            bytecode.append(opcode_hash)
            bytecode.append(len(instruction.operands))

            for operand in instruction.operands:
                operand_bytes = operand.encode('utf-8')
                bytecode.append(len(operand_bytes))
                bytecode.extend(operand_bytes)

        return bytes(bytecode)

    def _convert_hlil_to_mlil(self, hlil_instr, builder, function):
        """Convert HLIL instruction to MLIL"""
        from ..ir.hlil import HLILConstant, HLILReturn, HLILVariable, HLILBinaryOp, HLILCall, HLILBuiltinCall
        from ..ir.mlil import MLILConstant, MLILReturn, MLILVariable, MLILBinaryOp, MLILCall, MLILBuiltinCall

        if isinstance(hlil_instr, HLILConstant):
            return MLILConstant(hlil_instr.value, hlil_instr.size)

        elif isinstance(hlil_instr, HLILReturn):
            if hlil_instr.value:
                value = self._convert_hlil_to_mlil(hlil_instr.value, builder, function)
                return MLILReturn(value)
            return MLILReturn()

        elif isinstance(hlil_instr, HLILVariable):
            return MLILVariable(hlil_instr.variable)

        elif isinstance(hlil_instr, HLILBinaryOp):
            left = self._convert_hlil_to_mlil(hlil_instr.left, builder, function)
            right = self._convert_hlil_to_mlil(hlil_instr.right, builder, function)
            return MLILBinaryOp(hlil_instr.operation, left, right, hlil_instr.size, hlil_instr.expr_type)

        elif isinstance(hlil_instr, HLILCall):
            target = self._convert_hlil_to_mlil(hlil_instr.target, builder, function)
            arguments = [self._convert_hlil_to_mlil(arg, builder, function) for arg in hlil_instr.arguments]
            return MLILCall(target, arguments, hlil_instr.size, hlil_instr.expr_type)

        elif isinstance(hlil_instr, HLILBuiltinCall):
            arguments = [self._convert_hlil_to_mlil(arg, builder, function) for arg in hlil_instr.arguments]
            return MLILBuiltinCall(hlil_instr.builtin_name, arguments, hlil_instr.size, hlil_instr.expr_type)

        return None

    def _convert_mlil_to_llil(self, mlil_instr, builder, function):
        """Convert MLIL instruction to LLIL"""
        from ..ir.mlil import MLILConstant, MLILReturn, MLILVariable, MLILBinaryOp, MLILCall, MLILAssignment
        from ..ir.llil import LLILConstant, LLILReturn, LLILRegister, LLILBinaryOp, LLILCall, LLILStore

        if isinstance(mlil_instr, MLILConstant):
            return LLILConstant(mlil_instr.value, mlil_instr.size)

        elif isinstance(mlil_instr, MLILReturn):
            if mlil_instr.value:
                value = self._convert_mlil_to_llil(mlil_instr.value, builder, function)
                return LLILReturn(value)
            return LLILReturn()

        elif isinstance(mlil_instr, MLILVariable):
            # Convert variable to register or stack slot
            if self.target_capability.is_stack_machine:
                from ..ir.llil import LLILStack
                offset = self._get_variable_offset(mlil_instr.variable)
                return LLILStack(offset, mlil_instr.size)
            else:
                register = self._get_variable_register(mlil_instr.variable)
                return LLILRegister(register, mlil_instr.size)

        elif isinstance(mlil_instr, MLILBinaryOp):
            left = self._convert_mlil_to_llil(mlil_instr.left, builder, function)
            right = self._convert_mlil_to_llil(mlil_instr.right, builder, function)
            return LLILBinaryOp(mlil_instr.operation, left, right, mlil_instr.size)

        elif isinstance(mlil_instr, MLILCall):
            target = self._convert_mlil_to_llil(mlil_instr.target, builder, function)
            arguments = [self._convert_mlil_to_llil(arg, builder, function) for arg in mlil_instr.arguments]
            return LLILCall(target, arguments, mlil_instr.size)

        elif isinstance(mlil_instr, MLILAssignment):
            dest = self._convert_mlil_to_llil(mlil_instr.dest, builder, function)
            source = self._convert_mlil_to_llil(mlil_instr.source, builder, function)
            return LLILStore(dest, source, mlil_instr.size)

        return None

    def _map_from_typescript_type(self, ts_type: Optional[str]) -> Optional[str]:
        """Map TypeScript type to IR type"""
        if not ts_type:
            return None

        type_mapping = {
            "number": "int32",
            "string": "string",
            "boolean": "boolean",
            "void": "void",
            "any": "pointer",
            "object": "pointer"
        }

        return type_mapping.get(ts_type, "pointer")

    def _get_variable_offset(self, variable) -> int:
        """Get stack offset for variable (placeholder)"""
        return hash(variable.name) % 256

    def _get_variable_register(self, variable) -> str:
        """Get register for variable (placeholder)"""
        registers = ["eax", "ebx", "ecx", "edx"]
        return registers[hash(variable.name) % len(registers)]


# Convenience functions
def compile_typescript_to_bytecode(typescript_code: str, target: str = "x86") -> bytes:
    """Convenience function to compile TypeScript to bytecode"""
    pipeline = CompilerPipeline(target)
    return pipeline.compile_from_typescript(typescript_code)
"""
Main decompilation pipeline

Orchestrates the complete decompilation process:
bytecode → LLIL → MLIL → HLIL → TypeScript
"""

from typing import List, Dict, Optional, Any, Union
from ..ir.base import IRFunction, IRContext, Architecture
# 注意：这些Function类型目前不存在，使用基础的IRFunction
# from ..ir.llil import LLILFunction  # 不存在
# from ..ir.mlil import MLILFunction  # 不存在
# from ..ir.hlil import HLILFunction  # 不存在
from ..typescript.generator import TypeScriptGenerator, PrettyTypeScriptGenerator
from ..target.capability import get_target_capability


class DecompilerPipeline:
    """Main decompilation pipeline"""

    def __init__(self, architecture: str = "x86", output_style: str = "pretty"):
        self.architecture = architecture
        self.output_style = output_style
        self.target_capability = get_target_capability(architecture)

        if not self.target_capability:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Initialize components
        self.context = IRContext(Architecture(architecture))
        self.typescript_generator = self._create_typescript_generator()

        # Pipeline stages
        self.llil_lifter = None      # Would lift from bytecode to LLIL
        self.mlil_transformer = None # Would transform LLIL to MLIL
        self.hlil_transformer = None # Would transform MLIL to HLIL

    def decompile_function(self, bytecode: bytes, address: Optional[int] = None,
                          function_name: Optional[str] = None) -> str:
        """Decompile bytecode function to TypeScript"""
        # Stage 1: Bytecode → LLIL
        llil_function = self._lift_to_llil(bytecode, address, function_name)

        # Stage 2: LLIL → MLIL
        mlil_function = self._transform_to_mlil(llil_function)

        # Stage 3: MLIL → HLIL
        hlil_function = self._transform_to_hlil(mlil_function)

        # Stage 4: HLIL → TypeScript
        typescript_code = self.typescript_generator.generate_function(hlil_function)

        return typescript_code

    def decompile_to_typescript(self, bytecode: bytes, metadata: Optional[Dict] = None) -> str:
        """Main decompilation entry point"""
        if not bytecode:
            return "// Empty function\nfunction anonymous(): void {\n  return;\n}"

        try:
            return self.decompile_function(
                bytecode,
                metadata.get("address") if metadata else None,
                metadata.get("name") if metadata else None
            )
        except Exception as e:
            return f"// Decompilation failed: {str(e)}\nfunction error(): void {{\n  throw new Error('{str(e)}');\n}}"

    def _lift_to_llil(self, bytecode: bytes, address: Optional[int], function_name: Optional[str]) -> IRFunction:
        """Lift bytecode to LLIL (placeholder implementation)"""
        # This would normally parse the bytecode and generate LLIL
        # For now, create a simple placeholder function

        from ..ir.llil import LLILBuilder, LLILConstant, LLILReturn
        from ..ir.base import IRBasicBlock

        function = IRFunction(function_name or "anonymous", address)
        block = IRBasicBlock(address)
        function.basic_blocks.append(block)

        builder = LLILBuilder(function)
        builder.set_current_block(block)

        # Simple placeholder - real implementation would decode bytecode
        if self.target_capability.name == "falcom_vm":
            # Simulate Falcom VM bytecode lifting
            if len(bytecode) >= 4:
                # Interpret first 4 bytes as operation
                op_code = bytecode[0]
                if op_code == 0x01:  # CONST
                    value = int.from_bytes(bytecode[1:4], 'little')
                    const_expr = builder.const(value)
                    builder.add_instruction(const_expr)
                elif op_code == 0x10:  # RETURN
                    ret_expr = builder.ret()
                    builder.add_instruction(ret_expr)
                else:
                    # Unknown opcode
                    builder.add_instruction(builder.const(0))
                    builder.add_instruction(builder.ret())
            else:
                # Empty or invalid bytecode
                builder.add_instruction(builder.ret())
        else:
            # Generic bytecode handling
            builder.add_instruction(builder.const(42))  # Placeholder
            builder.add_instruction(builder.ret(builder.const(0)))

        return function

    def _transform_to_mlil(self, llil_function: IRFunction) -> IRFunction:
        """Transform LLIL to MLIL"""
        # This would normally perform stack elimination, variable recovery, etc.
        # For now, create a basic MLIL version

        from ..ir.mlil import MLILBuilder, MLILConstant, MLILReturn
        from ..ir.base import IRBasicBlock, IRVariable

        mlil_function = IRFunction(llil_function.name, llil_function.address)
        mlil_function.parameters = llil_function.parameters.copy()
        mlil_function.return_type = llil_function.return_type

        # Create variables for stack locations
        for i, block in enumerate(llil_function.basic_blocks):
            mlil_block = IRBasicBlock(block.address)
            mlil_function.basic_blocks.append(mlil_block)

            builder = MLILBuilder(mlil_function)
            builder.set_current_block(mlil_block)

            # Convert LLIL instructions to MLIL
            for instruction in block.instructions:
                mlil_instruction = self._convert_llil_to_mlil(instruction, builder, mlil_function)
                if mlil_instruction:
                    builder.add_instruction(mlil_instruction)

        return mlil_function

    def _transform_to_hlil(self, mlil_function: IRFunction) -> IRFunction:
        """Transform MLIL to HLIL using the sophisticated MLIL Lifter"""
        from ..ir.mlil_lifter import MLILLifter

        # Use the new MLIL Lifter for comprehensive transformation
        lifter = MLILLifter()
        hlil_function = lifter.lift_function(mlil_function)

        # Ensure return type is set
        if not hlil_function.return_type:
            hlil_function.return_type = self._map_to_typescript_type(mlil_function.return_type)

        # Add type information to variables if missing
        for var_name, variable in hlil_function.variables.items():
            if not variable.var_type:
                variable.var_type = "any"  # Default type

        return hlil_function

    def _convert_llil_to_mlil(self, llil_instr, builder, function):
        """Convert LLIL instruction to MLIL"""
        from ..ir.llil import LLILConstant, LLILReturn, LLILRegister, LLILBinaryOp, LLILJump, LLILIf, LLILStore, LLILLoad, LLILCall
        from ..ir.mlil import MLILConstant, MLILReturn, MLILVariable, MLILBinaryOp, MLILJump, MLILIf, MLILStore, MLILLoad, MLILCall

        if isinstance(llil_instr, LLILConstant):
            return MLILConstant(llil_instr.value, llil_instr.size)

        elif isinstance(llil_instr, LLILReturn):
            if llil_instr.value:
                value = self._convert_llil_to_mlil(llil_instr.value, builder, function)
                return MLILReturn(value)
            return MLILReturn()

        elif isinstance(llil_instr, LLILRegister):
            # Convert register to variable
            var_name = f"var_{llil_instr.register}"
            if var_name not in function.variables:
                variable = function.create_variable(var_name, llil_instr.size)
            else:
                variable = function.variables[var_name]
            return MLILVariable(variable)

        elif isinstance(llil_instr, LLILBinaryOp):
            left = self._convert_llil_to_mlil(llil_instr.left, builder, function)
            right = self._convert_llil_to_mlil(llil_instr.right, builder, function)
            return MLILBinaryOp(llil_instr.operation, left, right, llil_instr.size)

        elif isinstance(llil_instr, LLILJump):
            # Convert unconditional jump
            return MLILJump(llil_instr.target)

        elif isinstance(llil_instr, LLILIf):
            # Convert conditional branch
            condition = self._convert_llil_to_mlil(llil_instr.condition, builder, function)
            return MLILIf(condition, llil_instr.true_target, llil_instr.false_target)

        elif isinstance(llil_instr, LLILStore):
            # Convert store operation
            from ..ir.llil import LLILStack
            if isinstance(llil_instr.address, LLILStack):
                # Stack variable assignment
                var_name = f"stack_var_{abs(llil_instr.address.offset)}"
                if var_name not in function.variables:
                    variable = function.create_variable(var_name, llil_instr.size)
                else:
                    variable = function.variables[var_name]
                value = self._convert_llil_to_mlil(llil_instr.value, builder, function)
                return MLILStore(MLILVariable(variable), value, llil_instr.size)
            else:
                # Memory store
                addr = self._convert_llil_to_mlil(llil_instr.address, builder, function)
                value = self._convert_llil_to_mlil(llil_instr.value, builder, function)
                return MLILStore(addr, value, llil_instr.size)

        elif isinstance(llil_instr, LLILLoad):
            # Convert load operation
            from ..ir.llil import LLILStack
            if isinstance(llil_instr.address, LLILStack):
                # Stack variable load
                var_name = f"stack_var_{abs(llil_instr.address.offset)}"
                if var_name not in function.variables:
                    variable = function.create_variable(var_name, llil_instr.size)
                else:
                    variable = function.variables[var_name]
                return MLILLoad(MLILVariable(variable), llil_instr.size)
            else:
                # Memory load
                addr = self._convert_llil_to_mlil(llil_instr.address, builder, function)
                return MLILLoad(addr, llil_instr.size)

        elif isinstance(llil_instr, LLILCall):
            # Convert function call
            target = self._convert_llil_to_mlil(llil_instr.target, builder, function)
            arguments = [self._convert_llil_to_mlil(arg, builder, function) for arg in llil_instr.arguments]
            return MLILCall(target, arguments, llil_instr.size)

        return None

    def _convert_mlil_to_hlil(self, mlil_instr, builder, function):
        """Convert MLIL instruction to HLIL"""
        from ..ir.mlil import MLILConstant, MLILReturn, MLILVariable, MLILBinaryOp, MLILCall, MLILBuiltinCall
        from ..ir.hlil import HLILConstant, HLILReturn, HLILVariable, HLILBinaryOp, HLILCall, HLILBuiltinCall

        if isinstance(mlil_instr, MLILConstant):
            const_type = self._infer_constant_type(mlil_instr.value)
            return HLILConstant(mlil_instr.value, mlil_instr.size, const_type)

        elif isinstance(mlil_instr, MLILReturn):
            if mlil_instr.value:
                value = self._convert_mlil_to_hlil(mlil_instr.value, builder, function)
                return HLILReturn(value)
            return HLILReturn()

        elif isinstance(mlil_instr, MLILVariable):
            var_type = mlil_instr.variable.var_type or "any"
            return HLILVariable(mlil_instr.variable, var_type)

        elif isinstance(mlil_instr, MLILBinaryOp):
            left = self._convert_mlil_to_hlil(mlil_instr.left, builder, function)
            right = self._convert_mlil_to_hlil(mlil_instr.right, builder, function)
            result_type = self._infer_binary_op_type(mlil_instr.operation, left, right)
            return HLILBinaryOp(mlil_instr.operation, left, right, mlil_instr.size, result_type)

        elif isinstance(mlil_instr, MLILCall):
            target = self._convert_mlil_to_hlil(mlil_instr.target, builder, function)
            arguments = [self._convert_mlil_to_hlil(arg, builder, function) for arg in mlil_instr.arguments]
            return HLILCall(target, arguments, mlil_instr.size, mlil_instr.return_type)

        elif isinstance(mlil_instr, MLILBuiltinCall):
            arguments = [self._convert_mlil_to_hlil(arg, builder, function) for arg in mlil_instr.arguments]
            return HLILBuiltinCall(mlil_instr.builtin_name, arguments, mlil_instr.size, mlil_instr.return_type)

        return None

    def _create_typescript_generator(self) -> TypeScriptGenerator:
        """Create appropriate TypeScript generator"""
        if self.output_style == "pretty":
            return PrettyTypeScriptGenerator()
        else:
            from ..typescript.generator import RoundTripTypeScriptGenerator
            return RoundTripTypeScriptGenerator()

    def _map_to_typescript_type(self, ir_type: Optional[str]) -> Optional[str]:
        """Map IR type to TypeScript type"""
        if not ir_type:
            return None

        type_mapping = {
            "int8": "number",
            "int16": "number",
            "int32": "number",
            "int64": "number",
            "float32": "number",
            "float64": "number",
            "pointer": "any",
            "string": "string",
            "boolean": "boolean",
            "void": "void"
        }

        return type_mapping.get(ir_type, "any")

    def _infer_constant_type(self, value: Any) -> str:
        """Infer TypeScript type for a constant value"""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, str):
            return "string"
        else:
            return "any"

    def _infer_binary_op_type(self, operation, left, right) -> str:
        """Infer result type for binary operation"""
        from ..ir.base import OperationType

        # Comparison operations return boolean
        if operation in [OperationType.CMP_E, OperationType.CMP_NE,
                        OperationType.CMP_SLT, OperationType.CMP_SLE,
                        OperationType.CMP_ULT, OperationType.CMP_ULE]:
            return "boolean"

        # Arithmetic operations typically return number
        if operation in [OperationType.ADD, OperationType.SUB,
                        OperationType.MUL, OperationType.DIV, OperationType.MOD]:
            return "number"

        # Bitwise operations return number
        if operation in [OperationType.AND, OperationType.OR, OperationType.XOR,
                        OperationType.LSL, OperationType.LSR, OperationType.ASR]:
            return "number"

        return "any"


# Convenience functions
def decompile_bytecode_to_typescript(bytecode: bytes, architecture: str = "x86",
                                   output_style: str = "pretty") -> str:
    """Convenience function to decompile bytecode to TypeScript"""
    pipeline = DecompilerPipeline(architecture, output_style)
    return pipeline.decompile_to_typescript(bytecode)
"""
Legalization pass

Transforms MLIL to target-legal LLIL by breaking down operations
that cannot be directly supported by the target architecture.
"""

from typing import List, Dict, Optional, Any
from ..ir.base import IRFunction, IRBasicBlock, IRExpression, IRTransformer, OperationType
from ..ir.mlil import *
from ..ir.llil import *
from .capability import TargetCapability, DataType, AddressingMode, get_target_capability


class LegalizationPass(IRTransformer):
    """Legalizes IR for a specific target architecture"""

    def __init__(self, target_capability: TargetCapability):
        self.target = target_capability
        self.temp_counter = 0

    def transform_function(self, function: IRFunction) -> IRFunction:
        """Legalize entire function"""
        new_function = IRFunction(function.name, function.address)
        new_function.parameters = function.parameters.copy()
        new_function.return_type = function.return_type
        new_function.ssa_form = function.ssa_form

        # Copy and transform basic blocks
        for block in function.basic_blocks:
            new_block = self._transform_basic_block(block)
            new_function.basic_blocks.append(new_block)

        # Copy variables
        new_function.variables = function.variables.copy()

        return new_function

    def transform_expression(self, expr: IRExpression) -> IRExpression:
        """Transform individual expression"""
        if isinstance(expr, MLILBinaryOp):
            return self._legalize_binary_op(expr)
        elif isinstance(expr, MLILUnaryOp):
            return self._legalize_unary_op(expr)
        elif isinstance(expr, MLILCall):
            return self._legalize_call(expr)
        elif isinstance(expr, MLILLoad):
            return self._legalize_load(expr)
        elif isinstance(expr, MLILStore):
            return self._legalize_store(expr)
        elif isinstance(expr, MLILBuiltinCall):
            return self._legalize_builtin_call(expr)
        else:
            # Default conversion from MLIL to LLIL
            return self._convert_to_llil(expr)

    def _transform_basic_block(self, block: IRBasicBlock) -> IRBasicBlock:
        """Transform a basic block"""
        new_block = IRBasicBlock(block.address)
        new_block.id = block.id

        for instruction in block.instructions:
            legalized = self.transform_expression(instruction)
            if isinstance(legalized, list):
                new_block.instructions.extend(legalized)
            else:
                new_block.instructions.append(legalized)

        return new_block

    def _legalize_binary_op(self, expr: MLILBinaryOp) -> IRExpression:
        """Legalize binary operations"""
        # Check if target supports this operation
        if not self.target.supports_operation(expr.operation):
            return self._expand_unsupported_operation(expr)

        # Check if operands are legal
        left = self.transform_expression(expr.left)
        right = self.transform_expression(expr.right)

        # Handle immediate operands that are too large
        if isinstance(right, MLILConstant):
            if not self.target.can_represent_immediate(right.value):
                # Load large immediate into temporary
                temp_var = self._create_temp_variable(right.size)
                return LLILBinaryOp(expr.operation, left, LLILRegister(temp_var, right.size))

        # Convert to LLIL
        return LLILBinaryOp(expr.operation, left, right)

    def _legalize_unary_op(self, expr: MLILUnaryOp) -> IRExpression:
        """Legalize unary operations"""
        if not self.target.supports_operation(expr.operation):
            return self._expand_unary_operation(expr)

        operand = self.transform_expression(expr.operand)
        return LLILUnaryOp(expr.operation, operand)

    def _legalize_call(self, expr: MLILCall) -> IRExpression:
        """Legalize function calls"""
        # Transform target and arguments
        target = self.transform_expression(expr.target)
        arguments = [self.transform_expression(arg) for arg in expr.arguments]

        # For stack machines, we need to push arguments
        if self.target.is_stack_machine:
            instructions = []
            # Push arguments in reverse order (right to left)
            for arg in reversed(arguments):
                instructions.append(LLILPush(arg))
            instructions.append(LLILCall(target, []))
            return instructions

        return LLILCall(target, arguments)

    def _legalize_load(self, expr: MLILLoad) -> IRExpression:
        """Legalize memory loads"""
        address = self.transform_expression(expr.address)

        # Check if addressing mode is supported
        if not self._is_addressing_mode_legal(address, AddressingMode.MEMORY):
            # Convert complex address to register
            temp_reg = self._create_temp_register()
            return [
                LLILStore(LLILRegister(temp_reg), address),
                LLILLoad(LLILRegister(temp_reg))
            ]

        return LLILLoad(address)

    def _legalize_store(self, expr: MLILStore) -> IRExpression:
        """Legalize memory stores"""
        address = self.transform_expression(expr.address)
        value = self.transform_expression(expr.value)

        if not self._is_addressing_mode_legal(address, AddressingMode.MEMORY):
            temp_reg = self._create_temp_register()
            return [
                LLILStore(LLILRegister(temp_reg), address),
                LLILStore(LLILLoad(LLILRegister(temp_reg)), value)
            ]

        return LLILStore(address, value)

    def _legalize_builtin_call(self, expr: MLILBuiltinCall) -> IRExpression:
        """Legalize built-in function calls"""
        from ..builtin.registry import get_builtin
        from ..builtin.semantics import BuiltinExpander

        builtin = get_builtin(expr.builtin_name)
        if not builtin:
            # Unknown built-in - convert to regular call
            target = LLILConstant(f"__builtin_{expr.builtin_name}")
            arguments = [self.transform_expression(arg) for arg in expr.arguments]
            return LLILCall(target, arguments)

        # Try to get target-specific mapping
        mapping = builtin.get_mapping(self.target.name)
        if not mapping:
            mapping = builtin.get_mapping("generic")

        if mapping and mapping.direct_opcode:
            # Direct mapping - keep as builtin call for now
            # Instruction selection will handle the actual opcode mapping
            arguments = [self.transform_expression(arg) for arg in expr.arguments]
            return expr  # Keep as-is for instruction selection

        elif mapping and mapping.expansion:
            # Expand to sequence of operations
            expander = BuiltinExpander()
            expanded = expander.expand_for_target(expr, self.target.name)
            if expanded:
                return [self.transform_expression(inst) for inst in expanded]

        elif mapping and mapping.library_call:
            # Convert to library call
            target = LLILConstant(mapping.library_call)
            arguments = [self.transform_expression(arg) for arg in expr.arguments]
            return LLILCall(target, arguments)

        # Fallback - convert to generic call
        target = LLILConstant(f"__builtin_{expr.builtin_name}")
        arguments = [self.transform_expression(arg) for arg in expr.arguments]
        return LLILCall(target, arguments)

    def _expand_unsupported_operation(self, expr: MLILBinaryOp) -> IRExpression:
        """Expand operations not supported by target"""
        if expr.operation == OperationType.DIV and not self.target.has_division:
            # Convert division to library call
            target = LLILConstant("__div")
            left = self.transform_expression(expr.left)
            right = self.transform_expression(expr.right)
            return LLILCall(target, [left, right])

        elif expr.operation == OperationType.MOD and not self.target.has_modulo:
            # Convert modulo to library call
            target = LLILConstant("__mod")
            left = self.transform_expression(expr.left)
            right = self.transform_expression(expr.right)
            return LLILCall(target, [left, right])

        elif expr.operation in [OperationType.MUL] and self.target.is_stack_machine:
            # For stack machines, binary ops work on stack
            left = self.transform_expression(expr.left)
            right = self.transform_expression(expr.right)
            return [
                LLILPush(left),
                LLILPush(right),
                LLILBinaryOp(expr.operation, LLILPop(), LLILPop())
            ]

        # Default: try to convert directly
        left = self.transform_expression(expr.left)
        right = self.transform_expression(expr.right)
        return LLILBinaryOp(expr.operation, left, right)

    def _expand_unary_operation(self, expr: MLILUnaryOp) -> IRExpression:
        """Expand unsupported unary operations"""
        if expr.operation == OperationType.NEG:
            # Convert -x to (0 - x)
            operand = self.transform_expression(expr.operand)
            zero = LLILConstant(0)
            return LLILBinaryOp(OperationType.SUB, zero, operand)

        # Default conversion
        operand = self.transform_expression(expr.operand)
        return LLILUnaryOp(expr.operation, operand)

    def _convert_to_llil(self, expr: IRExpression) -> IRExpression:
        """Convert MLIL expression to equivalent LLIL"""
        if isinstance(expr, MLILVariable):
            # Convert variable to register or stack reference
            if self.target.is_stack_machine:
                # Use stack offset for variables
                offset = self._get_variable_stack_offset(expr.variable)
                return LLILStack(offset, expr.size)
            else:
                # Use register
                reg_name = self._get_variable_register(expr.variable)
                return LLILRegister(reg_name, expr.size)

        elif isinstance(expr, MLILConstant):
            return LLILConstant(expr.value, expr.size)

        elif isinstance(expr, MLILAssignment):
            dest = self._convert_to_llil(expr.dest)
            source = self.transform_expression(expr.source)

            if isinstance(dest, LLILStack):
                # Stack assignment
                return LLILStore(LLILStack(dest.offset), source)
            elif isinstance(dest, LLILRegister):
                # Register assignment (represented as store for consistency)
                return LLILStore(dest, source)

        elif isinstance(expr, MLILFieldAccess):
            # Convert field access to memory load with offset
            base = self.transform_expression(expr.base)
            # This is simplified - real implementation would need structure layout info
            field_offset = LLILConstant(0)  # Placeholder
            address = LLILBinaryOp(OperationType.ADD, base, field_offset)
            return LLILLoad(address, expr.size)

        # Default: return as-is
        return expr

    def _is_addressing_mode_legal(self, address_expr: IRExpression, mode: AddressingMode) -> bool:
        """Check if addressing mode is legal for target"""
        return mode in self.target.addressing_modes

    def _create_temp_variable(self, size: int) -> str:
        """Create a temporary variable name"""
        name = f"__temp_{self.temp_counter}"
        self.temp_counter += 1
        return name

    def _create_temp_register(self) -> str:
        """Create a temporary register name"""
        return self._create_temp_variable(4)

    def _get_variable_stack_offset(self, variable: IRVariable) -> int:
        """Get stack offset for a variable (placeholder)"""
        # This would need proper stack frame analysis
        return hash(variable.name) % 256

    def _get_variable_register(self, variable: IRVariable) -> str:
        """Get register assignment for a variable (placeholder)"""
        # This would be handled by register allocation
        registers = ["eax", "ebx", "ecx", "edx", "esi", "edi"]
        return registers[hash(variable.name) % len(registers)]


class TypeLegalizer:
    """Legalizes data types for target architecture"""

    def __init__(self, target_capability: TargetCapability):
        self.target = target_capability

    def legalize_type(self, data_type: str, size: int) -> tuple[str, int]:
        """Legalize a data type for the target"""
        # Convert high-level types to target types
        type_mapping = {
            "number": DataType.INT32 if size <= 4 else DataType.INT64,
            "float": DataType.FLOAT32 if size <= 4 else DataType.FLOAT64,
            "string": DataType.POINTER,
            "boolean": DataType.INT32,  # Represent as integer
            "pointer": DataType.POINTER,
            "any": DataType.POINTER,  # Use pointer for dynamic types
        }

        target_type = type_mapping.get(data_type, DataType.INT32)

        # Check if target supports this type
        supported_types = []
        for capability in self.target.supported_operations.values():
            supported_types.extend(capability.supported_types)

        if target_type not in supported_types:
            # Fall back to supported type
            if DataType.INT32 in supported_types:
                return "int32", 4
            elif DataType.POINTER in supported_types:
                return "pointer", self.target.pointer_size

        # Return legalized type
        type_names = {
            DataType.INT8: "int8",
            DataType.INT16: "int16",
            DataType.INT32: "int32",
            DataType.INT64: "int64",
            DataType.FLOAT32: "float32",
            DataType.FLOAT64: "float64",
            DataType.POINTER: "pointer",
            DataType.BOOLEAN: "boolean",
        }

        return type_names.get(target_type, "int32"), self._get_type_size(target_type)

    def _get_type_size(self, data_type: DataType) -> int:
        """Get size in bytes for a data type"""
        sizes = {
            DataType.INT8: 1,
            DataType.INT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.FLOAT32: 4,
            DataType.FLOAT64: 8,
            DataType.POINTER: self.target.pointer_size,
            DataType.BOOLEAN: 1,
        }
        return sizes.get(data_type, 4)


def legalize_for_target(function: IRFunction, target_name: str) -> IRFunction:
    """Convenience function to legalize a function for a target"""
    capability = get_target_capability(target_name)
    if not capability:
        raise ValueError(f"Unknown target: {target_name}")

    legalizer = LegalizationPass(capability)
    return legalizer.transform_function(function)
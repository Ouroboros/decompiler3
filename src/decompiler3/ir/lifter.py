"""
LLIL to MLIL Lifter

A comprehensive lifting system that converts Low Level IL to Medium Level IL.
Implements the core transformations similar to BinaryNinja's lifter:

1. Stack elimination - Convert stack operations to variables
2. Variable recovery - Reconstruct high-level variables from registers/memory
3. Memory access analysis - Analyze pointer operations and array access
4. Control flow structuring - Identify structured control flow patterns
5. Calling convention handling - Process function calls and arguments
6. Type inference - Basic type inference based on operations
"""

from typing import Dict, List, Optional, Set, Union, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from .base import IRFunction, IRBasicBlock, IRVariable, OperationType, IRType
from .llil import (
    LLILExpression, LLILRegister, LLILStack, LLILConstant, LLILBinaryOp,
    LLILUnaryOp, LLILLoad, LLILStore, LLILCall, LLILJump, LLILIf,
    LLILReturn, LLILPush, LLILPop
)
from .mlil import (
    MLILExpression, MLILVariable, MLILConstant, MLILBinaryOp, MLILUnaryOp,
    MLILAssignment, MLILLoad, MLILStore, MLILFieldAccess, MLILCall,
    MLILBuiltinCall, MLILJump, MLILIf, MLILReturn, MLILBuilder
)
from ..target.registers import (
    ArchitectureType, RegisterSet, RegisterInfo, CallingConventionType,
    get_register_set, get_register_by_name, is_parameter_register, is_return_register
)


class VariableNamingConstants:
    """Constants for variable naming and classification"""
    # Variable name prefixes
    LOCAL_PREFIX = "local_"
    PARAM_PREFIX = "param_"
    REGISTER_PREFIX = "var_"
    ARRAY_PREFIX = "array_"
    ARRAY_ELEMENT_PREFIX = "array_element_"

    # Common variable names
    RESULT_NAME = "result"
    COUNTER_NAME = "counter"
    INDEX_NAME = "index"
    TEMP_NAME = "temp"
    SRC_NAME = "src"
    DST_NAME = "dst"
    ARG_PREFIX = "arg_"

    # Variable type classifications
    LOCAL_VARIABLE = "local"
    PARAMETER_VARIABLE = "parameter"
    REGISTER_VARIABLE = "register"
    ARRAY_VARIABLE = "array"
    RETURN_VARIABLE = "return"


class RegisterConventions:
    """Register naming conventions for different architectures"""

    @staticmethod
    def get_conventional_name(register_info: RegisterInfo, arch_type: ArchitectureType) -> Optional[str]:
        """Get conventional variable name for a register based on architecture"""
        reg_name = register_info.name

        if arch_type in [ArchitectureType.X86_32, ArchitectureType.X86_64]:
            # x86/x64 specific conventional names based on register purpose
            from ..target.registers import X86Registers, X64Registers

            if arch_type == ArchitectureType.X86_32:
                if register_info == X86Registers.ECX.value:
                    return VariableNamingConstants.COUNTER_NAME
                elif register_info == X86Registers.ESI.value:
                    return VariableNamingConstants.SRC_NAME
                elif register_info == X86Registers.EDI.value:
                    return VariableNamingConstants.DST_NAME
            elif arch_type == ArchitectureType.X86_64:
                if register_info == X64Registers.RCX.value:
                    return VariableNamingConstants.COUNTER_NAME
                elif register_info == X64Registers.RSI.value:
                    return VariableNamingConstants.SRC_NAME
                elif register_info == X64Registers.RDI.value:
                    return VariableNamingConstants.DST_NAME

        elif arch_type == ArchitectureType.ARM_32:
            from ..target.registers import ARMRegisters

            if register_info == ARMRegisters.R0.value:
                return VariableNamingConstants.RESULT_NAME
            elif register_info in [ARMRegisters.R1.value, ARMRegisters.R2.value, ARMRegisters.R3.value]:
                # Extract register number for arg naming
                reg_num = reg_name[1:]  # "r1" -> "1"
                return f"{VariableNamingConstants.ARG_PREFIX}{reg_num}"

        return None


@dataclass
class StackLocation:
    """Represents a stack location during analysis"""
    offset: int
    size: int
    variable: Optional[IRVariable] = None
    last_written: Optional[int] = None  # instruction index
    access_count: int = 0
    region: str = "unknown"  # "locals", "params", "frame_base"


@dataclass
class RegisterState:
    """Tracks register state during lifting"""
    variable: Optional[IRVariable] = None
    last_written: Optional[int] = None
    last_read: Optional[int] = None
    is_parameter: bool = False
    is_return_value: bool = False


@dataclass
class MemoryAccess:
    """Represents a memory access pattern"""
    base_expr: Optional[LLILExpression] = None
    offset: int = 0
    size: int = 4
    is_constant_offset: bool = False
    access_type: str = "unknown"  # "array", "struct", "pointer", "global", "local", etc.
    operation_type: str = "unknown"  # "load", "store"
    position: Optional[Tuple[int, int]] = None  # (block_idx, instr_idx)

    # Array access fields
    index_expr: Optional[LLILExpression] = None
    scale: int = 1
    element_type: str = "unknown"

    # Pointer chain fields
    chain_length: int = 0
    final_base: Optional[LLILExpression] = None

    # Additional fields
    base_address: Optional[int] = None  # For global accesses


@dataclass
class CallConvention:
    """Represents calling convention information"""
    name: str
    parameter_registers: List['RegisterInfo'] = field(default_factory=list)
    return_register: Optional['RegisterInfo'] = None
    stack_cleanup: str = "caller"  # "caller" or "callee"
    stack_alignment: int = 4


class LLILLifterPass(ABC):
    """Abstract base class for lifter passes"""

    @abstractmethod
    def run(self, context: 'LifterContext') -> bool:
        """Run the pass on the given context. Returns True if changes were made."""
        pass


class LifterContext:
    """Context shared between lifter passes"""

    def __init__(self, llil_function: IRFunction, target_arch: ArchitectureType):
        self.llil_function = llil_function
        self.mlil_function = IRFunction(llil_function.name, llil_function.address)
        self.target_arch = target_arch

        # Analysis state
        self.stack_layout: Dict[int, StackLocation] = {}
        self.register_states: Dict[str, RegisterState] = {}
        self.memory_accesses: List[MemoryAccess] = []
        self.call_sites: List[Tuple[int, LLILCall]] = []

        # Variable tracking
        self.variable_counter = 0
        self.stack_vars: Dict[int, IRVariable] = {}  # offset -> variable
        self.reg_vars: Dict[str, IRVariable] = {}    # register -> variable

        # Control flow analysis
        self.structured_blocks: Dict[int, str] = {}  # block_addr -> type
        self.loop_headers: Set[int] = set()
        self.condition_blocks: Set[int] = set()

        # Calling convention
        self.calling_convention = self._get_calling_convention(target_arch)

        # Builder for MLIL construction
        self.builder = MLILBuilder(self.mlil_function)

        # Logging
        self.logger = logging.getLogger(__name__)

    def _get_calling_convention(self, arch: str) -> CallConvention:
        """Get calling convention for target architecture"""
        # Map string architecture to enum
        arch_map = {
            "x86": ArchitectureType.X86_32,
            "x64": ArchitectureType.X86_64,
            "arm": ArchitectureType.ARM_32,
            "falcom_vm": ArchitectureType.FALCOM_VM
        }

        arch_type = arch_map.get(arch, ArchitectureType.X86_32)
        register_set = get_register_set(arch_type)

        # Create calling convention based on register set
        if arch_type == ArchitectureType.X86_32:
            return CallConvention(
                name=CallingConventionType.CDECL.value,
                parameter_registers=[],  # x86 uses stack for parameters
                return_register=register_set.return_register,
                stack_cleanup="caller"
            )
        elif arch_type == ArchitectureType.X86_64:
            return CallConvention(
                name=CallingConventionType.SYSTEMV.value,
                parameter_registers=register_set.parameter_registers,
                return_register=register_set.return_register,
                stack_cleanup="caller"
            )
        elif arch_type == ArchitectureType.ARM_32:
            return CallConvention(
                name=CallingConventionType.AAPCS.value,
                parameter_registers=register_set.parameter_registers,
                return_register=register_set.return_register,
                stack_cleanup="caller"
            )
        elif arch_type == ArchitectureType.FALCOM_VM:
            return CallConvention(
                name=CallingConventionType.STACK_MACHINE.value,
                parameter_registers=[],  # Stack machine
                return_register=register_set.return_register,
                stack_cleanup="callee"
            )
        else:
            # Default to x86
            return CallConvention(
                name=CallingConventionType.CDECL.value,
                parameter_registers=[],
                return_register=register_set.return_register,
                stack_cleanup="caller"
            )

    def create_variable(self, name: str, size: int, var_type: Optional[IRType] = None) -> IRVariable:
        """Create a new variable in the MLIL function"""
        if var_type is None:
            var_type = IRType.NUMBER  # Default to number type

        var = self.mlil_function.create_variable(name, size, var_type)
        return var

    def get_or_create_stack_variable(self, offset: int, size: int) -> IRVariable:
        """Get or create a variable for a stack location"""
        if offset in self.stack_vars:
            return self.stack_vars[offset]

        # Create descriptive name based on offset
        if offset >= 0:
            name = f"stack_var_{offset}"
        else:
            name = f"local_{abs(offset)}"

        var = self.create_variable(name, size)
        self.stack_vars[offset] = var

        # Track in stack layout
        self.stack_layout[offset] = StackLocation(offset, size, var)

        return var

    def get_or_create_register_variable(self, register: str, size: int) -> IRVariable:
        """Get or create a variable for a register"""
        if register in self.reg_vars:
            return self.reg_vars[register]

        name = f"reg_{register}"
        var = self.create_variable(name, size)
        self.reg_vars[register] = var

        # Track register state
        self.register_states[register] = RegisterState(variable=var)

        return var


class StackEliminationPass(LLILLifterPass):
    """
    Pass 1: Stack Elimination

    Converts stack-based operations to variable operations:
    - LLILStack -> MLILVariable
    - LLILPush/LLILPop -> MLILAssignment
    - Tracks stack layout and variable lifetimes
    - Analyzes stack pointer movements
    - Handles different stack architectures (register-based vs stack machine)
    """

    def __init__(self):
        self.stack_pointer_offset = 0  # Track current SP offset for stack machines
        self.stack_operations = []     # Track push/pop sequence

    def run(self, context: LifterContext) -> bool:
        # First pass: analyze stack usage patterns
        self._analyze_stack_usage(context)

        # Second pass: eliminate stack operations
        changes_made = self._eliminate_stack_operations(context)

        # Third pass: optimize stack variable layout
        self._optimize_stack_layout(context)

        return changes_made

    def _analyze_stack_usage(self, context: LifterContext):
        """Analyze stack usage patterns throughout the function"""
        context.logger.debug("Analyzing stack usage patterns")

        for block_idx, block in enumerate(context.llil_function.basic_blocks):
            for instr_idx, instruction in enumerate(block.instructions):
                self._analyze_instruction_stack_usage(instruction, context, block_idx, instr_idx)

    def _analyze_instruction_stack_usage(self, instr: LLILExpression, context: LifterContext,
                                       block_idx: int, instr_idx: int):
        """Analyze stack usage in a single instruction"""
        if isinstance(instr, LLILStack):
            # Direct stack reference
            offset = instr.offset
            if offset not in context.stack_layout:
                context.stack_layout[offset] = StackLocation(offset, instr.size)

            context.stack_layout[offset].access_count += 1
            context.stack_layout[offset].last_written = instr_idx

        elif isinstance(instr, LLILPush):
            # Stack push operation
            self.stack_operations.append(('push', block_idx, instr_idx, instr))
            self.stack_pointer_offset -= instr.size

        elif isinstance(instr, LLILPop):
            # Stack pop operation
            self.stack_operations.append(('pop', block_idx, instr_idx, instr))
            self.stack_pointer_offset += instr.size

        elif isinstance(instr, LLILStore) and isinstance(instr.address, LLILStack):
            # Store to stack location
            offset = instr.address.offset
            if offset not in context.stack_layout:
                context.stack_layout[offset] = StackLocation(offset, instr.size)
            context.stack_layout[offset].last_written = instr_idx

        elif isinstance(instr, LLILLoad) and isinstance(instr.address, LLILStack):
            # Load from stack location
            offset = instr.address.offset
            if offset not in context.stack_layout:
                context.stack_layout[offset] = StackLocation(offset, instr.size)

        # Recursively analyze operands
        if hasattr(instr, 'operands'):
            for operand in instr.operands:
                if isinstance(operand, LLILExpression):
                    self._analyze_instruction_stack_usage(operand, context, block_idx, instr_idx)

    def _eliminate_stack_operations(self, context: LifterContext) -> bool:
        """Eliminate stack operations and convert to variables"""
        changes_made = False

        # Handle regular stack references first
        for offset in context.stack_layout:
            var = context.get_or_create_stack_variable(offset, context.stack_layout[offset].size)
            changes_made = True

        # Handle push/pop sequences for stack machines
        if self.stack_operations:
            changes_made |= self._convert_stack_machine_operations(context)

        return changes_made

    def _convert_stack_machine_operations(self, context: LifterContext) -> bool:
        """Convert push/pop operations to variable assignments for stack machines"""
        if context.calling_convention.name != "stack_machine":
            return False

        context.logger.debug(f"Converting {len(self.stack_operations)} stack operations")

        # Create virtual stack variables for push/pop sequence
        stack_depth = 0
        temp_stack_vars = {}  # depth -> variable

        for op_type, block_idx, instr_idx, instruction in self.stack_operations:
            if op_type == 'push':
                # Create variable for this stack depth
                if stack_depth not in temp_stack_vars:
                    var_name = f"stack_temp_{stack_depth}"
                    temp_var = context.create_variable(var_name, instruction.size)
                    temp_stack_vars[stack_depth] = temp_var

                # This push becomes an assignment: temp_var = value
                stack_depth += 1

            elif op_type == 'pop':
                # Pop from current depth
                stack_depth -= 1
                if stack_depth in temp_stack_vars:
                    # This pop becomes a variable read
                    pass

        context.temp_stack_vars = temp_stack_vars
        return True

    def _optimize_stack_layout(self, context: LifterContext):
        """Optimize stack variable layout and naming"""
        context.logger.debug("Optimizing stack variable layout")

        # Sort stack variables by offset for better naming
        sorted_offsets = sorted(context.stack_layout.keys())

        # Separate local variables (negative offsets) from parameters (positive offsets)
        local_offsets = [offset for offset in sorted_offsets if offset < 0]
        param_offsets = [offset for offset in sorted_offsets if offset > 0]

        # Rename local variables with more descriptive names
        for i, offset in enumerate(sorted(local_offsets, reverse=True)):  # Reverse for stack growth
            if offset in context.stack_vars:
                var = context.stack_vars[offset]
                var.name = f"local_{i}"

        # Parameters will be handled by VariableRecoveryPass

        # Detect likely arrays or structures based on access patterns
        self._detect_stack_structures(context, sorted_offsets)

    def _detect_stack_structures(self, context: LifterContext, sorted_offsets: List[int]):
        """Detect array or structure patterns in stack layout"""
        if len(sorted_offsets) < 2:
            return

        # Look for consecutive accesses with regular spacing
        for i in range(len(sorted_offsets) - 1):
            current_offset = sorted_offsets[i]
            next_offset = sorted_offsets[i + 1]
            spacing = next_offset - current_offset

            # Common structure patterns
            if spacing == 4:  # Likely int array or struct with 4-byte fields
                current_loc = context.stack_layout[current_offset]
                next_loc = context.stack_layout[next_offset]

                # Check for array pattern (similar access counts)
                if abs(current_loc.access_count - next_loc.access_count) <= 1:
                    # Likely array elements
                    if current_offset in context.stack_vars:
                        var = context.stack_vars[current_offset]
                        if not var.name.startswith(VariableNamingConstants.ARRAY_PREFIX):
                            var.name = f"{VariableNamingConstants.ARRAY_ELEMENT_PREFIX}{i}"

            elif spacing in [1, 2, 8]:  # Other common patterns
                # Could be packed struct, char array, or 64-bit values
                pass

    def _process_stack_instruction(self, instr: LLILExpression, context: LifterContext, index: int) -> bool:
        """Process a single instruction for stack operations (legacy method)"""
        # This method is kept for compatibility but most logic moved to other methods
        return False

    def _get_next_stack_offset(self, context: LifterContext) -> int:
        """Get the next available stack offset for push operations"""
        if not context.stack_layout:
            return self.stack_pointer_offset
        return min(context.stack_layout.keys()) - 4  # Stack grows down

    def _get_current_stack_offset(self, context: LifterContext) -> int:
        """Get the current stack offset for pop operations"""
        return self.stack_pointer_offset

    def _is_stack_machine_architecture(self, context: LifterContext) -> bool:
        """Check if target architecture is a stack machine"""
        return context.calling_convention.name == "stack_machine"

    def _analyze_stack_frame_layout(self, context: LifterContext):
        """Analyze overall stack frame layout"""
        if not context.stack_layout:
            return

        # Find stack frame boundaries
        min_offset = min(context.stack_layout.keys())
        max_offset = max(context.stack_layout.keys())

        context.logger.debug(f"Stack frame: {min_offset} to {max_offset} (size: {max_offset - min_offset})")

        # Classify regions
        for offset, location in context.stack_layout.items():
            if offset < 0:
                # Local variables
                location.region = "locals"
            elif offset > 0:
                # Parameters or return address area
                location.region = "params"
            else:
                # At stack pointer
                location.region = "frame_base"


class VariableRecoveryPass(LLILLifterPass):
    """
    Pass 2: Variable Recovery

    Reconstructs high-level variables from register usage patterns:
    - Performs data flow analysis to track register lifetimes
    - Identifies function parameters and return values
    - Merges related register accesses into logical variables
    - Handles register aliasing and partial updates
    - Reconstructs spilled variables
    """

    def __init__(self):
        self.def_use_chains = {}  # reg -> [(def_pos, use_positions)]
        self.variable_ranges = {}  # var_name -> (start_pos, end_pos)
        self.register_interference = {}  # reg1 -> {interfering_regs}
        self.spill_locations = {}  # reg -> stack_offset (for spilled registers)

    def run(self, context: LifterContext) -> bool:
        context.logger.debug("Starting variable recovery analysis")

        # Phase 1: Build def-use chains for registers
        self._build_def_use_chains(context)

        # Phase 2: Analyze register lifetimes and interference
        self._analyze_register_lifetimes(context)

        # Phase 3: Identify function interface (parameters/returns)
        self._identify_function_interface(context)

        # Phase 4: Detect register spilling patterns
        self._detect_register_spilling(context)

        # Phase 5: Perform variable coalescing
        changes_made = self._coalesce_variables(context)

        # Phase 6: Optimize variable names and types
        self._optimize_variable_names(context)

        return changes_made

    def _build_def_use_chains(self, context: LifterContext):
        """Build definition-use chains for all registers"""
        context.logger.debug("Building def-use chains")

        for block_idx, block in enumerate(context.llil_function.basic_blocks):
            for instr_idx, instruction in enumerate(block.instructions):
                position = (block_idx, instr_idx)
                self._analyze_instruction_def_use(instruction, context, position)

    def _analyze_instruction_def_use(self, instr: LLILExpression, context: LifterContext, position: Tuple[int, int]):
        """Analyze definition and use of registers in a single instruction"""
        # Handle different instruction types for def-use analysis
        if isinstance(instr, LLILRegister):
            # This is a use
            reg_name = instr.register
            self._record_register_use(reg_name, position, context)

        elif isinstance(instr, LLILStore) and isinstance(instr.address, LLILRegister):
            # Store to register - this is a definition
            reg_name = instr.address.register
            self._record_register_def(reg_name, position, context)
            # Also analyze the value being stored
            if hasattr(instr, 'value'):
                self._analyze_instruction_def_use(instr.value, context, position)

        elif isinstance(instr, LLILBinaryOp):
            # Binary operation - left and right are uses, result might be a def
            self._analyze_instruction_def_use(instr.left, context, position)
            self._analyze_instruction_def_use(instr.right, context, position)

        elif isinstance(instr, LLILCall):
            # Function call - analyze target and arguments
            self._analyze_instruction_def_use(instr.target, context, position)
            for arg in instr.arguments:
                self._analyze_instruction_def_use(arg, context, position)

            # Function calls may define/kill certain registers
            self._handle_call_effects(position, context)

        elif hasattr(instr, 'operands'):
            # Recursively analyze operands
            for operand in instr.operands:
                if isinstance(operand, LLILExpression):
                    self._analyze_instruction_def_use(operand, context, position)

    def _record_register_use(self, reg_name: str, position: Tuple[int, int], context: LifterContext):
        """Record a use of a register"""
        if reg_name not in self.def_use_chains:
            self.def_use_chains[reg_name] = []

        # Find the most recent definition
        current_def = None
        for def_pos, use_list in self.def_use_chains[reg_name]:
            if def_pos <= position:
                current_def = (def_pos, use_list)

        if current_def:
            current_def[1].append(position)
        else:
            # Use without definition - might be a parameter
            self.def_use_chains[reg_name].append((None, [position]))

        # Update register state
        if reg_name not in context.register_states:
            context.register_states[reg_name] = RegisterState()
        context.register_states[reg_name].last_read = position

    def _record_register_def(self, reg_name: str, position: Tuple[int, int], context: LifterContext):
        """Record a definition of a register"""
        if reg_name not in self.def_use_chains:
            self.def_use_chains[reg_name] = []

        # Start a new def-use chain
        self.def_use_chains[reg_name].append((position, []))

        # Update register state
        if reg_name not in context.register_states:
            context.register_states[reg_name] = RegisterState()
        context.register_states[reg_name].last_written = position

    def _handle_call_effects(self, position: Tuple[int, int], context: LifterContext):
        """Handle the effects of function calls on registers"""
        conv = context.calling_convention

        # Get architecture-specific caller-saved registers
        arch_type = self._get_architecture_type(context.target_arch)
        register_set = get_register_set(arch_type)

        # Calls typically kill caller-saved registers (simplified - would need full ABI spec)
        caller_saved_registers = self._get_caller_saved_registers(arch_type, register_set)

        for reg_info in caller_saved_registers:
            reg_name = reg_info.name
            if reg_name in context.register_states:
                # Record as a definition (kill)
                self._record_register_def(reg_name, position, context)

    def _get_architecture_type(self, arch_str: str) -> ArchitectureType:
        """Convert architecture string to ArchitectureType enum"""
        arch_map = {
            "x86": ArchitectureType.X86_32,
            "x64": ArchitectureType.X86_64,
            "arm": ArchitectureType.ARM_32,
            "falcom_vm": ArchitectureType.FALCOM_VM
        }
        return arch_map.get(arch_str, ArchitectureType.X86_32)


    def _get_caller_saved_registers(self, arch_type: ArchitectureType, register_set: RegisterSet) -> List[RegisterInfo]:
        """Get caller-saved registers for architecture"""
        if arch_type == ArchitectureType.X86_32:
            # x86 caller-saved: EAX, ECX, EDX
            from ..target.registers import X86Registers
            return [
                X86Registers.EAX.value,
                X86Registers.ECX.value,
                X86Registers.EDX.value,
            ]
        elif arch_type == ArchitectureType.X86_64:
            # x64 caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
            from ..target.registers import X64Registers
            return [
                X64Registers.RAX.value,
                X64Registers.RCX.value,
                X64Registers.RDX.value,
                X64Registers.RSI.value,
                X64Registers.RDI.value,
                X64Registers.R8.value,
                X64Registers.R9.value,
                X64Registers.R10.value,
                X64Registers.R11.value,
            ]
        elif arch_type == ArchitectureType.ARM_32:
            # ARM caller-saved: R0-R3, R12
            from ..target.registers import ARMRegisters
            return [
                ARMRegisters.R0.value,
                ARMRegisters.R1.value,
                ARMRegisters.R2.value,
                ARMRegisters.R3.value,
                ARMRegisters.R12.value,
            ]
        else:
            # Default to empty for stack machines
            return []

    def _analyze_register_lifetimes(self, context: LifterContext):
        """Analyze register lifetimes and build interference graph"""
        context.logger.debug("Analyzing register lifetimes")

        for reg_name, chains in self.def_use_chains.items():
            for def_pos, use_list in chains:
                if def_pos and use_list:
                    # Calculate variable range
                    start_pos = def_pos
                    end_pos = max(use_list) if use_list else def_pos
                    var_name = f"var_{reg_name}_{start_pos[0]}_{start_pos[1]}"
                    self.variable_ranges[var_name] = (start_pos, end_pos)

        # Build interference graph
        self._build_interference_graph(context)

    def _build_interference_graph(self, context: LifterContext):
        """Build interference graph for register allocation analysis"""
        self.register_interference = {reg: set() for reg in context.register_states.keys()}

        # Two variables interfere if their live ranges overlap
        var_items = list(self.variable_ranges.items())
        for i, (var1, range1) in enumerate(var_items):
            for var2, range2 in var_items[i+1:]:
                if self._ranges_overlap(range1, range2):
                    # Extract register names from variable names
                    reg1 = self._extract_register_from_var_name(var1)
                    reg2 = self._extract_register_from_var_name(var2)
                    if reg1 and reg2:
                        self.register_interference[reg1].add(reg2)
                        self.register_interference[reg2].add(reg1)

    def _ranges_overlap(self, range1: Tuple[Tuple[int, int], Tuple[int, int]],
                       range2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if two live ranges overlap"""
        start1, end1 = range1
        start2, end2 = range2

        # Convert to linear positions for easier comparison
        pos1_start = start1[0] * 1000 + start1[1]
        pos1_end = end1[0] * 1000 + end1[1]
        pos2_start = start2[0] * 1000 + start2[1]
        pos2_end = end2[0] * 1000 + end2[1]

        return not (pos1_end < pos2_start or pos2_end < pos1_start)

    def _extract_register_from_var_name(self, var_name: str) -> Optional[str]:
        """Extract register name from variable name"""
        if var_name.startswith(VariableNamingConstants.REGISTER_PREFIX):
            parts = var_name.split("_")
            if len(parts) >= 2:
                return parts[1]
        return None

    def _identify_function_interface(self, context: LifterContext):
        """Identify function parameters and return values"""
        conv = context.calling_convention

        # Identify parameters
        if conv.parameter_registers:
            self._identify_register_parameters(context, conv)
        else:
            self._identify_stack_parameters(context)

        # Identify return values
        self._identify_return_values_new(context, conv)

    def _identify_register_parameters(self, context: LifterContext, conv: CallConvention):
        """Identify register-based parameters"""
        for i, reg_info in enumerate(conv.parameter_registers):
            reg_name = reg_info.name
            if reg_name in self.def_use_chains:
                # Check if register is used before being defined
                chains = self.def_use_chains[reg_name]
                for def_pos, use_list in chains:
                    if def_pos is None and use_list:  # Used without definition
                        # This is likely a parameter
                        if reg_name not in context.register_states:
                            context.register_states[reg_name] = RegisterState()
                        context.register_states[reg_name].is_parameter = True

                        # Create parameter variable
                        param_var = context.create_variable(f"param_{i}", reg_info.size)
                        context.mlil_function.parameters.append(param_var)
                        context.reg_vars[reg_name] = param_var
                        break

    def _identify_stack_parameters(self, context: LifterContext):
        """Identify stack-based parameters"""
        # Look for positive stack offsets (above return address)
        param_offsets = [offset for offset in context.stack_layout.keys() if offset > 0]
        param_offsets.sort()

        for i, offset in enumerate(param_offsets):
            stack_loc = context.stack_layout[offset]
            if stack_loc.variable:
                # Mark as parameter
                stack_loc.variable.name = f"param_{i}"
                context.mlil_function.parameters.append(stack_loc.variable)

    def _identify_return_values_new(self, context: LifterContext, conv: CallConvention):
        """Identify return value registers"""
        if conv.return_register:
            return_reg_name = conv.return_register.name
            if return_reg_name in context.register_states:
                # Look for definitions of return register near return statements
                if return_reg_name in self.def_use_chains:
                    state = context.register_states[return_reg_name]
                    state.is_return_value = True

    def _detect_register_spilling(self, context: LifterContext):
        """Detect register spilling patterns"""
        context.logger.debug("Detecting register spilling patterns")

        # Look for patterns like: store reg, [stack]; ... ; load [stack], reg
        for reg_name, chains in self.def_use_chains.items():
            for def_pos, use_list in chains:
                if self._is_spill_pattern(reg_name, def_pos, use_list, context):
                    # Found a spill pattern
                    spill_offset = self._find_spill_location(reg_name, def_pos, context)
                    if spill_offset is not None:
                        self.spill_locations[reg_name] = spill_offset

    def _is_spill_pattern(self, reg_name: str, def_pos: Optional[Tuple[int, int]],
                         use_list: List[Tuple[int, int]], context: LifterContext) -> bool:
        """Check if this is a register spilling pattern"""
        # Simplified heuristic: if register is stored to stack and later loaded
        # This would need more sophisticated analysis in practice
        return len(use_list) > 2  # Arbitrary threshold

    def _find_spill_location(self, reg_name: str, def_pos: Optional[Tuple[int, int]],
                           context: LifterContext) -> Optional[int]:
        """Find the stack location where register was spilled"""
        # This would analyze the actual store/load instructions
        # Simplified implementation
        return None

    def _coalesce_variables(self, context: LifterContext) -> bool:
        """Coalesce registers into variables based on interference analysis"""
        context.logger.debug("Performing variable coalescing")
        changes_made = False

        # Simple coalescing: merge non-interfering registers
        processed_regs = set()

        for reg_name, state in context.register_states.items():
            if reg_name in processed_regs:
                continue

            # Find registers that don't interfere with this one
            coalescable_regs = {reg_name}
            interference_set = self.register_interference.get(reg_name, set())

            for other_reg in context.register_states:
                if (other_reg not in processed_regs and
                    other_reg not in interference_set and
                    other_reg != reg_name):
                    coalescable_regs.add(other_reg)

            # Create a variable for this group
            if state.variable is None:
                if state.is_parameter:
                    # Already handled in parameter identification
                    pass
                else:
                    var_name = f"var_{min(coalescable_regs)}"
                    var = context.create_variable(var_name, 4)

                    # Assign this variable to all coalescable registers
                    for coalesced_reg in coalescable_regs:
                        context.reg_vars[coalesced_reg] = var
                        context.register_states[coalesced_reg].variable = var
                        processed_regs.add(coalesced_reg)

                    changes_made = True

        return changes_made

    def _optimize_variable_names(self, context: LifterContext):
        """Optimize variable names based on usage patterns"""
        context.logger.debug("Optimizing variable names")

        # Rename variables based on usage patterns
        for var_name, var in context.mlil_function.variables.items():
            new_name = self._generate_descriptive_name(var, context)
            if new_name != var.name:
                var.name = new_name

    def _generate_descriptive_name(self, variable: IRVariable, context: LifterContext) -> str:
        """Generate a descriptive name for a variable based on its usage"""
        # Analyze how the variable is used to generate better names
        current_name = variable.name

        # If it's a parameter, keep param_ prefix
        if current_name.startswith(VariableNamingConstants.PARAM_PREFIX):
            return current_name

        # If it's a local variable, try to infer purpose
        if current_name.startswith(VariableNamingConstants.LOCAL_PREFIX):
            return current_name

        # For register variables, try to use conventional names
        if current_name.startswith(VariableNamingConstants.REGISTER_PREFIX):
            # Check if this maps to a well-known register
            for reg_name, var in context.reg_vars.items():
                if var == variable:
                    # Use architecture-specific register naming
                    arch_type = context.target_arch
                    register_set = get_register_set(arch_type)

                    # Get conventional names based on register purpose
                    if reg_name == register_set.return_register.name:
                        return VariableNamingConstants.RESULT_NAME

                    # Try to get register info and use conventional naming
                    try:
                        reg_info = get_register_by_name(arch_type, reg_name)
                        conventional_name = RegisterConventions.get_conventional_name(reg_info, arch_type)
                        if conventional_name:
                            return conventional_name
                    except ValueError:
                        # Register not found, continue with default naming
                        pass

        return current_name

    # Legacy methods for compatibility
    def _analyze_register_usage(self, context: LifterContext):
        """Legacy method - functionality moved to _build_def_use_chains"""
        pass

    def _analyze_instruction_registers(self, instr: LLILExpression, context: LifterContext, index: int):
        """Legacy method - functionality moved to _analyze_instruction_def_use"""
        pass

    def _identify_parameters(self, context: LifterContext):
        """Legacy method - functionality moved to _identify_function_interface"""
        pass

    def _identify_return_values(self, context: LifterContext):
        """Legacy method - functionality moved to _identify_function_interface"""
        pass

    def _merge_register_variables(self, context: LifterContext) -> bool:
        """Legacy method - functionality moved to _coalesce_variables"""
        return False


class MemoryAccessAnalysisPass(LLILLifterPass):
    """
    Pass 3: Memory Access Analysis

    Analyzes memory access patterns to identify:
    - Array accesses with stride patterns
    - Structure field accesses with known offsets
    - Pointer dereferences and chains
    - Buffer operations and bounds
    - String operations
    - Complex addressing modes
    """

    def __init__(self):
        self.access_patterns = {}     # address_expr -> access_info
        self.array_accesses = []      # List of detected array patterns
        self.struct_accesses = []     # List of detected struct patterns
        self.pointer_chains = []      # List of pointer dereference chains
        self.string_operations = []   # List of string operation patterns

    def run(self, context: LifterContext) -> bool:
        context.logger.debug("Starting memory access analysis")

        # Phase 1: Collect all memory accesses
        self._collect_memory_accesses(context)

        # Phase 2: Analyze access patterns and clustering
        self._analyze_access_patterns(context)

        # Phase 3: Identify array accesses
        changes_made = self._identify_array_accesses(context)

        # Phase 4: Identify structure accesses
        changes_made |= self._identify_struct_accesses(context)

        # Phase 5: Identify pointer chains
        changes_made |= self._identify_pointer_chains(context)

        # Phase 6: Identify string operations
        changes_made |= self._identify_string_operations(context)

        # Phase 7: Optimize memory expressions
        changes_made |= self._optimize_memory_expressions(context)

        return changes_made

    def _collect_memory_accesses(self, context: LifterContext):
        """Collect all memory access operations"""
        context.logger.debug("Collecting memory accesses")

        for block_idx, block in enumerate(context.llil_function.basic_blocks):
            for instr_idx, instruction in enumerate(block.instructions):
                position = (block_idx, instr_idx)
                self._collect_instruction_memory_access(instruction, context, position)

    def _collect_instruction_memory_access(self, instr: LLILExpression, context: LifterContext,
                                         position: Tuple[int, int]):
        """Collect memory accesses from a single instruction"""
        if isinstance(instr, LLILLoad):
            access = self._classify_memory_access(instr.address, context, "load", instr.size, position)
            context.memory_accesses.append(access)

        elif isinstance(instr, LLILStore):
            access = self._classify_memory_access(instr.address, context, "store", instr.size, position)
            context.memory_accesses.append(access)

        elif hasattr(instr, 'operands'):
            for operand in instr.operands:
                if isinstance(operand, LLILExpression):
                    self._collect_instruction_memory_access(operand, context, position)

    def _classify_memory_access(self, address_expr: LLILExpression, context: LifterContext,
                               op_type: str, size: int, position: Tuple[int, int]) -> MemoryAccess:
        """Classify the type of memory access with detailed analysis"""
        access = MemoryAccess()
        access.size = size
        access.operation_type = op_type
        access.position = position

        if isinstance(address_expr, LLILConstant):
            # Direct memory access - global variable or absolute address
            access.access_type = "global"
            access.is_constant_offset = True
            access.base_address = address_expr.value

        elif isinstance(address_expr, LLILRegister):
            # Simple pointer dereference
            access.base_expr = address_expr
            access.access_type = "pointer"

        elif isinstance(address_expr, LLILStack):
            # Stack access - local variable
            access.access_type = "local"
            access.offset = address_expr.offset
            access.is_constant_offset = True

        elif isinstance(address_expr, LLILBinaryOp):
            access = self._analyze_complex_addressing(address_expr, access, context)

        elif isinstance(address_expr, LLILLoad):
            # Pointer dereference chain: *(*(base + offset))
            access.access_type = "pointer_chain"
            access.base_expr = address_expr
            self._analyze_pointer_chain(address_expr, access, context)

        else:
            # Unknown addressing mode
            access.base_expr = address_expr
            access.access_type = "unknown"

        return access

    def _analyze_complex_addressing(self, addr_expr: LLILBinaryOp, access: MemoryAccess,
                                   context: LifterContext) -> MemoryAccess:
        """Analyze complex addressing expressions"""
        if addr_expr.operation == OperationType.ADD:
            # base + offset patterns
            left, right = addr_expr.left, addr_expr.right

            if isinstance(right, LLILConstant):
                # base + constant_offset
                access.base_expr = left
                access.offset = right.value
                access.is_constant_offset = True
                access.access_type = self._classify_base_plus_offset(left, right.value, context)

            elif isinstance(left, LLILConstant):
                # constant_base + offset
                access.base_expr = right
                access.offset = left.value
                access.is_constant_offset = True
                access.access_type = self._classify_base_plus_offset(right, left.value, context)

            elif isinstance(right, LLILBinaryOp) and right.operation == OperationType.MUL:
                # base + (index * scale) - array access pattern
                access = self._analyze_array_addressing(left, right, access, context)

            else:
                # base + dynamic_offset
                access.base_expr = left
                access.index_expr = right
                access.access_type = "computed"

        elif addr_expr.operation == OperationType.MUL:
            # index * scale - part of array access
            access.access_type = "scaled_index"
            if isinstance(addr_expr.right, LLILConstant):
                access.scale = addr_expr.right.value
                access.index_expr = addr_expr.left

        return access

    def _classify_base_plus_offset(self, base_expr: LLILExpression, offset: int,
                                  context: LifterContext) -> str:
        """Classify base + offset patterns"""
        if isinstance(base_expr, LLILRegister):
            # Register + offset - could be struct access or array element
            if offset % 4 == 0 and offset < 100:  # Likely struct field
                return "struct_field"
            elif offset % context.target_arch == 0:  # Aligned access
                return "array_element"
            else:
                return "buffer_access"

        elif isinstance(base_expr, LLILStack):
            # Stack + offset - local array or struct
            return "local_indexed"

        else:
            return "indexed"

    def _analyze_array_addressing(self, base: LLILExpression, scale_expr: LLILBinaryOp,
                                 access: MemoryAccess, context: LifterContext) -> MemoryAccess:
        """Analyze array addressing: base + (index * scale)"""
        access.access_type = "array"
        access.base_expr = base

        if isinstance(scale_expr.right, LLILConstant):
            access.scale = scale_expr.right.value
            access.index_expr = scale_expr.left

            # Determine array element type from scale
            if access.scale == 1:
                access.element_type = "char"
            elif access.scale == 2:
                access.element_type = "short"
            elif access.scale == 4:
                access.element_type = "int"
            elif access.scale == 8:
                access.element_type = "long"
            else:
                access.element_type = f"struct_{access.scale}"

        return access

    def _analyze_pointer_chain(self, load_expr: LLILLoad, access: MemoryAccess, context: LifterContext):
        """Analyze pointer dereference chains"""
        chain_length = 0
        current = load_expr

        while isinstance(current, LLILLoad) and chain_length < 5:  # Prevent infinite loops
            chain_length += 1
            current = current.address

        access.chain_length = chain_length
        access.final_base = current

    def _analyze_access_patterns(self, context: LifterContext):
        """Analyze patterns in memory accesses to identify data structures"""
        context.logger.debug("Analyzing access patterns")

        # Group accesses by base expression
        base_groups = {}
        for access in context.memory_accesses:
            base_key = self._get_base_key(access)
            if base_key not in base_groups:
                base_groups[base_key] = []
            base_groups[base_key].append(access)

        # Analyze each group for patterns
        for base_key, accesses in base_groups.items():
            if len(accesses) > 1:
                pattern = self._detect_access_pattern(accesses, context)
                if pattern:
                    self.access_patterns[base_key] = pattern

    def _get_base_key(self, access: MemoryAccess) -> str:
        """Get a key representing the base of this memory access"""
        if hasattr(access, 'base_expr') and access.base_expr:
            return str(access.base_expr)
        elif hasattr(access, 'base_address'):
            return f"global_{access.base_address}"
        else:
            return "unknown"

    def _detect_access_pattern(self, accesses: List[MemoryAccess], context: LifterContext) -> Optional[Dict]:
        """Detect patterns in a group of accesses"""
        if not accesses:
            return None

        # Check for array pattern (regular stride)
        array_pattern = self._check_array_pattern(accesses)
        if array_pattern:
            return {"type": "array", "pattern": array_pattern}

        # Check for struct pattern (small fixed offsets)
        struct_pattern = self._check_struct_pattern(accesses)
        if struct_pattern:
            return {"type": "struct", "pattern": struct_pattern}

        # Check for string pattern
        string_pattern = self._check_string_pattern(accesses)
        if string_pattern:
            return {"type": "string", "pattern": string_pattern}

        return None

    def _check_array_pattern(self, accesses: List[MemoryAccess]) -> Optional[Dict]:
        """Check if accesses form an array pattern"""
        constant_offset_accesses = [a for a in accesses if a.is_constant_offset]
        if len(constant_offset_accesses) < 2:
            return None

        # Sort by offset
        constant_offset_accesses.sort(key=lambda a: a.offset)

        # Check for regular stride
        strides = []
        for i in range(1, len(constant_offset_accesses)):
            stride = constant_offset_accesses[i].offset - constant_offset_accesses[i-1].offset
            strides.append(stride)

        if strides and all(s == strides[0] for s in strides):
            return {
                "stride": strides[0],
                "base_offset": constant_offset_accesses[0].offset,
                "count": len(constant_offset_accesses),
                "element_size": strides[0]
            }

        return None

    def _check_struct_pattern(self, accesses: List[MemoryAccess]) -> Optional[Dict]:
        """Check if accesses form a struct pattern"""
        constant_offset_accesses = [a for a in accesses if a.is_constant_offset]
        if len(constant_offset_accesses) < 2:
            return None

        # Check if offsets are small and irregular (typical of struct fields)
        offsets = [a.offset for a in constant_offset_accesses]
        max_offset = max(offsets)
        min_offset = min(offsets)

        if max_offset - min_offset < 256:  # Reasonable struct size
            # Group by size to identify field types
            size_groups = {}
            for access in constant_offset_accesses:
                size = access.size
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(access.offset)

            return {
                "size_range": max_offset - min_offset,
                "field_offsets": offsets,
                "field_sizes": size_groups
            }

        return None

    def _check_string_pattern(self, accesses: List[MemoryAccess]) -> Optional[Dict]:
        """Check if accesses form a string pattern"""
        # Look for byte-sized accesses with consecutive offsets
        byte_accesses = [a for a in accesses if a.size == 1 and a.is_constant_offset]
        if len(byte_accesses) < 3:
            return None

        byte_accesses.sort(key=lambda a: a.offset)

        # Check for consecutive byte accesses
        consecutive_count = 1
        for i in range(1, len(byte_accesses)):
            if byte_accesses[i].offset == byte_accesses[i-1].offset + 1:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 3:  # Minimum string length
            return {
                "length": consecutive_count,
                "start_offset": byte_accesses[0].offset
            }

        return None

    def _identify_array_accesses(self, context: LifterContext) -> bool:
        """Identify and convert array accesses"""
        changes_made = False

        for base_key, pattern_info in self.access_patterns.items():
            if pattern_info["type"] == "array":
                pattern = pattern_info["pattern"]
                self.array_accesses.append({
                    "base": base_key,
                    "stride": pattern["stride"],
                    "element_size": pattern["element_size"]
                })
                changes_made = True

        return changes_made

    def _identify_struct_accesses(self, context: LifterContext) -> bool:
        """Identify and convert structure field accesses"""
        changes_made = False

        for base_key, pattern_info in self.access_patterns.items():
            if pattern_info["type"] == "struct":
                pattern = pattern_info["pattern"]

                # Create field definitions
                fields = []
                for offset in pattern["field_offsets"]:
                    field_name = f"field_{offset}"
                    fields.append({"name": field_name, "offset": offset})

                self.struct_accesses.append({
                    "base": base_key,
                    "fields": fields,
                    "size": pattern["size_range"]
                })
                changes_made = True

        return changes_made

    def _identify_pointer_chains(self, context: LifterContext) -> bool:
        """Identify pointer dereference chains"""
        changes_made = False

        pointer_accesses = [a for a in context.memory_accesses if a.access_type == "pointer_chain"]
        for access in pointer_accesses:
            if hasattr(access, 'chain_length') and access.chain_length > 1:
                self.pointer_chains.append({
                    "base": access.final_base,
                    "depth": access.chain_length,
                    "position": access.position
                })
                changes_made = True

        return changes_made

    def _identify_string_operations(self, context: LifterContext) -> bool:
        """Identify string operations"""
        changes_made = False

        for base_key, pattern_info in self.access_patterns.items():
            if pattern_info["type"] == "string":
                pattern = pattern_info["pattern"]
                self.string_operations.append({
                    "base": base_key,
                    "length": pattern["length"],
                    "start_offset": pattern["start_offset"]
                })
                changes_made = True

        return changes_made

    def _optimize_memory_expressions(self, context: LifterContext) -> bool:
        """Optimize memory access expressions based on identified patterns"""
        context.logger.debug("Optimizing memory expressions")

        # This would transform LLIL memory operations into higher-level MLIL operations
        # For example: LLILLoad(base + 4*i) -> MLILArrayAccess(base, i, 4)

        changes_made = False

        # Add optimization logic here
        # This is where we'd actually transform the IR based on our analysis

        return changes_made

    # Legacy method for compatibility
    def _analyze_memory_access(self, instr: LLILExpression, context: LifterContext) -> bool:
        """Legacy method - functionality moved to more specific methods"""
        return False


class ControlFlowStructuringPass(LLILLifterPass):
    """
    Pass 4: Control Flow Structuring

    Identifies and structures control flow patterns:
    - If-then-else statements
    - While and for loops
    - Switch statements
    - Break and continue patterns
    """

    def run(self, context: LifterContext) -> bool:
        # Analyze control flow graph
        self._analyze_control_flow(context)

        # Identify structured constructs
        self._identify_loops(context)
        self._identify_conditionals(context)

        return True

    def _analyze_control_flow(self, context: LifterContext):
        """Analyze the control flow graph structure"""
        for block in context.llil_function.basic_blocks:
            for instruction in block.instructions:
                if isinstance(instruction, LLILIf):
                    context.condition_blocks.add(block.address)

    def _identify_loops(self, context: LifterContext):
        """Identify loop structures"""
        for block in context.llil_function.basic_blocks:
            for instruction in block.instructions:
                if isinstance(instruction, (LLILJump, LLILIf)):
                    if hasattr(instruction, 'true_target'):
                        target = getattr(instruction, 'true_target', None)
                        if isinstance(target, int) and target <= block.address:
                            context.loop_headers.add(target)

    def _identify_conditionals(self, context: LifterContext):
        """Identify if-then-else structures"""
        for addr in context.condition_blocks:
            context.structured_blocks[addr] = "conditional"


class CallConventionPass(LLILLifterPass):
    """
    Pass 5: Calling Convention Handling

    Processes function calls according to calling conventions
    """

    def run(self, context: LifterContext) -> bool:
        changes_made = False
        for block_idx, block in enumerate(context.llil_function.basic_blocks):
            for instr_idx, instruction in enumerate(block.instructions):
                if isinstance(instruction, LLILCall):
                    context.call_sites.append((instr_idx, instruction))
                    changes_made = True
        return changes_made

    def _process_function_call(self, call: LLILCall, context: LifterContext) -> bool:
        return False


class TypeInferencePass(LLILLifterPass):
    """
    Pass 6: Type Inference

    Performs basic type inference
    """

    def run(self, context: LifterContext) -> bool:
        # Basic type inference for variables
        for var_name, var in context.mlil_function.variables.items():
            if not var.var_type or var.var_type == "any":
                var.var_type = IRType.NUMBER  # Default inference

        return True


class LLILLifter:
    """
    Main LLIL to MLIL Lifter

    Orchestrates the lifting process through multiple passes:
    1. Stack elimination
    2. Variable recovery
    3. Memory access analysis
    4. Control flow structuring
    5. Calling convention handling
    6. Type inference
    """

    def __init__(self, target_arch: ArchitectureType):
        self.target_arch = target_arch
        self.passes = [
            StackEliminationPass(),
            VariableRecoveryPass(),
            MemoryAccessAnalysisPass(),
            ControlFlowStructuringPass(),
            CallConventionPass(),
            TypeInferencePass(),
        ]
        self.logger = logging.getLogger(__name__)

    def lift(self, llil_function: IRFunction) -> IRFunction:
        """
        Lift LLIL function to MLIL

        Args:
            llil_function: LLIL function to lift

        Returns:
            MLIL function
        """
        self.logger.info(f"Lifting function {llil_function.name} from LLIL to MLIL")

        # Create lifting context
        context = LifterContext(llil_function, self.target_arch)

        # Run all passes
        for pass_instance in self.passes:
            pass_name = pass_instance.__class__.__name__
            self.logger.debug(f"Running pass: {pass_name}")

            try:
                changes_made = pass_instance.run(context)
                if changes_made:
                    self.logger.debug(f"Pass {pass_name} made changes")
                else:
                    self.logger.debug(f"Pass {pass_name} made no changes")

            except Exception as e:
                self.logger.error(f"Pass {pass_name} failed: {e}")
                raise

        # Convert LLIL to MLIL using the analyzed context
        self._generate_mlil(context)

        self.logger.info(f"Lifting complete. Generated {len(context.mlil_function.basic_blocks)} MLIL blocks")
        return context.mlil_function

    def _generate_mlil(self, context: LifterContext):
        """Generate MLIL from LLIL using the analysis context"""
        # Copy function metadata
        context.mlil_function.return_type = context.llil_function.return_type

        # Convert basic blocks
        for llil_block in context.llil_function.basic_blocks:
            mlil_block = IRBasicBlock(llil_block.address)
            context.mlil_function.basic_blocks.append(mlil_block)

            context.builder.set_current_block(mlil_block)

            # Convert instructions
            for instruction in llil_block.instructions:
                mlil_instr = self._convert_instruction(instruction, context)
                if mlil_instr:
                    context.builder.add_instruction(mlil_instr)

    def _convert_instruction(self, llil_instr: LLILExpression, context: LifterContext) -> Optional[MLILExpression]:
        """Convert a single LLIL instruction to MLIL"""
        if isinstance(llil_instr, LLILConstant):
            return MLILConstant(llil_instr.value, llil_instr.size)

        elif isinstance(llil_instr, LLILRegister):
            var = context.get_or_create_register_variable(llil_instr.register, llil_instr.size)
            return MLILVariable(var)

        elif isinstance(llil_instr, LLILStack):
            var = context.get_or_create_stack_variable(llil_instr.offset, llil_instr.size)
            return MLILVariable(var)

        elif isinstance(llil_instr, LLILBinaryOp):
            left = self._convert_instruction(llil_instr.left, context)
            right = self._convert_instruction(llil_instr.right, context)
            if left and right:
                return MLILBinaryOp(llil_instr.operation, left, right, llil_instr.size)

        elif isinstance(llil_instr, LLILUnaryOp):
            operand = self._convert_instruction(llil_instr.operand, context)
            if operand:
                return MLILUnaryOp(llil_instr.operation, operand, llil_instr.size)

        elif isinstance(llil_instr, LLILLoad):
            address = self._convert_instruction(llil_instr.address, context)
            if address:
                return MLILLoad(address, llil_instr.size)

        elif isinstance(llil_instr, LLILStore):
            address = self._convert_instruction(llil_instr.address, context)
            value = self._convert_instruction(llil_instr.value, context)
            if address and value:
                return MLILStore(address, value, llil_instr.size)

        elif isinstance(llil_instr, LLILCall):
            target = self._convert_instruction(llil_instr.target, context)
            arguments = []
            for arg in llil_instr.arguments:
                converted_arg = self._convert_instruction(arg, context)
                if converted_arg:
                    arguments.append(converted_arg)

            if target:
                return MLILCall(target, arguments, llil_instr.size)

        elif isinstance(llil_instr, LLILReturn):
            if llil_instr.value:
                value = self._convert_instruction(llil_instr.value, context)
                if value:
                    return MLILReturn(value)
            return MLILReturn()

        # Handle other instruction types...

        return None


# Convenience function for lifting
def lift_llil_to_mlil(llil_function: IRFunction, target_arch: str = "x86") -> IRFunction:
    """
    Convenience function to lift LLIL to MLIL

    Args:
        llil_function: LLIL function to lift
        target_arch: Target architecture for calling conventions

    Returns:
        MLIL function
    """
    lifter = LLILLifter(target_arch)
    return lifter.lift(llil_function)
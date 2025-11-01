"""
Target capability models

Defines what each target architecture can and cannot do,
used for legalization and instruction selection.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
try:
    from ..ir.base import OperationType
except ImportError:
    from decompiler3.ir.base import OperationType


class AddressingMode(Enum):
    """Supported addressing modes"""
    IMMEDIATE = auto()      # #constant
    REGISTER = auto()       # register
    MEMORY = auto()         # [address]
    REGISTER_OFFSET = auto() # [reg + offset]
    REGISTER_INDEX = auto()  # [base + index * scale]
    STACK_RELATIVE = auto()  # [sp + offset]


class DataType(Enum):
    """Supported data types"""
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    POINTER = auto()
    BOOLEAN = auto()


@dataclass
class RegisterClass:
    """Register class definition"""
    name: str
    count: int
    size: int  # in bytes
    can_hold_types: List[DataType]
    aliases: List[str] = field(default_factory=list)


@dataclass
class InstructionCapability:
    """Capability information for an instruction"""
    operation: OperationType
    supported_types: List[DataType]
    addressing_modes: List[AddressingMode]
    latency: int = 1
    throughput: int = 1
    has_side_effects: bool = False


class TargetCapability:
    """Target architecture capability model"""

    def __init__(self, name: str):
        self.name = name
        self.pointer_size = 4  # bytes
        self.word_size = 4     # bytes
        self.endianness = "little"  # little or big
        self.stack_grows_down = True

        # Instruction set capabilities
        self.supported_operations: Dict[OperationType, InstructionCapability] = {}
        self.has_division = True
        self.has_modulo = True
        self.has_floating_point = True
        self.has_conditional_moves = False
        self.has_jump_tables = True
        self.has_call_stack = True

        # Memory and addressing
        self.max_immediate_size = 32  # bits
        self.addressing_modes: Set[AddressingMode] = {
            AddressingMode.IMMEDIATE,
            AddressingMode.REGISTER,
            AddressingMode.MEMORY
        }

        # Register model
        self.is_stack_machine = False
        self.register_classes: Dict[str, RegisterClass] = {}
        self.special_registers: Dict[str, str] = {}  # purpose -> register name

        # Calling convention
        self.calling_convention = "cdecl"
        self.argument_registers: List[str] = []
        self.return_registers: List[str] = []
        self.callee_saved_registers: List[str] = []
        self.caller_saved_registers: List[str] = []

        # Stack frame
        self.stack_alignment = 4  # bytes
        self.has_frame_pointer = True
        self.frame_pointer_register = "ebp"
        self.stack_pointer_register = "esp"

        # Optimization preferences
        self.prefer_registers = True
        self.prefer_small_immediates = True
        self.branch_cost = 2
        self.memory_latency = 3

    def add_register_class(self, reg_class: RegisterClass):
        """Add a register class to this target"""
        self.register_classes[reg_class.name] = reg_class

    def add_instruction_capability(self, capability: InstructionCapability):
        """Add instruction capability"""
        self.supported_operations[capability.operation] = capability

    def supports_operation(self, op: OperationType) -> bool:
        """Check if target supports an operation"""
        return op in self.supported_operations

    def supports_type(self, op: OperationType, data_type: DataType) -> bool:
        """Check if target supports an operation on a specific type"""
        if op not in self.supported_operations:
            return False
        return data_type in self.supported_operations[op].supported_types

    def supports_addressing_mode(self, op: OperationType, mode: AddressingMode) -> bool:
        """Check if target supports an addressing mode for an operation"""
        if op not in self.supported_operations:
            return False
        return mode in self.supported_operations[op].addressing_modes

    def get_register_for_type(self, data_type: DataType) -> Optional[str]:
        """Get a suitable register class for a data type"""
        for class_name, reg_class in self.register_classes.items():
            if data_type in reg_class.can_hold_types:
                return class_name
        return None

    def can_represent_immediate(self, value: int) -> bool:
        """Check if an immediate value can be represented"""
        return -(2**(self.max_immediate_size-1)) <= value < 2**(self.max_immediate_size-1)


class X86Capability(TargetCapability):
    """x86 architecture capability model"""

    def __init__(self):
        super().__init__("x86")
        self.pointer_size = 4
        self.word_size = 4

        # Register classes
        self.add_register_class(RegisterClass(
            "general", 8, 4,
            [DataType.INT8, DataType.INT16, DataType.INT32, DataType.POINTER],
            ["eax", "ebx", "ecx", "edx", "esi", "edi", "esp", "ebp"]
        ))

        self.add_register_class(RegisterClass(
            "floating", 8, 8,
            [DataType.FLOAT32, DataType.FLOAT64],
            ["st0", "st1", "st2", "st3", "st4", "st5", "st6", "st7"]
        ))

        # Special registers
        self.special_registers = {
            "stack_pointer": "esp",
            "frame_pointer": "ebp",
            "accumulator": "eax",
            "counter": "ecx"
        }

        # Calling convention
        self.argument_registers = []  # x86 uses stack for arguments
        self.return_registers = ["eax", "edx"]
        self.callee_saved_registers = ["ebx", "esi", "edi", "ebp"]
        self.caller_saved_registers = ["eax", "ecx", "edx"]

        # Addressing modes
        self.addressing_modes = {
            AddressingMode.IMMEDIATE,
            AddressingMode.REGISTER,
            AddressingMode.MEMORY,
            AddressingMode.REGISTER_OFFSET,
            AddressingMode.REGISTER_INDEX
        }

        # Instruction capabilities
        self._add_x86_instructions()

    def _add_x86_instructions(self):
        """Add x86-specific instruction capabilities"""
        # Arithmetic
        self.add_instruction_capability(InstructionCapability(
            OperationType.ADD,
            [DataType.INT8, DataType.INT16, DataType.INT32],
            [AddressingMode.REGISTER, AddressingMode.IMMEDIATE, AddressingMode.MEMORY],
            latency=1, throughput=1
        ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.SUB,
            [DataType.INT8, DataType.INT16, DataType.INT32],
            [AddressingMode.REGISTER, AddressingMode.IMMEDIATE, AddressingMode.MEMORY],
            latency=1, throughput=1
        ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.MUL,
            [DataType.INT8, DataType.INT16, DataType.INT32],
            [AddressingMode.REGISTER, AddressingMode.IMMEDIATE, AddressingMode.MEMORY],
            latency=3, throughput=1
        ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.DIV,
            [DataType.INT16, DataType.INT32],  # No 8-bit div
            [AddressingMode.REGISTER, AddressingMode.MEMORY],
            latency=25, throughput=25  # Very expensive
        ))

        # Bitwise
        self.add_instruction_capability(InstructionCapability(
            OperationType.AND,
            [DataType.INT8, DataType.INT16, DataType.INT32],
            [AddressingMode.REGISTER, AddressingMode.IMMEDIATE, AddressingMode.MEMORY],
            latency=1, throughput=1
        ))

        # Memory operations
        self.add_instruction_capability(InstructionCapability(
            OperationType.LOAD,
            [DataType.INT8, DataType.INT16, DataType.INT32, DataType.POINTER],
            [AddressingMode.MEMORY, AddressingMode.REGISTER_OFFSET, AddressingMode.REGISTER_INDEX],
            latency=3, throughput=1
        ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.STORE,
            [DataType.INT8, DataType.INT16, DataType.INT32, DataType.POINTER],
            [AddressingMode.MEMORY, AddressingMode.REGISTER_OFFSET, AddressingMode.REGISTER_INDEX],
            latency=1, throughput=1, has_side_effects=True
        ))

        # Control flow
        self.add_instruction_capability(InstructionCapability(
            OperationType.JUMP,
            [DataType.POINTER],
            [AddressingMode.IMMEDIATE, AddressingMode.REGISTER, AddressingMode.MEMORY],
            latency=1, throughput=1
        ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.CALL,
            [DataType.POINTER],
            [AddressingMode.IMMEDIATE, AddressingMode.REGISTER, AddressingMode.MEMORY],
            latency=2, throughput=1, has_side_effects=True
        ))


class FalcomVMCapability(TargetCapability):
    """Falcom script VM capability model"""

    def __init__(self):
        super().__init__("falcom_vm")
        self.pointer_size = 4
        self.word_size = 4

        # Stack-based VM
        self.is_stack_machine = True
        self.has_call_stack = True
        self.prefer_registers = False

        # No traditional registers - everything through stack
        self.register_classes = {}

        # Special stack operations
        self.addressing_modes = {
            AddressingMode.IMMEDIATE,
            AddressingMode.STACK_RELATIVE
        }

        # VM instruction capabilities
        self._add_falcom_vm_instructions()

    def _add_falcom_vm_instructions(self):
        """Add Falcom VM specific instruction capabilities"""
        # Stack operations
        self.add_instruction_capability(InstructionCapability(
            OperationType.CONST,
            [DataType.INT32, DataType.FLOAT32, DataType.POINTER],
            [AddressingMode.IMMEDIATE],
            latency=1, throughput=1
        ))

        # Arithmetic (stack-based)
        for op in [OperationType.ADD, OperationType.SUB, OperationType.MUL]:
            self.add_instruction_capability(InstructionCapability(
                op,
                [DataType.INT32, DataType.FLOAT32],
                [],  # Stack-based, no addressing modes
                latency=1, throughput=1
            ))

        # Memory operations
        self.add_instruction_capability(InstructionCapability(
            OperationType.LOAD,
            [DataType.INT32, DataType.FLOAT32, DataType.POINTER],
            [AddressingMode.STACK_RELATIVE],
            latency=1, throughput=1
        ))

        self.add_instruction_capability(InstructionCapability(
            OperationType.STORE,
            [DataType.INT32, DataType.FLOAT32, DataType.POINTER],
            [AddressingMode.STACK_RELATIVE],
            latency=1, throughput=1, has_side_effects=True
        ))

        # Control flow
        self.add_instruction_capability(InstructionCapability(
            OperationType.CALL,
            [DataType.INT32],  # Function ID
            [AddressingMode.IMMEDIATE],
            latency=10, throughput=1, has_side_effects=True
        ))


class ARMCapability(TargetCapability):
    """ARM architecture capability model"""

    def __init__(self):
        super().__init__("arm")
        self.pointer_size = 4
        self.word_size = 4
        self.has_conditional_moves = True

        # Register classes
        self.add_register_class(RegisterClass(
            "general", 16, 4,
            [DataType.INT8, DataType.INT16, DataType.INT32, DataType.POINTER],
            [f"r{i}" for i in range(16)]
        ))

        # Special registers
        self.special_registers = {
            "stack_pointer": "sp",
            "link_register": "lr",
            "program_counter": "pc",
            "frame_pointer": "fp"
        }

        # Calling convention (AAPCS)
        self.argument_registers = ["r0", "r1", "r2", "r3"]
        self.return_registers = ["r0", "r1"]
        self.callee_saved_registers = ["r4", "r5", "r6", "r7", "r8", "r9", "r10", "fp"]
        self.caller_saved_registers = ["r0", "r1", "r2", "r3", "r12", "lr"]

        # Addressing modes
        self.addressing_modes = {
            AddressingMode.IMMEDIATE,
            AddressingMode.REGISTER,
            AddressingMode.REGISTER_OFFSET,
            AddressingMode.REGISTER_INDEX
        }

        self._add_arm_instructions()

    def _add_arm_instructions(self):
        """Add ARM-specific instruction capabilities"""
        # Most ARM instructions can be conditional
        for op in [OperationType.ADD, OperationType.SUB, OperationType.MUL]:
            self.add_instruction_capability(InstructionCapability(
                op,
                [DataType.INT32],
                [AddressingMode.REGISTER, AddressingMode.IMMEDIATE],
                latency=1, throughput=1
            ))


# Target capability registry
TARGET_CAPABILITIES: Dict[str, TargetCapability] = {
    "x86": X86Capability(),
    "falcom_vm": FalcomVMCapability(),
    "arm": ARMCapability(),
}


def get_target_capability(target_name: str) -> Optional[TargetCapability]:
    """Get capability model for a target"""
    return TARGET_CAPABILITIES.get(target_name)


def register_target_capability(capability: TargetCapability):
    """Register a new target capability"""
    TARGET_CAPABILITIES[capability.name] = capability
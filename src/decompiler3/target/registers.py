"""
Register definitions for different architectures

Provides type-safe register constants and enums instead of string literals.
"""

from enum import Enum, auto
from typing import List, Dict, Set
from dataclasses import dataclass


class ArchitectureType(Enum):
    """Supported target architectures"""
    X86_32 = "x86"
    X86_64 = "x64"
    ARM_32 = "arm"
    ARM_64 = "arm64"
    FALCOM_VM = "falcom_vm"
    MIPS_32 = "mips"
    RISCV_64 = "riscv64"


class RegisterClass(Enum):
    """Register classification"""
    GENERAL_PURPOSE = auto()
    FLOATING_POINT = auto()
    VECTOR = auto()
    CONTROL = auto()
    SEGMENT = auto()
    STACK_POINTER = auto()
    BASE_POINTER = auto()
    INSTRUCTION_POINTER = auto()


@dataclass
class RegisterInfo:
    """Information about a specific register"""
    name: str
    size: int  # Size in bytes
    reg_class: RegisterClass
    aliases: List[str] = None  # Alternative names (e.g., AX for EAX low 16 bits)

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class X86Registers(Enum):
    """x86 32-bit registers"""
    # General purpose registers
    EAX = RegisterInfo("eax", 4, RegisterClass.GENERAL_PURPOSE, ["ax", "al", "ah"])
    EBX = RegisterInfo("ebx", 4, RegisterClass.GENERAL_PURPOSE, ["bx", "bl", "bh"])
    ECX = RegisterInfo("ecx", 4, RegisterClass.GENERAL_PURPOSE, ["cx", "cl", "ch"])
    EDX = RegisterInfo("edx", 4, RegisterClass.GENERAL_PURPOSE, ["dx", "dl", "dh"])
    ESI = RegisterInfo("esi", 4, RegisterClass.GENERAL_PURPOSE, ["si"])
    EDI = RegisterInfo("edi", 4, RegisterClass.GENERAL_PURPOSE, ["di"])
    EBP = RegisterInfo("ebp", 4, RegisterClass.BASE_POINTER, ["bp"])
    ESP = RegisterInfo("esp", 4, RegisterClass.STACK_POINTER, ["sp"])

    # Segment registers
    CS = RegisterInfo("cs", 2, RegisterClass.SEGMENT)
    DS = RegisterInfo("ds", 2, RegisterClass.SEGMENT)
    ES = RegisterInfo("es", 2, RegisterClass.SEGMENT)
    FS = RegisterInfo("fs", 2, RegisterClass.SEGMENT)
    GS = RegisterInfo("gs", 2, RegisterClass.SEGMENT)
    SS = RegisterInfo("ss", 2, RegisterClass.SEGMENT)

    # Control registers
    EIP = RegisterInfo("eip", 4, RegisterClass.INSTRUCTION_POINTER)
    EFLAGS = RegisterInfo("eflags", 4, RegisterClass.CONTROL)


class X64Registers(Enum):
    """x86-64 registers"""
    # General purpose registers (64-bit)
    RAX = RegisterInfo("rax", 8, RegisterClass.GENERAL_PURPOSE, ["eax", "ax", "al", "ah"])
    RBX = RegisterInfo("rbx", 8, RegisterClass.GENERAL_PURPOSE, ["ebx", "bx", "bl", "bh"])
    RCX = RegisterInfo("rcx", 8, RegisterClass.GENERAL_PURPOSE, ["ecx", "cx", "cl", "ch"])
    RDX = RegisterInfo("rdx", 8, RegisterClass.GENERAL_PURPOSE, ["edx", "dx", "dl", "dh"])
    RSI = RegisterInfo("rsi", 8, RegisterClass.GENERAL_PURPOSE, ["esi", "si", "sil"])
    RDI = RegisterInfo("rdi", 8, RegisterClass.GENERAL_PURPOSE, ["edi", "di", "dil"])
    RBP = RegisterInfo("rbp", 8, RegisterClass.BASE_POINTER, ["ebp", "bp", "bpl"])
    RSP = RegisterInfo("rsp", 8, RegisterClass.STACK_POINTER, ["esp", "sp", "spl"])

    # Additional 64-bit registers
    R8 = RegisterInfo("r8", 8, RegisterClass.GENERAL_PURPOSE, ["r8d", "r8w", "r8b"])
    R9 = RegisterInfo("r9", 8, RegisterClass.GENERAL_PURPOSE, ["r9d", "r9w", "r9b"])
    R10 = RegisterInfo("r10", 8, RegisterClass.GENERAL_PURPOSE, ["r10d", "r10w", "r10b"])
    R11 = RegisterInfo("r11", 8, RegisterClass.GENERAL_PURPOSE, ["r11d", "r11w", "r11b"])
    R12 = RegisterInfo("r12", 8, RegisterClass.GENERAL_PURPOSE, ["r12d", "r12w", "r12b"])
    R13 = RegisterInfo("r13", 8, RegisterClass.GENERAL_PURPOSE, ["r13d", "r13w", "r13b"])
    R14 = RegisterInfo("r14", 8, RegisterClass.GENERAL_PURPOSE, ["r14d", "r14w", "r14b"])
    R15 = RegisterInfo("r15", 8, RegisterClass.GENERAL_PURPOSE, ["r15d", "r15w", "r15b"])

    # Control registers
    RIP = RegisterInfo("rip", 8, RegisterClass.INSTRUCTION_POINTER)
    RFLAGS = RegisterInfo("rflags", 8, RegisterClass.CONTROL)


class ARMRegisters(Enum):
    """ARM 32-bit registers"""
    # General purpose registers
    R0 = RegisterInfo("r0", 4, RegisterClass.GENERAL_PURPOSE)
    R1 = RegisterInfo("r1", 4, RegisterClass.GENERAL_PURPOSE)
    R2 = RegisterInfo("r2", 4, RegisterClass.GENERAL_PURPOSE)
    R3 = RegisterInfo("r3", 4, RegisterClass.GENERAL_PURPOSE)
    R4 = RegisterInfo("r4", 4, RegisterClass.GENERAL_PURPOSE)
    R5 = RegisterInfo("r5", 4, RegisterClass.GENERAL_PURPOSE)
    R6 = RegisterInfo("r6", 4, RegisterClass.GENERAL_PURPOSE)
    R7 = RegisterInfo("r7", 4, RegisterClass.GENERAL_PURPOSE)
    R8 = RegisterInfo("r8", 4, RegisterClass.GENERAL_PURPOSE)
    R9 = RegisterInfo("r9", 4, RegisterClass.GENERAL_PURPOSE)
    R10 = RegisterInfo("r10", 4, RegisterClass.GENERAL_PURPOSE)
    R11 = RegisterInfo("r11", 4, RegisterClass.BASE_POINTER, ["fp"])
    R12 = RegisterInfo("r12", 4, RegisterClass.GENERAL_PURPOSE, ["ip"])
    R13 = RegisterInfo("r13", 4, RegisterClass.STACK_POINTER, ["sp"])
    R14 = RegisterInfo("r14", 4, RegisterClass.GENERAL_PURPOSE, ["lr"])
    R15 = RegisterInfo("r15", 4, RegisterClass.INSTRUCTION_POINTER, ["pc"])

    # Status register
    CPSR = RegisterInfo("cpsr", 4, RegisterClass.CONTROL)


class FalcomVMRegisters(Enum):
    """Falcom VM virtual registers"""
    # Stack machine registers
    STACK_TOP = RegisterInfo("stack_top", 4, RegisterClass.STACK_POINTER)
    STACK_BASE = RegisterInfo("stack_base", 4, RegisterClass.BASE_POINTER)

    # VM control registers
    PC = RegisterInfo("pc", 4, RegisterClass.INSTRUCTION_POINTER)
    FLAGS = RegisterInfo("flags", 4, RegisterClass.CONTROL)

    # Virtual general purpose registers
    TEMP0 = RegisterInfo("temp0", 4, RegisterClass.GENERAL_PURPOSE)
    TEMP1 = RegisterInfo("temp1", 4, RegisterClass.GENERAL_PURPOSE)
    TEMP2 = RegisterInfo("temp2", 4, RegisterClass.GENERAL_PURPOSE)
    TEMP3 = RegisterInfo("temp3", 4, RegisterClass.GENERAL_PURPOSE)


@dataclass
class RegisterSet:
    """Complete register set for an architecture"""
    architecture: ArchitectureType
    registers: Dict[str, RegisterInfo]
    general_purpose: List[RegisterInfo]
    parameter_registers: List[RegisterInfo]
    return_register: RegisterInfo
    stack_pointer: RegisterInfo
    base_pointer: RegisterInfo

    @classmethod
    def for_architecture(cls, arch: ArchitectureType) -> 'RegisterSet':
        """Create register set for specific architecture"""
        if arch == ArchitectureType.X86_32:
            return cls._create_x86_register_set()
        elif arch == ArchitectureType.X86_64:
            return cls._create_x64_register_set()
        elif arch == ArchitectureType.ARM_32:
            return cls._create_arm_register_set()
        elif arch == ArchitectureType.FALCOM_VM:
            return cls._create_falcom_vm_register_set()
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    @classmethod
    def _create_x86_register_set(cls) -> 'RegisterSet':
        """Create x86 32-bit register set"""
        registers = {reg.value.name: reg.value for reg in X86Registers}

        general_purpose = [
            X86Registers.EAX.value,
            X86Registers.EBX.value,
            X86Registers.ECX.value,
            X86Registers.EDX.value,
            X86Registers.ESI.value,
            X86Registers.EDI.value,
        ]

        return cls(
            architecture=ArchitectureType.X86_32,
            registers=registers,
            general_purpose=general_purpose,
            parameter_registers=[],  # x86 uses stack for parameters
            return_register=X86Registers.EAX.value,
            stack_pointer=X86Registers.ESP.value,
            base_pointer=X86Registers.EBP.value,
        )

    @classmethod
    def _create_x64_register_set(cls) -> 'RegisterSet':
        """Create x86-64 register set"""
        registers = {reg.value.name: reg.value for reg in X64Registers}

        general_purpose = [
            X64Registers.RAX.value,
            X64Registers.RBX.value,
            X64Registers.RCX.value,
            X64Registers.RDX.value,
            X64Registers.RSI.value,
            X64Registers.RDI.value,
            X64Registers.R8.value,
            X64Registers.R9.value,
            X64Registers.R10.value,
            X64Registers.R11.value,
            X64Registers.R12.value,
            X64Registers.R13.value,
            X64Registers.R14.value,
            X64Registers.R15.value,
        ]

        parameter_registers = [
            X64Registers.RDI.value,
            X64Registers.RSI.value,
            X64Registers.RDX.value,
            X64Registers.RCX.value,
            X64Registers.R8.value,
            X64Registers.R9.value,
        ]

        return cls(
            architecture=ArchitectureType.X86_64,
            registers=registers,
            general_purpose=general_purpose,
            parameter_registers=parameter_registers,
            return_register=X64Registers.RAX.value,
            stack_pointer=X64Registers.RSP.value,
            base_pointer=X64Registers.RBP.value,
        )

    @classmethod
    def _create_arm_register_set(cls) -> 'RegisterSet':
        """Create ARM 32-bit register set"""
        registers = {reg.value.name: reg.value for reg in ARMRegisters}

        general_purpose = [
            ARMRegisters.R0.value,
            ARMRegisters.R1.value,
            ARMRegisters.R2.value,
            ARMRegisters.R3.value,
            ARMRegisters.R4.value,
            ARMRegisters.R5.value,
            ARMRegisters.R6.value,
            ARMRegisters.R7.value,
            ARMRegisters.R8.value,
            ARMRegisters.R9.value,
            ARMRegisters.R10.value,
            ARMRegisters.R12.value,
        ]

        parameter_registers = [
            ARMRegisters.R0.value,
            ARMRegisters.R1.value,
            ARMRegisters.R2.value,
            ARMRegisters.R3.value,
        ]

        return cls(
            architecture=ArchitectureType.ARM_32,
            registers=registers,
            general_purpose=general_purpose,
            parameter_registers=parameter_registers,
            return_register=ARMRegisters.R0.value,
            stack_pointer=ARMRegisters.R13.value,
            base_pointer=ARMRegisters.R11.value,
        )

    @classmethod
    def _create_falcom_vm_register_set(cls) -> 'RegisterSet':
        """Create Falcom VM register set"""
        registers = {reg.value.name: reg.value for reg in FalcomVMRegisters}

        general_purpose = [
            FalcomVMRegisters.TEMP0.value,
            FalcomVMRegisters.TEMP1.value,
            FalcomVMRegisters.TEMP2.value,
            FalcomVMRegisters.TEMP3.value,
        ]

        return cls(
            architecture=ArchitectureType.FALCOM_VM,
            registers=registers,
            general_purpose=general_purpose,
            parameter_registers=[],  # Stack machine
            return_register=FalcomVMRegisters.STACK_TOP.value,
            stack_pointer=FalcomVMRegisters.STACK_TOP.value,
            base_pointer=FalcomVMRegisters.STACK_BASE.value,
        )


class CallingConventionType(Enum):
    """Standard calling conventions"""
    CDECL = "cdecl"         # x86 C calling convention
    STDCALL = "stdcall"     # x86 Windows API
    FASTCALL = "fastcall"   # x86 fast calling convention
    SYSTEMV = "systemv"     # x64 System V ABI (Linux)
    MS_X64 = "ms_x64"       # x64 Microsoft calling convention (Windows)
    AAPCS = "aapcs"         # ARM AAPCS
    STACK_MACHINE = "stack_machine"  # Pure stack machine


# Register lookup functions
def get_register_set(arch: ArchitectureType) -> RegisterSet:
    """Get complete register set for architecture"""
    return RegisterSet.for_architecture(arch)


def get_register_by_name(arch: ArchitectureType, name: str) -> RegisterInfo:
    """Get register info by name"""
    register_set = get_register_set(arch)
    if name in register_set.registers:
        return register_set.registers[name]

    # Check aliases
    for reg_info in register_set.registers.values():
        if name in reg_info.aliases:
            return reg_info

    raise ValueError(f"Unknown register '{name}' for architecture {arch}")


def is_parameter_register(arch: ArchitectureType, reg_name: str) -> bool:
    """Check if register is used for parameter passing"""
    register_set = get_register_set(arch)
    try:
        reg_info = get_register_by_name(arch, reg_name)
        return reg_info in register_set.parameter_registers
    except ValueError:
        return False


def is_return_register(arch: ArchitectureType, reg_name: str) -> bool:
    """Check if register is used for return values"""
    register_set = get_register_set(arch)
    try:
        reg_info = get_register_by_name(arch, reg_name)
        return reg_info == register_set.return_register
    except ValueError:
        return False
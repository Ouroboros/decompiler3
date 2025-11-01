"""
Instruction selection

Maps legalized LLIL to target-specific machine instructions.
Implements pattern matching and cost-based selection.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
try:
    from ..ir.base import IRFunction, IRBasicBlock, IRExpression, OperationType
    from ..ir.llil import *
    from .capability import TargetCapability, get_target_capability
except ImportError:
    from decompiler3.ir.base import IRFunction, IRBasicBlock, IRExpression, OperationType
    from decompiler3.ir.llil import *
    from decompiler3.target.capability import TargetCapability, get_target_capability


@dataclass
class MachineInstruction:
    """Target machine instruction"""
    opcode: str
    operands: List[str]
    size: int = 4
    cost: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __str__(self) -> str:
        if self.operands:
            return f"{self.opcode} {', '.join(self.operands)}"
        return self.opcode


class InstructionPattern:
    """Pattern for matching IR to machine instructions"""

    def __init__(self, pattern_name: str, ir_pattern: Any, machine_template: List[str], cost: int = 1):
        self.pattern_name = pattern_name
        self.ir_pattern = ir_pattern  # Could be operation type, class, or lambda
        self.machine_template = machine_template
        self.cost = cost

    def matches(self, expr: IRExpression) -> bool:
        """Check if this pattern matches an IR expression"""
        if isinstance(self.ir_pattern, type):
            return isinstance(expr, self.ir_pattern)
        elif isinstance(self.ir_pattern, OperationType):
            return expr.operation == self.ir_pattern
        elif callable(self.ir_pattern):
            return self.ir_pattern(expr)
        return False

    def instantiate(self, expr: IRExpression, context: 'InstructionSelector') -> List[MachineInstruction]:
        """Instantiate machine instructions from template"""
        instructions = []

        for template in self.machine_template:
            # Simple template substitution
            opcode, *operand_templates = template.split()
            operands = []

            for operand_template in operand_templates:
                operand = context.resolve_operand(operand_template, expr)
                operands.append(operand)

            instructions.append(MachineInstruction(opcode, operands, cost=self.cost))

        return instructions


class InstructionSelector:
    """Selects target instructions for LLIL expressions"""

    def __init__(self, target_capability: TargetCapability):
        self.target = target_capability
        self.patterns: List[InstructionPattern] = []
        self.register_allocator = SimpleRegisterAllocator(target_capability)
        self._initialize_patterns()

    def select_instructions(self, function: IRFunction) -> List[MachineInstruction]:
        """Select instructions for entire function"""
        instructions = []

        # Function prologue
        instructions.extend(self._generate_prologue(function))

        # Process basic blocks
        for block in function.basic_blocks:
            instructions.extend(self._select_block_instructions(block))

        # Function epilogue
        instructions.extend(self._generate_epilogue(function))

        return instructions

    def _select_block_instructions(self, block: IRBasicBlock) -> List[MachineInstruction]:
        """Select instructions for a basic block"""
        instructions = []

        # Block label
        instructions.append(MachineInstruction(f"block_{block.id[:8]}:", []))

        # Process each instruction
        for expr in block.instructions:
            selected = self._select_instruction(expr)
            instructions.extend(selected)

        return instructions

    def _select_instruction(self, expr: IRExpression) -> List[MachineInstruction]:
        """Select instructions for a single IR expression"""
        # Try to match patterns
        for pattern in self.patterns:
            if pattern.matches(expr):
                return pattern.instantiate(expr, self)

        # Fallback to default selection
        return self._default_selection(expr)

    def _default_selection(self, expr: IRExpression) -> List[MachineInstruction]:
        """Default instruction selection"""
        if isinstance(expr, LLILConstant):
            reg = self.register_allocator.allocate_register()
            if self.target.name == "x86":
                return [MachineInstruction("mov", [reg, f"${expr.value}"])]
            elif self.target.name == "falcom_vm":
                return [MachineInstruction("CONST", [str(expr.value)])]

        elif isinstance(expr, LLILBinaryOp):
            if self.target.name == "x86":
                return self._select_x86_binary_op(expr)
            elif self.target.name == "falcom_vm":
                return self._select_falcom_binary_op(expr)

        elif isinstance(expr, LLILCall):
            return self._select_call(expr)

        elif isinstance(expr, LLILLoad):
            return self._select_load(expr)

        elif isinstance(expr, LLILStore):
            return self._select_store(expr)

        # Unknown instruction
        return [MachineInstruction("unknown", [str(expr)])]

    def _select_x86_binary_op(self, expr: LLILBinaryOp) -> List[MachineInstruction]:
        """Select x86 instructions for binary operation"""
        op_map = {
            OperationType.ADD: "add",
            OperationType.SUB: "sub",
            OperationType.MUL: "imul",
            OperationType.AND: "and",
            OperationType.OR: "or",
            OperationType.XOR: "xor",
        }

        opcode = op_map.get(expr.operation, "unknown")
        left_reg = self.resolve_operand("$left", expr)
        right_operand = self.resolve_operand("$right", expr)

        return [MachineInstruction(opcode, [left_reg, right_operand])]

    def _select_falcom_binary_op(self, expr: LLILBinaryOp) -> List[MachineInstruction]:
        """Select Falcom VM instructions for binary operation"""
        op_map = {
            OperationType.ADD: "ADD",
            OperationType.SUB: "SUB",
            OperationType.MUL: "MUL",
            OperationType.DIV: "DIV",
            OperationType.MOD: "MOD",
            OperationType.AND: "AND",
            OperationType.OR: "OR",
            OperationType.XOR: "XOR",
            OperationType.CMP_E: "EQ",
            OperationType.CMP_NE: "NE",
            OperationType.CMP_SLT: "LT",
            OperationType.CMP_SLE: "LE",
        }

        instructions = []

        # Push operands (stack machine)
        instructions.extend(self._select_instruction(expr.left))
        instructions.extend(self._select_instruction(expr.right))

        # Execute operation
        opcode = op_map.get(expr.operation, "UNKNOWN")
        instructions.append(MachineInstruction(opcode, []))

        return instructions

    def _select_call(self, expr: LLILCall) -> List[MachineInstruction]:
        """Select call instructions"""
        instructions = []

        if self.target.name == "x86":
            # Push arguments in reverse order
            for arg in reversed(expr.arguments):
                arg_instructions = self._select_instruction(arg)
                instructions.extend(arg_instructions)
                # Assuming result is in a register, push it
                instructions.append(MachineInstruction("push", ["eax"]))

            # Call target
            if isinstance(expr.target, LLILConstant):
                instructions.append(MachineInstruction("call", [str(expr.target.value)]))
            else:
                target_reg = self.resolve_operand("$target", expr)
                instructions.append(MachineInstruction("call", [target_reg]))

        elif self.target.name == "falcom_vm":
            # Push arguments
            for arg in expr.arguments:
                instructions.extend(self._select_instruction(arg))

            # Call
            if isinstance(expr.target, LLILConstant):
                instructions.append(MachineInstruction("CALL", [str(expr.target.value)]))
            else:
                instructions.extend(self._select_instruction(expr.target))
                instructions.append(MachineInstruction("CALL_INDIRECT", []))

        return instructions

    def _select_load(self, expr: LLILLoad) -> List[MachineInstruction]:
        """Select load instructions"""
        if self.target.name == "x86":
            if isinstance(expr.address, LLILRegister):
                dest_reg = self.register_allocator.allocate_register()
                return [MachineInstruction("mov", [dest_reg, f"[{expr.address.register}]"])]
            elif isinstance(expr.address, LLILStack):
                dest_reg = self.register_allocator.allocate_register()
                offset = expr.address.offset
                return [MachineInstruction("mov", [dest_reg, f"[ebp{offset:+d}]"])]

        elif self.target.name == "falcom_vm":
            if isinstance(expr.address, LLILStack):
                return [MachineInstruction("LOAD_LOCAL", [str(expr.address.offset)])]
            else:
                # Load from computed address
                instructions = self._select_instruction(expr.address)
                instructions.append(MachineInstruction("LOAD", []))
                return instructions

        return [MachineInstruction("unknown_load", [str(expr.address)])]

    def _select_store(self, expr: LLILStore) -> List[MachineInstruction]:
        """Select store instructions"""
        instructions = []

        if self.target.name == "x86":
            # Get value into register
            value_instructions = self._select_instruction(expr.value)
            instructions.extend(value_instructions)

            if isinstance(expr.address, LLILRegister):
                instructions.append(MachineInstruction("mov", [f"[{expr.address.register}]", "eax"]))
            elif isinstance(expr.address, LLILStack):
                offset = expr.address.offset
                instructions.append(MachineInstruction("mov", [f"[ebp{offset:+d}]", "eax"]))

        elif self.target.name == "falcom_vm":
            # Push value
            instructions.extend(self._select_instruction(expr.value))

            if isinstance(expr.address, LLILStack):
                instructions.append(MachineInstruction("STORE_LOCAL", [str(expr.address.offset)]))
            else:
                # Store to computed address
                instructions.extend(self._select_instruction(expr.address))
                instructions.append(MachineInstruction("STORE", []))

        return instructions

    def resolve_operand(self, template: str, expr: IRExpression) -> str:
        """Resolve operand template to actual operand"""
        if template == "$left" and hasattr(expr, 'left'):
            return self._expr_to_operand(expr.left)
        elif template == "$right" and hasattr(expr, 'right'):
            return self._expr_to_operand(expr.right)
        elif template == "$operand" and hasattr(expr, 'operand'):
            return self._expr_to_operand(expr.operand)
        elif template == "$target" and hasattr(expr, 'target'):
            return self._expr_to_operand(expr.target)
        elif template.startswith("$"):
            # Register allocation
            return self.register_allocator.allocate_register()
        return template

    def _expr_to_operand(self, expr: IRExpression) -> str:
        """Convert IR expression to operand string"""
        if isinstance(expr, LLILConstant):
            if self.target.name == "x86":
                return f"${expr.value}"
            else:
                return str(expr.value)
        elif isinstance(expr, LLILRegister):
            return expr.register
        elif isinstance(expr, LLILStack):
            if expr.offset >= 0:
                return f"[ebp+{expr.offset}]"
            else:
                return f"[ebp{expr.offset}]"
        else:
            # Need to generate instructions to compute this
            return self.register_allocator.allocate_register()

    def _generate_prologue(self, function: IRFunction) -> List[MachineInstruction]:
        """Generate function prologue"""
        if self.target.name == "x86":
            return [
                MachineInstruction("push", ["ebp"]),
                MachineInstruction("mov", ["ebp", "esp"]),
                MachineInstruction("sub", ["esp", "32"])  # Reserve stack space
            ]
        elif self.target.name == "falcom_vm":
            return [
                MachineInstruction("FUNC_ENTRY", [function.name or "anonymous"])
            ]
        return []

    def _generate_epilogue(self, function: IRFunction) -> List[MachineInstruction]:
        """Generate function epilogue"""
        if self.target.name == "x86":
            return [
                MachineInstruction("mov", ["esp", "ebp"]),
                MachineInstruction("pop", ["ebp"]),
                MachineInstruction("ret", [])
            ]
        elif self.target.name == "falcom_vm":
            return [
                MachineInstruction("FUNC_EXIT", [])
            ]
        return []

    def _initialize_patterns(self):
        """Initialize instruction selection patterns"""
        # Add common patterns
        if self.target.name == "x86":
            self._add_x86_patterns()
        elif self.target.name == "falcom_vm":
            self._add_falcom_patterns()

    def _add_x86_patterns(self):
        """Add x86-specific patterns"""
        # mov reg, immediate
        self.patterns.append(InstructionPattern(
            "load_immediate",
            lambda expr: isinstance(expr, LLILConstant),
            ["mov $reg ${value}"],
            cost=1
        ))

        # add reg, reg
        self.patterns.append(InstructionPattern(
            "add_reg_reg",
            lambda expr: (isinstance(expr, LLILBinaryOp) and
                         expr.operation == OperationType.ADD),
            ["add $left $right"],
            cost=1
        ))

    def _add_falcom_patterns(self):
        """Add Falcom VM specific patterns"""
        # CONST value
        self.patterns.append(InstructionPattern(
            "const",
            lambda expr: isinstance(expr, LLILConstant),
            ["CONST ${value}"],
            cost=1
        ))


class SimpleRegisterAllocator:
    """Simple register allocator for instruction selection"""

    def __init__(self, target_capability: TargetCapability):
        self.target = target_capability
        self.current_register = 0
        self.available_registers = self._get_available_registers()

    def _get_available_registers(self) -> List[str]:
        """Get list of available registers"""
        if self.target.name == "x86":
            return ["eax", "ebx", "ecx", "edx", "esi", "edi"]
        elif self.target.name == "arm":
            return [f"r{i}" for i in range(12)]  # r0-r11
        else:
            return ["reg0", "reg1", "reg2", "reg3"]  # Generic

    def allocate_register(self) -> str:
        """Allocate a register"""
        if self.target.is_stack_machine:
            return "stack"  # Stack machines don't use registers

        reg = self.available_registers[self.current_register % len(self.available_registers)]
        self.current_register += 1
        return reg

    def free_register(self, register: str):
        """Free a register (placeholder)"""
        pass


def select_instructions_for_target(function: IRFunction, target_name: str) -> List[MachineInstruction]:
    """Convenience function to select instructions for a target"""
    capability = get_target_capability(target_name)
    if not capability:
        raise ValueError(f"Unknown target: {target_name}")

    selector = InstructionSelector(capability)
    return selector.select_instructions(function)
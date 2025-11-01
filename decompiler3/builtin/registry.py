"""
Built-in function registry and management system

Provides unified semantic entry points for script domain extensions.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum, auto
try:
    from ..ir.base import IRExpression, Architecture
except ImportError:
    from decompiler3.ir.base import IRExpression, Architecture


class SideEffect(Enum):
    """Types of side effects a built-in function can have"""
    NONE = auto()          # Pure function, no side effects
    MEMORY_READ = auto()   # Reads memory
    MEMORY_WRITE = auto()  # Writes memory
    IO = auto()            # Input/output operations
    CONTROL = auto()       # Control flow changes
    SYSTEM = auto()        # System calls
    UNKNOWN = auto()       # Unknown or mixed side effects


@dataclass
class BuiltinSignature:
    """Signature information for a built-in function"""
    name: str
    parameters: List[str]  # Parameter type names
    return_type: Optional[str]
    side_effects: List[SideEffect]
    description: str
    category: str = "general"

    def __post_init__(self):
        if not self.side_effects:
            self.side_effects = [SideEffect.UNKNOWN]


@dataclass
class BuiltinMapping:
    """Mapping between built-in and target implementation"""
    direct_opcode: Optional[str] = None        # Direct 1:1 mapping to opcode
    expansion: Optional[List[str]] = None      # Expand to sequence of opcodes
    library_call: Optional[str] = None         # Call to runtime library function
    fallback_error: Optional[str] = None       # Error if not supported


class BuiltinFunction:
    """Built-in function definition with semantic information"""

    def __init__(self, signature: BuiltinSignature):
        self.signature = signature
        self.mappings: Dict[str, BuiltinMapping] = {}  # target -> mapping
        self.validators: List[Callable[[List[IRExpression]], bool]] = []

    def add_target_mapping(self, target: str, mapping: BuiltinMapping):
        """Add target-specific mapping for this built-in"""
        self.mappings[target] = mapping

    def add_validator(self, validator: Callable[[List[IRExpression]], bool]):
        """Add argument validator"""
        self.validators.append(validator)

    def validate_args(self, args: List[IRExpression]) -> bool:
        """Validate arguments against all validators"""
        return all(validator(args) for validator in self.validators)

    def get_mapping(self, target: str) -> Optional[BuiltinMapping]:
        """Get mapping for specific target"""
        return self.mappings.get(target)


class BuiltinRegistry:
    """Registry for all built-in functions"""

    def __init__(self):
        self.functions: Dict[str, BuiltinFunction] = {}
        self.categories: Dict[str, List[str]] = {}
        self._register_standard_builtins()

    def register(self, function: BuiltinFunction):
        """Register a built-in function"""
        self.functions[function.signature.name] = function

        # Add to category
        category = function.signature.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(function.signature.name)

    def get(self, name: str) -> Optional[BuiltinFunction]:
        """Get built-in function by name"""
        return self.functions.get(name)

    def list_by_category(self, category: str) -> List[str]:
        """List all built-ins in a category"""
        return self.categories.get(category, [])

    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())

    def _register_standard_builtins(self):
        """Register standard built-in functions"""
        # Math operations
        self._register_math_builtins()

        # String operations
        self._register_string_builtins()

        # Memory operations
        self._register_memory_builtins()

        # Control flow
        self._register_control_builtins()

        # Type operations
        self._register_type_builtins()

        # Script-specific operations
        self._register_script_builtins()

    def _register_math_builtins(self):
        """Register mathematical built-in functions"""
        # abs(x) - absolute value
        abs_builtin = BuiltinFunction(BuiltinSignature(
            name="abs",
            parameters=["number"],
            return_type="number",
            side_effects=[SideEffect.NONE],
            description="Absolute value of a number",
            category="math"
        ))
        abs_builtin.add_target_mapping("generic", BuiltinMapping(
            expansion=["load_arg_0", "dup", "const_0", "cmp_slt", "if_neg", "neg", "endif"]
        ))
        abs_builtin.add_target_mapping("x86", BuiltinMapping(direct_opcode="abs"))
        self.register(abs_builtin)

        # pow(x, y) - power function
        pow_builtin = BuiltinFunction(BuiltinSignature(
            name="pow",
            parameters=["number", "number"],
            return_type="number",
            side_effects=[SideEffect.NONE],
            description="Raise x to the power of y",
            category="math"
        ))
        pow_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="__pow_f64"
        ))
        pow_builtin.add_target_mapping("x86", BuiltinMapping(
            expansion=["fld", "fyl2x", "f2xm1", "fld1", "fadd", "fscale"]
        ))
        self.register(pow_builtin)

        # sqrt(x) - square root
        sqrt_builtin = BuiltinFunction(BuiltinSignature(
            name="sqrt",
            parameters=["number"],
            return_type="number",
            side_effects=[SideEffect.NONE],
            description="Square root of a number",
            category="math"
        ))
        sqrt_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="__sqrt_f64"
        ))
        sqrt_builtin.add_target_mapping("x86", BuiltinMapping(direct_opcode="fsqrt"))
        self.register(sqrt_builtin)

    def _register_string_builtins(self):
        """Register string manipulation built-in functions"""
        # strlen(s) - string length
        strlen_builtin = BuiltinFunction(BuiltinSignature(
            name="strlen",
            parameters=["string"],
            return_type="number",
            side_effects=[SideEffect.MEMORY_READ],
            description="Get length of a string",
            category="string"
        ))
        strlen_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="strlen"
        ))
        self.register(strlen_builtin)

        # strcmp(s1, s2) - string comparison
        strcmp_builtin = BuiltinFunction(BuiltinSignature(
            name="strcmp",
            parameters=["string", "string"],
            return_type="number",
            side_effects=[SideEffect.MEMORY_READ],
            description="Compare two strings",
            category="string"
        ))
        strcmp_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="strcmp"
        ))
        self.register(strcmp_builtin)

        # strcat(dest, src) - string concatenation
        strcat_builtin = BuiltinFunction(BuiltinSignature(
            name="strcat",
            parameters=["string", "string"],
            return_type="string",
            side_effects=[SideEffect.MEMORY_WRITE],
            description="Concatenate two strings",
            category="string"
        ))
        strcat_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="strcat"
        ))
        self.register(strcat_builtin)

    def _register_memory_builtins(self):
        """Register memory operation built-in functions"""
        # memcpy(dest, src, size) - memory copy
        memcpy_builtin = BuiltinFunction(BuiltinSignature(
            name="memcpy",
            parameters=["pointer", "pointer", "number"],
            return_type="pointer",
            side_effects=[SideEffect.MEMORY_READ, SideEffect.MEMORY_WRITE],
            description="Copy memory block",
            category="memory"
        ))
        memcpy_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="memcpy"
        ))
        memcpy_builtin.add_target_mapping("x86", BuiltinMapping(
            expansion=["mov_esi_src", "mov_edi_dest", "mov_ecx_size", "rep_movsb"]
        ))
        self.register(memcpy_builtin)

        # memset(dest, value, size) - memory fill
        memset_builtin = BuiltinFunction(BuiltinSignature(
            name="memset",
            parameters=["pointer", "number", "number"],
            return_type="pointer",
            side_effects=[SideEffect.MEMORY_WRITE],
            description="Fill memory block with value",
            category="memory"
        ))
        memset_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="memset"
        ))
        self.register(memset_builtin)

    def _register_control_builtins(self):
        """Register control flow built-in functions"""
        # assert(condition) - assertion
        assert_builtin = BuiltinFunction(BuiltinSignature(
            name="assert",
            parameters=["boolean"],
            return_type=None,
            side_effects=[SideEffect.CONTROL, SideEffect.IO],
            description="Assert a condition is true",
            category="control"
        ))
        assert_builtin.add_target_mapping("generic", BuiltinMapping(
            expansion=["test_condition", "jz_assert_fail", "goto_continue",
                      "assert_fail:", "call_abort", "continue:"]
        ))
        self.register(assert_builtin)

        # unreachable() - mark unreachable code
        unreachable_builtin = BuiltinFunction(BuiltinSignature(
            name="unreachable",
            parameters=[],
            return_type=None,
            side_effects=[SideEffect.CONTROL],
            description="Mark code as unreachable",
            category="control"
        ))
        unreachable_builtin.add_target_mapping("generic", BuiltinMapping(
            expansion=["ud2"]  # Undefined instruction
        ))
        self.register(unreachable_builtin)

    def _register_type_builtins(self):
        """Register type operation built-in functions"""
        # typeof(x) - get type of value
        typeof_builtin = BuiltinFunction(BuiltinSignature(
            name="typeof",
            parameters=["any"],
            return_type="string",
            side_effects=[SideEffect.NONE],
            description="Get type of a value",
            category="type"
        ))
        typeof_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="__typeof"
        ))
        self.register(typeof_builtin)

        # is_number(x) - check if value is number
        is_number_builtin = BuiltinFunction(BuiltinSignature(
            name="is_number",
            parameters=["any"],
            return_type="boolean",
            side_effects=[SideEffect.NONE],
            description="Check if value is a number",
            category="type"
        ))
        is_number_builtin.add_target_mapping("generic", BuiltinMapping(
            expansion=["load_arg_0", "get_type_tag", "const_number_tag", "cmp_e"]
        ))
        self.register(is_number_builtin)

    def _register_script_builtins(self):
        """Register script domain specific built-in functions"""
        # debug_print(x) - debug output
        debug_print_builtin = BuiltinFunction(BuiltinSignature(
            name="debug_print",
            parameters=["any"],
            return_type=None,
            side_effects=[SideEffect.IO],
            description="Print value for debugging",
            category="script"
        ))
        debug_print_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="__debug_print"
        ))
        self.register(debug_print_builtin)

        # script_call(func_id, args...) - call script function by ID
        script_call_builtin = BuiltinFunction(BuiltinSignature(
            name="script_call",
            parameters=["number", "...any"],
            return_type="any",
            side_effects=[SideEffect.CONTROL, SideEffect.MEMORY_READ, SideEffect.MEMORY_WRITE],
            description="Call script function by ID",
            category="script"
        ))
        script_call_builtin.add_target_mapping("falcom_vm", BuiltinMapping(
            direct_opcode="CALL_FUNC"
        ))
        script_call_builtin.add_target_mapping("generic", BuiltinMapping(
            library_call="__script_call"
        ))
        self.register(script_call_builtin)


# Global registry instance
builtin_registry = BuiltinRegistry()


def register_builtin(function: BuiltinFunction):
    """Register a built-in function in the global registry"""
    builtin_registry.register(function)


def get_builtin(name: str) -> Optional[BuiltinFunction]:
    """Get built-in function by name from global registry"""
    return builtin_registry.get(name)


def list_builtins_by_category(category: str) -> List[str]:
    """List all built-ins in a category"""
    return builtin_registry.list_by_category(category)
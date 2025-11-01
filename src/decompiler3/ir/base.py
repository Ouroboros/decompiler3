"""
Base IR classes and interfaces following BinaryNinja patterns.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Dict, Optional, Union, Any, TypeVar, Generic
from dataclasses import dataclass, field
import uuid


class IRLevel(Enum):
    """IR abstraction levels"""
    LLIL = "llil"  # Low Level IL
    MLIL = "mlil"  # Medium Level IL
    HLIL = "hlil"  # High Level IL


class IRForm(Enum):
    """IR forms (normal vs SSA)"""
    NORMAL = "normal"
    SSA = "ssa"


class OperationType(Enum):
    """Operation types across all IR levels"""
    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()

    # Bitwise
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    LSL = auto()
    LSR = auto()
    ASR = auto()

    # Comparison
    CMP_E = auto()    # ==
    CMP_NE = auto()   # !=
    CMP_SLT = auto()  # signed <
    CMP_ULT = auto()  # unsigned <
    CMP_SLE = auto()  # signed <=
    CMP_ULE = auto()  # unsigned <=

    # Memory
    LOAD = auto()
    STORE = auto()

    # Control flow
    JUMP = auto()
    CALL = auto()
    RET = auto()
    IF = auto()
    GOTO = auto()

    # Variables (MLIL+)
    VAR = auto()
    VAR_FIELD = auto()

    # Constants
    CONST = auto()

    # Built-in calls
    BUILTIN_CALL = auto()

    # SSA-specific
    PHI = auto()

    # High-level constructs (HLIL)
    FOR = auto()
    WHILE = auto()
    SWITCH = auto()
    BREAK = auto()
    CONTINUE = auto()


class IRType(Enum):
    """IR type system - high-level types used across all IR levels"""
    # Basic types
    NUMBER = auto()      # Unified numeric type (maps to appropriate DataType)
    STRING = auto()      # String/text data
    BOOLEAN = auto()     # Boolean true/false
    POINTER = auto()     # Memory pointer/address
    OBJECT = auto()      # Object/struct type
    ARRAY = auto()       # Array/list type
    FUNCTION = auto()    # Function/callable type
    ANY = auto()         # Dynamic/unknown type
    VOID = auto()        # No value/void type

    # Special types
    UNDEFINED = auto()   # Uninitialized/undefined value
    NULL = auto()        # Null/nil value

    @classmethod
    def from_string(cls, type_str: str) -> 'IRType':
        """Convert string type to IRType enum (for migration compatibility)"""
        type_mapping = {
            "number": cls.NUMBER,
            "string": cls.STRING,
            "boolean": cls.BOOLEAN,
            "pointer": cls.POINTER,
            "object": cls.OBJECT,
            "array": cls.ARRAY,
            "function": cls.FUNCTION,
            "any": cls.ANY,
            "void": cls.VOID,
            "undefined": cls.UNDEFINED,
            "null": cls.NULL,
            # Additional legacy mappings
            "int8": cls.NUMBER,
            "int16": cls.NUMBER,
            "int32": cls.NUMBER,
            "int64": cls.NUMBER,
            "float32": cls.NUMBER,
            "float64": cls.NUMBER,
        }
        return type_mapping.get(type_str.lower(), cls.ANY)

    def to_string(self) -> str:
        """Convert IRType to string (for backward compatibility)"""
        string_mapping = {
            self.NUMBER: "number",
            self.STRING: "string",
            self.BOOLEAN: "boolean",
            self.POINTER: "pointer",
            self.OBJECT: "object",
            self.ARRAY: "array",
            self.FUNCTION: "function",
            self.ANY: "any",
            self.VOID: "void",
            self.UNDEFINED: "undefined",
            self.NULL: "null",
        }
        return string_mapping.get(self, "any")


@dataclass
class SourceLocation:
    """Source location information"""
    address: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    file: Optional[str] = None


class IRExpression(ABC):
    """Base class for all IR expressions"""

    def __init__(self, operation: OperationType, size: int = 4):
        self.id = str(uuid.uuid4())
        self.operation = operation
        self.size = size  # Size in bytes
        self.source_location: Optional[SourceLocation] = None
        self.operands: List['IRExpression'] = []
        self.ssa_version: Optional[int] = None

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def accept(self, visitor: 'IRVisitor') -> Any:
        pass


class IRVariable:
    """Represents a variable in IR"""

    def __init__(self, name: str, size: int = 4, var_type: Optional[IRType] = None):
        self.name = name
        self.size = size
        self.var_type = var_type or IRType.ANY
        self.ssa_version: Optional[int] = None

    def __str__(self) -> str:
        if self.ssa_version is not None:
            return f"{self.name}#{self.ssa_version}"
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, self.ssa_version))

    def __eq__(self, other) -> bool:
        return (isinstance(other, IRVariable) and
                self.name == other.name and
                self.ssa_version == other.ssa_version)


class IRBasicBlock:
    """Basic block containing IR instructions"""

    def __init__(self, address: Optional[int] = None):
        self.address = address
        self.id = str(uuid.uuid4())
        self.instructions: List[IRExpression] = []
        self.predecessors: List['IRBasicBlock'] = []
        self.successors: List['IRBasicBlock'] = []
        self.dominance_frontier: List['IRBasicBlock'] = []

    def add_instruction(self, instruction: IRExpression):
        """Add instruction to this basic block"""
        self.instructions.append(instruction)

    def __str__(self) -> str:
        result = f"block_{self.id[:8]}:\n"
        for inst in self.instructions:
            result += f"  {inst}\n"
        return result


class IRFunction:
    """Function containing basic blocks and IR instructions"""

    def __init__(self, name: str, address: Optional[int] = None):
        self.name = name
        self.address = address
        self.basic_blocks: List[IRBasicBlock] = []
        self.variables: Dict[str, IRVariable] = {}
        self.parameters: List[IRVariable] = []
        self.return_type: Optional[IRType] = None
        self.ssa_form = False

    def create_variable(self, name: str, size: int = 4, var_type: Optional[IRType] = None) -> IRVariable:
        """Create a new variable in this function"""
        var = IRVariable(name, size, var_type)
        self.variables[name] = var
        return var

    def get_basic_block(self, address: int) -> Optional[IRBasicBlock]:
        """Get basic block by address"""
        for bb in self.basic_blocks:
            if bb.address == address:
                return bb
        return None

    def add_basic_block(self, bb: IRBasicBlock):
        """Add basic block to function"""
        self.basic_blocks.append(bb)


class IRVisitor(ABC):
    """Visitor pattern for IR expressions"""

    @abstractmethod
    def visit_expression(self, expr: IRExpression) -> Any:
        pass


class IRTransformer(ABC):
    """Base class for IR transformations"""

    @abstractmethod
    def transform_function(self, function: IRFunction) -> IRFunction:
        pass

    @abstractmethod
    def transform_expression(self, expr: IRExpression) -> IRExpression:
        pass


class Architecture:
    """Target architecture description"""

    def __init__(self, name: str, bits: int = 32):
        self.name = name
        self.bits = bits
        self.endian = "little"  # little or big
        self.stack_grows_down = True
        self.pointer_size = bits // 8

        # Instruction set capabilities
        self.has_division = True
        self.has_floating_point = True
        self.has_jump_tables = True
        self.has_conditional_branches = True

        # Register/stack model
        self.is_stack_machine = False  # vs register machine
        self.register_count = 16 if not self.is_stack_machine else 0


# Type aliases for generic IR operations
T = TypeVar('T', bound=IRExpression)


class IRContext:
    """Context for IR generation and transformation"""

    def __init__(self, architecture: Architecture):
        self.architecture = architecture
        self.functions: Dict[str, IRFunction] = {}
        self.global_variables: Dict[str, IRVariable] = {}
        self.builtin_functions: Dict[str, 'BuiltinFunction'] = {}

    def create_function(self, name: str, address: Optional[int] = None) -> IRFunction:
        """Create a new function"""
        function = IRFunction(name, address)
        self.functions[name] = function
        return function
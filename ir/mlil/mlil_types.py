'''
MLIL Type System - Type definitions for type inference

Defines types that can be inferred for MLIL variables:
- Primitive types: int, bool, string, float
- Pointer types
- Unknown/variant types
'''

from typing import Optional, Set
from enum import Enum, auto


class MLILTypeKind(Enum):
    '''Type kinds for MLIL variables'''
    UNKNOWN = auto()    # Type not yet determined
    INT = auto()        # Integer (32-bit)
    BOOL = auto()       # Boolean (0 or 1)
    STRING = auto()     # String pointer
    FLOAT = auto()      # Floating point
    POINTER = auto()    # Generic pointer
    VARIANT = auto()    # Multiple possible types (type conflict)
    VOID = auto()       # No value (for functions that don't return)


class MLILType:
    '''Base class for MLIL types'''

    def __init__(self, kind: MLILTypeKind):
        self.kind = kind

    def __eq__(self, other) -> bool:
        if not isinstance(other, MLILType):
            return False
        return self.kind == other.kind

    def __hash__(self) -> int:
        return hash(self.kind)

    def __str__(self) -> str:
        return self.kind.name.lower()

    def __repr__(self) -> str:
        return f'MLILType({self.kind.name})'

    def is_unknown(self) -> bool:
        '''Check if type is unknown'''
        return self.kind == MLILTypeKind.UNKNOWN

    def is_numeric(self) -> bool:
        '''Check if type is numeric (int, float, bool)'''
        return self.kind in (MLILTypeKind.INT, MLILTypeKind.FLOAT, MLILTypeKind.BOOL)

    def is_pointer(self) -> bool:
        '''Check if type is pointer (string, pointer)'''
        return self.kind in (MLILTypeKind.STRING, MLILTypeKind.POINTER)

    @classmethod
    def unknown(cls) -> 'MLILType':
        '''Create unknown type'''
        return MLILType(MLILTypeKind.UNKNOWN)

    @classmethod
    def int_type(cls) -> 'MLILType':
        '''Create int type'''
        return MLILType(MLILTypeKind.INT)

    @classmethod
    def bool_type(cls) -> 'MLILType':
        '''Create bool type'''
        return MLILType(MLILTypeKind.BOOL)

    @classmethod
    def string_type(cls) -> 'MLILType':
        '''Create string type'''
        return MLILType(MLILTypeKind.STRING)

    @classmethod
    def float_type(cls) -> 'MLILType':
        '''Create float type'''
        return MLILType(MLILTypeKind.FLOAT)

    @classmethod
    def pointer_type(cls) -> 'MLILType':
        '''Create generic pointer type'''
        return MLILType(MLILTypeKind.POINTER)

    @classmethod
    def variant_type(cls) -> 'MLILType':
        '''Create variant type (multiple conflicting types)'''
        return MLILType(MLILTypeKind.VARIANT)

    @classmethod
    def void_type(cls) -> 'MLILType':
        '''Create void type'''
        return MLILType(MLILTypeKind.VOID)


class MLILVariantType(MLILType):
    '''Variant type representing multiple possible types (type conflict)'''

    def __init__(self, types: Set[MLILType]):
        super().__init__(MLILTypeKind.VARIANT)
        self.types = types

    def __str__(self) -> str:
        type_names = ', '.join(str(t) for t in sorted(self.types, key = lambda t: t.kind.value))
        return f'variant<{type_names}>'

    def __repr__(self) -> str:
        return f'MLILVariantType({self.types})'


def unify_types(t1: MLILType, t2: MLILType) -> MLILType:
    '''Unify two types, returning the most specific common type

    Args:
        t1: First type
        t2: Second type

    Returns:
        Unified type, or variant if incompatible

    Examples:
        unify_types(int, int) → int
        unify_types(int, unknown) → int
        unify_types(int, string) → variant<int, string>
        unify_types(bool, int) → int (bool is subset of int)
    '''
    # Same type
    if t1 == t2:
        return t1

    # Unknown type: use the other type
    if t1.is_unknown():
        return t2
    if t2.is_unknown():
        return t1

    # Bool can be promoted to int
    if t1.kind == MLILTypeKind.BOOL and t2.kind == MLILTypeKind.INT:
        return t2
    if t1.kind == MLILTypeKind.INT and t2.kind == MLILTypeKind.BOOL:
        return t1

    # Variant types: merge
    if isinstance(t1, MLILVariantType) and isinstance(t2, MLILVariantType):
        return MLILVariantType(t1.types | t2.types)
    if isinstance(t1, MLILVariantType):
        return MLILVariantType(t1.types | {t2})
    if isinstance(t2, MLILVariantType):
        return MLILVariantType({t1} | t2.types)

    # Incompatible types: create variant
    return MLILVariantType({t1, t2})


def get_operation_result_type(op_name: str, lhs_type: MLILType, rhs_type: Optional[MLILType] = None) -> MLILType:
    '''Get the result type of an operation

    Args:
        op_name: Operation name (e.g., 'add', 'eq', 'neg')
        lhs_type: Left operand type
        rhs_type: Right operand type (None for unary ops)

    Returns:
        Result type of the operation
    '''
    # Arithmetic operations: preserve numeric type
    if op_name in ('add', 'sub', 'mul', 'div', 'mod', 'neg'):
        if rhs_type is None:
            # Unary
            return lhs_type if lhs_type.is_numeric() else MLILType.int_type()
        else:
            # Binary: unify operand types
            unified = unify_types(lhs_type, rhs_type)
            return unified if unified.is_numeric() else MLILType.int_type()

    # Bitwise operations: always int
    elif op_name in ('and', 'or', 'xor', 'shl', 'shr'):
        return MLILType.int_type()

    # Comparison operations: always bool
    elif op_name in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
        return MLILType.bool_type()

    # Logical operations: bool
    elif op_name in ('logical_and', 'logical_or', 'logical_not'):
        return MLILType.bool_type()

    # Unknown operation
    else:
        return MLILType.unknown()

'''Target interface: what the IR core needs to know about a VM

A target is a class whose attributes are the interface implementations below. IR functions carry their
target (set by the lifter and passed on by every conversion), so passes and backends read it from the
function. Every base class here is the conservative default: it folds nothing and assumes nothing.
'''

from enum import Enum, auto
from typing import Any, Iterable, List, Optional


class ValueOp(Enum):
    '''Operators of the IR whose result on constants the target defines'''
    ADD         = auto()
    SUB         = auto()
    MUL         = auto()
    DIV         = auto()
    MOD         = auto()
    BIT_AND     = auto()
    BIT_OR      = auto()
    BIT_XOR     = auto()
    SHL         = auto()
    SHR         = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR  = auto()
    EQ          = auto()
    NE          = auto()
    LT          = auto()
    LE          = auto()
    GT          = auto()
    GE          = auto()
    NEG         = auto()
    BIT_NOT     = auto()
    TEST_ZERO   = auto()    # 1 when the operand is zero, else 0
    LOGICAL_NOT = auto()    # structural negation of a branch condition


COMPARISON_VALUE_OPS = frozenset({ValueOp.EQ, ValueOp.NE, ValueOp.LT, ValueOp.LE, ValueOp.GT, ValueOp.GE})


class NotFoldable(Exception):
    '''Raised by ValueSemantics when an operation on constants has no known result'''


class ValueSemantics:
    '''How the target VM computes with constants (plain numbers, strings and RawWord)'''

    @classmethod
    def binary(cls, op: ValueOp, lhs: Any, rhs: Any) -> Any:
        raise NotFoldable(op.name)

    @classmethod
    def unary(cls, op: ValueOp, operand: Any) -> Any:
        raise NotFoldable(op.name)

    @classmethod
    def is_true(cls, value: Any) -> bool:
        '''Whether a branch on this constant takes the true edge'''
        raise NotFoldable('branch')

    @classmethod
    def labels_distinct(cls, values: Iterable[Any]) -> bool:
        '''Whether no two of these constants compare equal, so they can label distinct switch cases'''
        return False

    @classmethod
    def compare_literals(cls, op: ValueOp, lhs: Any, rhs: Any) -> Optional[bool]:
        '''Result of comparing two literals when a backend may print it instead of the comparison'''
        return None


class StorageModel:
    '''Indexed storage outside the stack frame: global variables and registers

    The names are how listings and backends spell each space; register_count sizes the register file.
    '''
    global_name     = 'globals'
    register_name   = 'regs'
    register_count  = 0


class CallTarget:
    '''Callee outside the module; the target defines its kinds (subclasses must be hashable and comparable)'''

    def format_call(self, args: List[str]) -> str:
        '''Listing form of a call with already formatted arguments'''
        return f'{self}({", ".join(args)})'


class CallModel:
    '''Calling convention: return_register holds the result of every call (None: no such register)'''
    return_register: Optional[int] = None

    @classmethod
    def parameter_number(cls, frame_slot: int, param_count: int) -> int:
        '''1-based parameter held by a frame slot (slot 0 is the lowest); by default parameter 1 is pushed first'''
        return frame_slot + 1


class BackendHints:
    '''How a backend spells target-defined constructs

    generator is the backend's code generator; hints use its format_expr for operands.
    '''

    @classmethod
    def typescript_external_call(cls, target: CallTarget, args: List[Any], generator) -> str:
        return target.format_call([generator.format_expr(arg) for arg in args])

    @classmethod
    def typescript_declarations(cls) -> str:
        '''Declarations of the target's intrinsics, appended to the generic TypeScript header'''
        return ''


class ModuleMetadata:
    '''File-level metadata of a module (ILModule.metadata); fields() names the items a writer must keep'''

    @classmethod
    def fields(cls) -> List[str]:
        return []


class Target:
    '''A VM the IR can describe; subclasses replace the parts they define

    word_size is the bytes per stack slot of the bytecode; the IR itself counts stack slots.
    '''
    name        = 'generic'
    word_size: Optional[int] = None
    values      = ValueSemantics
    storage     = StorageModel
    calls       = CallModel
    hints       = BackendHints
    metadata    = ModuleMetadata

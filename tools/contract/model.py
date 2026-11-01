'''Contract items in a layer-independent form

Every field that a layer does not represent is None; the audit reports it as missing.
Values are (tag, value) pairs with tag in VALUE_TAGS, so Raw and Integer stay distinct.
'''

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple


TAG_RAW     = 'raw'
TAG_INT     = 'int'
TAG_FLOAT   = 'float'
TAG_STR     = 'str'
VALUE_TAGS  = (TAG_RAW, TAG_INT, TAG_FLOAT, TAG_STR)

# Callee of a debug record that is not a function of this script (script or syscall call)
NO_CALLEE   = None

Value = Tuple[str, Any]
# (callee function index or NO_CALLEE, call type, ((value, arg type), ...))
DebugRecord = Tuple[Optional[int], int, Tuple[Tuple[Value, int], ...]]


@dataclass
class FunctionEntry:
    name: str
    name_hash: Optional[int] = None
    is_common: Optional[bool] = None
    byte06: Optional[int] = None
    param_flags: Optional[Tuple[int, ...]] = None
    # One slot per parameter; None for a parameter without default
    defaults: Optional[Tuple[Optional[Value], ...]] = None
    debug_records: Optional[Tuple[DebugRecord, ...]] = None


@dataclass
class ModuleTables:
    dword_14: Optional[int] = None
    # (name, type) per global, in table order
    globals: Optional[Tuple[Tuple[str, Any], ...]] = None
    # Function names in table order; None in layers without a module
    function_order: Optional[Tuple[str, ...]] = None
    functions: Dict[str, FunctionEntry] = field(default_factory = dict)


@dataclass
class CodeItems:
    '''Code-level items of one function; addresses are None in layers that do not keep them'''
    lines: Set[Tuple[Optional[int], int]] = field(default_factory = set)
    logs: Set[Tuple[Optional[int], int]] = field(default_factory = set)


@dataclass
class Difference:
    layer: str
    script: str
    item: str
    function: Optional[str] = None
    # Number of differing elements (records, line markers, ...)
    count: int = 1
    detail: str = ''


ModuleCode = Dict[str, CodeItems]

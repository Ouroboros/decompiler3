'''Comparison of contract items between a reference and a layer'''

from collections import Counter
from typing import Iterable, List, Optional

from .model import CodeItems, Difference, FunctionEntry, ModuleTables


FUNCTION_FIELDS = ('name_hash', 'is_common', 'byte06', 'param_flags', 'defaults', 'debug_records')
CODE_FIELDS     = ('lines', 'logs')
EXAMPLE_LENGTH  = 160


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= EXAMPLE_LENGTH else text[:EXAMPLE_LENGTH] + '...'


def _element_count(reference, actual) -> int:
    '''Elements of the reference sequence not matched in the actual one, plus unmatched extras'''
    if not isinstance(reference, tuple) or not isinstance(actual, tuple):
        return 1

    expected, found = Counter(reference), Counter(actual)
    return sum((expected - found).values()) + sum((found - expected).values())


def compare_tables(layer: str, script: str, reference: ModuleTables, actual: ModuleTables,
                   functions: Optional[Iterable[str]] = None) -> List[Difference]:
    '''functions: names to compare (default: all of the reference)'''
    differences = []
    for item in ('dword_14', 'globals', 'function_order'):
        expected, found = getattr(reference, item), getattr(actual, item)
        if found is None:
            differences.append(Difference(layer, script, item, detail = 'missing'))

        elif found != expected:
            differences.append(Difference(layer, script, item, count = _element_count(expected, found),
                                          detail = f'expected {_short(expected)}, found {_short(found)}'))

    for name in functions if functions is not None else reference.functions:
        expected_entry = reference.functions[name]
        entry: Optional[FunctionEntry] = actual.functions.get(name)
        if entry is None:
            differences.append(Difference(layer, script, 'function', name, detail = 'missing'))
            continue

        for item in FUNCTION_FIELDS:
            expected, found = getattr(expected_entry, item), getattr(entry, item)
            if found is None:
                count = len(expected) if isinstance(expected, tuple) else 1
                if count:
                    differences.append(Difference(layer, script, f'function.{item}', name, count, 'missing'))

            elif found != expected:
                differences.append(Difference(layer, script, f'function.{item}', name, _element_count(expected, found),
                                              f'expected {_short(expected)}, found {_short(found)}'))

    return differences


def compare_code(layer: str, script: str, name: str, reference: CodeItems, actual: CodeItems,
                 with_addresses: bool = True) -> List[Difference]:
    '''Without addresses (TypeScript) only the values are compared'''
    differences = []
    for item in CODE_FIELDS:
        expected, found = getattr(reference, item), getattr(actual, item)
        if not with_addresses:
            expected = {(None, value) for _, value in expected}

        missing, extra = expected - found, found - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f'missing {_short(sorted(missing, key = str))}')

            if extra:
                details.append(f'extra {_short(sorted(extra, key = str))}')

            differences.append(Difference(layer, script, f'code.{item}', name, len(missing) + len(extra), '; '.join(details)))

    return differences

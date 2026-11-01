'''Trace forms of VM values; the value semantics themselves live in falcom/ed9/vm_values.py'''

from dataclasses import dataclass
from typing import Any, Callable

from falcom.ed9.vm_values import UNDEF, Pointer, Raw, bits_to_float, float32_bits, normalize


@dataclass(frozen = True)
class ObservedFloat:
    '''Float as compared in traces: by bit pattern, so -0.0 and NaN compare exactly'''
    bits: int

    def __repr__(self) -> str:
        return f'{bits_to_float(self.bits)!r}f'


OBSERVED_RAW = 'raw'


def observable(value: Any, deref: Callable[[Pointer], Any], depth: int = 0, *, strict_raw: bool = False) -> Any:
    '''Hashable, representation-independent form of a value for trace comparison

    Raw words are observed by payload, like integers; with strict_raw they are observed with their tag,
    so a decompiler that turns Raw into Integer is caught.
    '''
    if value is UNDEF:
        return ('undef',)

    if isinstance(value, Pointer):
        # Pointer identity differs between interpreters; compare the referenced value instead
        inner = observable(deref(value), deref, depth + 1, strict_raw = strict_raw) if depth == 0 else ('nested',)
        return ('ptr', inner)

    if isinstance(value, Raw):
        payload = normalize(value).payload
        return (OBSERVED_RAW, payload) if strict_raw else payload

    if isinstance(value, float):
        return ObservedFloat(float32_bits(normalize(value)))

    if isinstance(value, (bool, int)):
        return normalize(value)

    if isinstance(value, str):
        return value

    return ('opaque', type(value).__name__)

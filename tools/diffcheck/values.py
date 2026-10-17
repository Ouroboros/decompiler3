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


def observable(value: Any, deref: Callable[[Pointer], Any], depth: int = 0) -> Any:
    '''Hashable, representation-independent form of a value for trace comparison

    Raw words are observed by payload: script output cannot distinguish them from integers.
    '''
    if value is UNDEF:
        return ('undef',)

    if isinstance(value, Pointer):
        # Pointer identity differs between interpreters; compare the referenced value instead
        return ('ptr', observable(deref(value), deref, depth + 1) if depth == 0 else ('nested',))

    if isinstance(value, Raw):
        return normalize(value).payload

    if isinstance(value, float):
        return ObservedFloat(float32_bits(normalize(value)))

    if isinstance(value, (bool, int)):
        return normalize(value)

    if isinstance(value, str):
        return value

    return ('opaque', type(value).__name__)

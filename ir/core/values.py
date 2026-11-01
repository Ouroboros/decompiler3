'''Constant values the IR carries besides plain Python numbers and strings'''

from dataclasses import dataclass


@dataclass(frozen = True)
class RawWord:
    '''Constant word without a type tag; the target's value semantics decide how operations read it'''
    payload: int

    def __str__(self) -> str:
        return f'raw({self.payload})'

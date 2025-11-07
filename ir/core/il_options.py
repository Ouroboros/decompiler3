'''
IL Options - Generic options for all IL levels

This module provides option classes that can be used across all IL levels
(LLIL, MLIL, HLIL, etc.) to control formatting, processing, and other
metadata without affecting IR semantics.
'''

class StrictBase:
    _allowed_attrs_: frozenset[str]

    def __init_subclass__(cls) -> None:
        ann = getattr(cls, '__annotations__', {})
        cls._allowed_attrs_ = frozenset(ann.keys())

    def __setattr__(self, name, value):
        if name not in self._allowed_attrs_:
            raise AttributeError(f"Unknown attribute {name!r}")

        return object.__setattr__(self, name, value)

class ILOptions(StrictBase):
    '''
        Attributes:
        hidden_for_formatter: If True, formatter skips this instruction
    '''

    hidden_for_formatter: bool

    def __init__(self):
        self.hidden_for_formatter = False

'''
Strict base class that prevents dynamic attribute assignment
'''


class StrictBase:
    '''Base class that only allows annotated attributes

    Prevents accidental typos when setting attributes.
    Subclasses must use type annotations to declare allowed attributes.
    '''
    _allowed_attrs_: frozenset[str]

    def __init_subclass__(cls):
        ann = getattr(cls, '__annotations__', {})
        cls._allowed_attrs_ = frozenset(ann.keys())

    def __setattr__(self, name, value):
        if name not in self._allowed_attrs_:
            raise AttributeError(f"Unknown attribute {name!r}")

        return object.__setattr__(self, name, value)

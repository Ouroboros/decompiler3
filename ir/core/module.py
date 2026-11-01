'''Generic module container: the functions of one module (a script file) at one IL level'''

from typing import Any, Dict, Iterator


class ILModule:
    '''Functions of one module at one IL level, with the module's file-level metadata

    The metadata is defined by the target and opaque to the core; every layer derived from a module
    carries the same metadata object, so file-level information reaches each layer unchanged.
    '''

    def __init__(self, name: str, metadata: Any = None):
        self.name = name
        self.metadata = metadata
        # Function name -> IL function, in insertion order
        self.functions: Dict[str, Any] = {}

    def derive(self) -> 'ILModule':
        '''Empty module of another layer with the same name and metadata'''
        return ILModule(self.name, self.metadata)

    def add_function(self, func) -> Any:
        self.functions[func.name] = func
        return func

    def remove_function(self, name: str):
        self.functions.pop(name, None)

    def __getitem__(self, name: str) -> Any:
        return self.functions[name]

    def __contains__(self, name: str) -> bool:
        return name in self.functions

    def __iter__(self) -> Iterator[Any]:
        return iter(self.functions.values())

    def __len__(self) -> int:
        return len(self.functions)

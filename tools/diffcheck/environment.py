'''External environment and trace recording shared by both interpreters

Both interpreters see the same world: parameter values, initial GLOBALS, and the
outcome of every external call are pure functions of (seed, origin). Origins are
keyed by meaning (parameter number, call sequence number), never by query order,
so a divergence in one interpreter cannot shift the inputs of the other.
'''

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .values import Pointer, VMFault, normalize, observable


ORIGIN_PARAM            = 'param'
ORIGIN_RETURN           = 'ret'
ORIGIN_GLOBAL           = 'global'
ORIGIN_OUT              = 'out'
ORIGIN_GLOBAL_MUTATION  = 'mutation'
ORIGIN_CELL             = 'cell'

EVENT_CALL              = 'call'
EVENT_SYSCALL           = 'syscall'
EVENT_EXTERN            = 'extern'
EVENT_RETURN            = 'return'
EVENT_FAULT             = 'fault'

DEFAULT_POOL            = (0, 1, -1, 2)
HINT_PROBABILITY        = 0.75
MUTATION_PROBABILITY    = 0.25
ENTRY_REG_VALUE         = 0
MISSING_GLOBAL_VALUE    = 0

MAX_EVENTS              = 1024
MAX_STEPS               = 2_000_000


Origin = Tuple[Any, ...]


@dataclass
class CallOutcome:
    '''Effects of one external call, applied identically by both interpreters'''
    call_index: int
    return_value: Any
    out_values: Dict[int, Any]
    global_writes: Dict[int, Any]


class InputPlan:
    '''Deterministic inputs for one run'''

    def __init__(self, seed: Any, pool: Iterable[Any], hints: Dict[Origin, List[Any]],
                 global_candidates: Iterable[int], param_count: int, forced: Optional[Dict[Origin, Any]] = None,
                 global_count: Optional[int] = None):
        self.seed = seed
        self.pool = list(pool) or list(DEFAULT_POOL)
        self.hints = hints
        self.global_count = global_count
        self.global_candidates = sorted(index for index in set(global_candidates) if self.global_exists(index))
        self.param_count = param_count
        self.forced = dict(forced or {})
        self._values: Dict[Origin, Any] = {}

    def global_exists(self, index: int) -> bool:
        '''The VM reads integer 0 from and ignores writes to indices beyond the script's global table'''
        return self.global_count is None or index < self.global_count

    @property
    def values(self) -> Dict[Origin, Any]:
        '''Every input value drawn so far in this run'''
        return dict(self._values)

    def _rng(self, *parts: Any) -> random.Random:
        # String seeds are hashed with SHA-512, stable across processes
        return random.Random(':'.join(str(part) for part in (self.seed,) + parts))

    def value_for(self, origin: Origin) -> Any:
        if origin not in self._values:
            rng = self._rng(*origin)
            hinted = self.hints.get(origin)
            if origin in self.forced:
                value = self.forced[origin]

            elif hinted and rng.random() < HINT_PROBABILITY:
                value = rng.choice(hinted)

            else:
                value = rng.choice(self.pool)

            self._values[origin] = normalize(value)

        return self._values[origin]

    def param(self, number: int) -> Any:
        return self.value_for((ORIGIN_PARAM, number))

    def initial_global(self, index: int) -> Any:
        return self.value_for((ORIGIN_GLOBAL, index))

    def call_outcome(self, call_index: int, pointer_positions: Iterable[int]) -> CallOutcome:
        out_values = {pos: self.value_for((ORIGIN_OUT, call_index, pos)) for pos in pointer_positions}
        global_writes: Dict[int, Any] = {}
        rng = self._rng(ORIGIN_GLOBAL_MUTATION, call_index)
        if self.global_candidates and rng.random() < MUTATION_PROBABILITY:
            index = rng.choice(self.global_candidates)
            global_writes[index] = self.value_for((ORIGIN_GLOBAL_MUTATION, call_index, index))

        return CallOutcome(call_index, self.value_for((ORIGIN_RETURN, call_index)), out_values, global_writes)

    def describe(self) -> Dict[str, Any]:
        return {str(origin): value for origin, value in sorted(self._values.items(), key = lambda item: str(item[0]))}


class RunStatus(Enum):
    COMPLETE    = 'complete'
    TRUNCATED   = 'truncated'
    STEP_LIMIT  = 'step_limit'
    FAULT       = 'fault'


@dataclass
class Event:
    '''One observable action; key takes part in comparison, address is for reporting'''
    key: Tuple[Any, ...]
    address: int = 0

    def __str__(self) -> str:
        return f'{self.key} @ 0x{self.address:X}'


class _RunEnd(Exception):
    def __init__(self, status: RunStatus):
        super().__init__(status.value)
        self.status = status


@dataclass
class Trace:
    events: List[Event] = field(default_factory = list)
    status: RunStatus = RunStatus.COMPLETE
    max_events: int = MAX_EVENTS

    def emit(self, key: Tuple[Any, ...], address: int):
        if len(self.events) >= self.max_events:
            raise _RunEnd(RunStatus.TRUNCATED)

        self.events.append(Event(key, address))


class WorldState:
    '''REGS/GLOBALS storage plus external-call bookkeeping, owned by one interpreter'''

    def __init__(self, plan: InputPlan, trace: Trace):
        self.plan = plan
        self.trace = trace
        self.regs: Dict[int, Any] = {0: ENTRY_REG_VALUE}
        self.globals: Dict[int, Any] = {}
        # Origin tuple of an input value, or whatever provenance the interpreter attached when storing
        self.global_origins: Dict[int, Any] = {}
        self.written_globals: set = set()
        self.cells: Dict[int, Any] = {}
        self.written_cells: set = set()
        self.call_index = 0

    def load_global(self, index: int) -> Any:
        if not self.plan.global_exists(index):
            return MISSING_GLOBAL_VALUE

        if index not in self.globals:
            self.globals[index] = self.plan.initial_global(index)
            self.global_origins[index] = (ORIGIN_GLOBAL, index)

        return self.globals[index]

    def store_global(self, index: int, value: Any, origin: Any = None):
        if not self.plan.global_exists(index):
            return

        self.globals[index] = value
        self.global_origins[index] = origin
        self.written_globals.add(index)

    def load_cell(self, param_number: int) -> Any:
        '''Caller memory addressed by a pointer parameter'''
        if param_number not in self.cells:
            self.cells[param_number] = self.plan.value_for((ORIGIN_CELL, param_number))

        return self.cells[param_number]

    def store_cell(self, param_number: int, value: Any):
        self.cells[param_number] = value
        self.written_cells.add(param_number)

    def globals_snapshot(self, deref) -> Tuple[Tuple[int, Any], ...]:
        return tuple((index, observable(self.globals[index], deref)) for index in sorted(self.written_globals))

    def external_call(self, key_prefix: Tuple[Any, ...], args: List[Any], deref, address: int) -> CallOutcome:
        '''Record a call event and return the environment's response'''
        observed_args = tuple(observable(arg, deref) for arg in args)
        self.trace.emit(key_prefix + (observed_args, self.globals_snapshot(deref)), address)
        pointer_positions = [pos for pos, arg in enumerate(args) if isinstance(arg, Pointer)]
        outcome = self.plan.call_outcome(self.call_index, pointer_positions)
        self.call_index += 1
        for index, value in outcome.global_writes.items():
            self.store_global(index, value, (ORIGIN_GLOBAL_MUTATION, outcome.call_index, index))

        self.regs[0] = outcome.return_value
        return outcome

    def finish(self, return_value: Any, deref, address: int):
        # Writes through pointer parameters are visible to the caller after return
        cells = tuple((number, observable(self.cells[number], deref)) for number in sorted(self.written_cells))
        self.trace.emit((EVENT_RETURN, observable(return_value, deref), self.globals_snapshot(deref), cells), address)

    def fault(self, fault: VMFault, address: int):
        self.trace.events.append(Event((EVENT_FAULT, fault.kind), address))
        self.trace.status = RunStatus.FAULT

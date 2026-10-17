'''Per-function differential check: bytecode reference vs decompiled HLIL'''

import contextlib
import io
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from falcom.ed9.hlil_converter import convert_falcom_mlil_to_hlil
from falcom.ed9.lifters.vm_lifter import ED9VMLifter
from falcom.ed9.mlil_converter import convert_falcom_llil_to_mlil
from ir.hlil import (
    CommonReturnExtractionPass, ControlFlowOptimizationPass, DeadCodeEliminationPass,
    ExpressionSimplificationPass, MLILToHLILPass,
)

from .bytecode_interp import (
    BytecodeInterpreter, ParamOrder, collect_blocks, collect_constants, collect_global_indices,
)
from .environment import DEFAULT_POOL, EVENT_FAULT, MAX_EVENTS, ORIGIN_RETURN, InputPlan, Origin, RunStatus, Trace
from .hlil_interp import HLILInterpreter
from .symbolic import flip


MIN_RUNS                = 8
MAX_RUNS                = 2048
PATIENCE                = 64
# Pending follow-ups do not keep the search alive forever once coverage has stopped growing
STALE_LIMIT             = 1024
POOL_LIMIT              = 96
NEIGHBOR_DELTA          = 1
MAX_ATTEMPTS_PER_HINT   = 8
MAX_QUEUE               = 4096
MAX_SOLVE_ATTEMPTS      = 4
DEFAULT_MAX_MISMATCHES  = 1

# Long straight-line scripts exceed the default event cap; the cap grows only while truncated runs
# still reach new blocks, so input-independent endless loops stay cheap
MAX_EVENT_CAP           = 16384
EVENT_CAP_GROWTH        = 2

# Faults that mean the reference interpreter cannot model the function
REFERENCE_LIMITATIONS = frozenset({
    'unsupported_opcode', 'deref_non_pointer', 'bad_call_frame', 'stack_underflow',
    'stack_out_of_range', 'missing_fallthrough', 'bad_branch_target',
})

# Mirrors falcom/ed9/hlil_converter.py; used only to localize a divergence to one pass
HLIL_STAGES = (
    ('MLILToHLIL', MLILToHLILPass),
    ('ExpressionSimplification', ExpressionSimplificationPass),
    ('ControlFlowOptimization', ControlFlowOptimizationPass),
    ('CommonReturnExtraction', CommonReturnExtractionPass),
    ('DeadCodeElimination', DeadCodeEliminationPass),
)


class Verdict(Enum):
    PASS    = 'PASS'
    FAIL    = 'FAIL'
    ERROR   = 'ERROR'


@dataclass
class CheckOptions:
    seed: int = 0
    max_runs: int = MAX_RUNS
    param_order: ParamOrder = ParamOrder.VM
    localize: bool = True
    max_mismatches: int = DEFAULT_MAX_MISMATCHES


@dataclass
class Mismatch:
    run: int
    index: int
    bytecode_event: str
    hlil_event: str
    bytecode_status: str
    hlil_status: str
    inputs: Dict[str, Any]
    stage: Optional[str] = None
    event_cap: int = 0


@dataclass
class FunctionReport:
    name: str
    address: int
    verdict: Verdict
    runs: int = 0
    blocks_covered: int = 0
    blocks_total: int = 0
    uncovered_blocks: List[str] = field(default_factory = list)
    mismatches: List[Mismatch] = field(default_factory = list)
    error: Optional[str] = None

    @property
    def coverage(self) -> float:
        return self.blocks_covered / self.blocks_total if self.blocks_total else 1.0


def first_divergence(reference: Trace, candidate: Trace) -> Optional[int]:
    '''Index of the first differing event, or None when the traces agree'''
    common = min(len(reference.events), len(candidate.events))
    for index in range(common):
        if reference.events[index].key != candidate.events[index].key:
            return index

    if reference.status == RunStatus.STEP_LIMIT:
        # The reference never finished; only the observed prefix is meaningful
        return None

    if len(reference.events) != len(candidate.events) or reference.status != candidate.status:
        return common

    return None


def _event_text(trace: Trace, index: int) -> str:
    return str(trace.events[index]) if index < len(trace.events) else f'<end: {trace.status.value}>'


def _event_address(trace: Trace, index: int) -> int:
    return trace.events[index].address if index < len(trace.events) else -1


def build_pool(constants: Set[Any]) -> List[Any]:
    '''Default input candidates: integer constants of the function, their neighbors and a few defaults

    Floats are not drawn by default: call results and flags are integers in practice, and feeding them
    floats makes the VM's integer-only operators (EZ, NOT, NEG) diverge on impossible inputs. A float
    becomes a candidate for an input once the function compares that input with a float literal.
    '''
    values: Set[Any] = set(DEFAULT_POOL)
    for constant in constants:
        if isinstance(constant, int):
            values.update({constant - NEIGHBOR_DELTA, constant, constant + NEIGHBOR_DELTA})

    pool = sorted(values)
    if len(pool) > POOL_LIMIT:
        step = len(pool) / POOL_LIMIT
        pool = [pool[int(i * step)] for i in range(POOL_LIMIT)]

    return pool


def decompile(parser, func):
    '''Run the production pipeline, silencing its progress output'''
    with contextlib.redirect_stdout(io.StringIO()):
        llil_func = ED9VMLifter(parser = parser).lift_function(func)
        mlil_func = convert_falcom_llil_to_mlil(llil_func, parser)
        return convert_falcom_mlil_to_hlil(mlil_func, func)


class FunctionChecker:
    def __init__(self, parser, func, options: CheckOptions):
        self.parser = parser
        self.func = func
        self.options = options
        self.blocks = collect_blocks(func.entry_block)
        self.pool = build_pool(collect_constants(self.blocks))
        self.global_candidates = collect_global_indices(self.blocks)
        self.param_count = len(func.params)
        self.hints: Dict[Origin, List[Any]] = {}
        self.event_cap = MAX_EVENTS

    def _plan(self, run: int, forced: Dict[Origin, Any]) -> InputPlan:
        return InputPlan(f'{self.options.seed}/{run}', self.pool, self.hints, self.global_candidates,
                         self.param_count, forced, len(self.parser.global_vars))

    def _reference(self, plan: InputPlan):
        return BytecodeInterpreter(self.parser, self.func, plan, param_order = self.options.param_order,
                                   blocks = self.blocks, max_events = self.event_cap).run()

    def _candidate(self, hlil_func, plan: InputPlan) -> Trace:
        return HLILInterpreter(hlil_func, self.param_count, plan, max_events = self.event_cap).run()

    @classmethod
    def _context(cls, origin: Origin, trace: Trace) -> Any:
        '''A call return is identified by the call that produced it, not only by its sequence number'''
        if origin[0] == ORIGIN_RETURN and origin[1] < len(trace.events):
            return trace.events[origin[1]].key

        return None

    def _learn(self, reference, plan: InputPlan, queue: deque, attempts: Counter):
        '''Queue one follow-up run per newly compared constant, replaying this run's other inputs'''
        drawn = plan.values
        for origin, values in reference.hints.items():
            merged = set(self.hints.get(origin, [])) | values
            self.hints[origin] = sorted(merged, key = lambda value: (type(value).__name__, value))
            context = self._context(origin, reference.trace)
            for value in sorted(values, key = lambda value: (type(value).__name__, value)):
                key = (origin, value, context)
                if drawn.get(origin) == value or attempts[key] >= MAX_ATTEMPTS_PER_HINT or len(queue) >= MAX_QUEUE:
                    continue

                attempts[key] += 1
                follow_up = dict(drawn)
                follow_up[origin] = value
                queue.append(follow_up)

    def _candidates(self, origin: Origin) -> List[Any]:
        return list(dict.fromkeys(list(self.hints.get(origin, [])) + list(DEFAULT_POOL)))

    def _solve_branches(self, reference, plan: InputPlan, queue: deque,
                        covered_edges: Set[Tuple[int, int]], solve_attempts: Counter):
        '''Queue runs that send recorded branches toward their still uncovered side'''
        drawn = plan.values
        for index, branch in enumerate(reference.branches):
            edge = (branch.block, branch.other)
            if edge in covered_edges or solve_attempts[edge] >= MAX_SOLVE_ATTEMPTS:
                continue

            solve_attempts[edge] += 1
            candidates = {origin: self._candidates(origin) for origin in branch.condition.inputs}
            path = [(earlier.condition, earlier.observed_zero) for earlier in reference.branches[:index]]
            assignment = flip(branch.condition, branch.observed_zero, drawn, candidates, path)
            if assignment:
                follow_up = dict(drawn)
                follow_up.update(assignment)
                queue.appendleft(follow_up)

    def check(self) -> FunctionReport:
        report = FunctionReport(self.func.name, self.func.offset, Verdict.PASS, blocks_total = len(self.blocks))
        try:
            hlil_func = decompile(self.parser, self.func)

        except Exception as e:
            report.verdict = Verdict.ERROR
            report.error = f'decompile: {type(e).__name__}: {e}'
            return report

        covered_blocks: Set[int] = set()
        covered_edges: Set[Tuple[int, int]] = set()
        queue: deque = deque()
        attempts: Counter = Counter()
        forced: Dict[Origin, Any] = {}
        seen_sites: Set[Tuple[int, int]] = set()
        solve_attempts: Counter = Counter()
        stale = 0
        for run in range(self.options.max_runs):
            plan = self._plan(run, forced)
            reference = self._reference(plan)
            fault = reference.trace.events[-1].key if reference.trace.status == RunStatus.FAULT else None
            if fault is not None and fault[0] == EVENT_FAULT and fault[1] in REFERENCE_LIMITATIONS:
                report.verdict = Verdict.ERROR
                report.error = f'reference interpreter: {reference.trace.events[-1]}'
                break

            candidate = self._candidate(hlil_func, plan)
            report.runs = run + 1
            new_edges = reference.covered_edges - covered_edges
            new_blocks = reference.covered_blocks - covered_blocks
            covered_edges |= reference.covered_edges
            covered_blocks |= reference.covered_blocks
            index = first_divergence(reference.trace, candidate)
            if index is not None:
                report.verdict = Verdict.FAIL
                site = (_event_address(reference.trace, index), _event_address(candidate, index))
                if site not in seen_sites:
                    seen_sites.add(site)
                    mismatch = Mismatch(
                        run, index, _event_text(reference.trace, index), _event_text(candidate, index),
                        reference.trace.status.value, candidate.status.value, plan.describe(),
                    )
                    if self.options.localize:
                        mismatch.stage = self._localize(run, forced, reference.trace)

                    mismatch.event_cap = self.event_cap

                    report.mismatches.append(mismatch)
                    if len(report.mismatches) >= self.options.max_mismatches:
                        break

            self._learn(reference, plan, queue, attempts)
            self._solve_branches(reference, plan, queue, covered_edges, solve_attempts)
            if reference.trace.status == RunStatus.TRUNCATED and new_blocks and self.event_cap < MAX_EVENT_CAP:
                self.event_cap = min(self.event_cap * EVENT_CAP_GROWTH, MAX_EVENT_CAP)
                queue.appendleft(dict(plan.values))

            stale = 0 if new_edges else stale + 1
            if run + 1 >= MIN_RUNS:
                if len(covered_blocks) == len(self.blocks):
                    break

                if not queue and stale >= PATIENCE:
                    break

                if stale >= STALE_LIMIT:
                    break

            forced = queue.popleft() if queue else {}

        report.blocks_covered = len(covered_blocks)
        report.uncovered_blocks = [f'0x{offset:X}' for offset in sorted(set(self.blocks) - covered_blocks)]
        return report

    def _localize(self, run: int, forced: Dict[Origin, Any], reference: Trace) -> str:
        '''Name the first HLIL stage whose output already diverges for the failing input'''
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                llil_func = ED9VMLifter(parser = self.parser).lift_function(self.func)
                data = convert_falcom_llil_to_mlil(llil_func, self.parser)
                for stage_name, stage_class in HLIL_STAGES:
                    data = stage_class().run(data)
                    candidate = self._candidate(data, self._plan(run, forced))
                    if first_divergence(reference, candidate) is not None:
                        return stage_name if stage_name != HLIL_STAGES[0][0] else f'{stage_name} (or MLIL)'

        except Exception as e:
            return f'localize failed: {type(e).__name__}: {e}'

        return 'not reproduced stage by stage'


def check_functions(parser, functions, options: CheckOptions, progress = None) -> List[FunctionReport]:
    reports = []
    for func in functions:
        try:
            report = FunctionChecker(parser, func, options).check()

        except Exception as e:
            report = FunctionReport(func.name, func.offset, Verdict.ERROR, error = f'checker: {type(e).__name__}: {e}\n{traceback.format_exc(limit = 3)}')

        reports.append(report)
        if progress is not None:
            progress(report)

    return reports

'''Command line entry: python -m tools.diffcheck SCP_FILE [...]'''

import argparse
import contextlib
import io
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ml import fileio
from common.config import default_encoding
from falcom.ed9.parser.scp import ScpParser

from .check import (
    DEFAULT_MAX_MISMATCHES, MAX_RUNS, CheckOptions, FunctionReport, Verdict, check_functions, decompile_function,
)


# Structuring and SSA passes recurse once per nesting level
RECURSION_LIMIT = 20000
PERCENT         = 100
PROGRESS_EVERY  = 100
DUMP_SUFFIX     = '.json'


def _log(message: str):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', file = sys.stderr, flush = True)


def _load_baseline(directory: Path, path: Path) -> dict:
    '''Dump of this script written by an earlier --dump run; empty when there is none'''
    file = directory / f'{path.stem}{DUMP_SUFFIX}'
    if not file.exists():
        return {}

    return json.loads(file.read_text(encoding = 'utf-8'))


def _report_to_dict(report: FunctionReport) -> dict:
    data = asdict(report)
    data['verdict'] = report.verdict.value
    data['coverage'] = round(report.coverage, 4)
    return data


def _print_report(report: FunctionReport):
    line = f'  {report.verdict.value:<5} {report.name} @ 0x{report.address:X}  runs={report.runs}  blocks={report.blocks_covered}/{report.blocks_total}'
    print(line)
    for mismatch in report.mismatches:
        print(f'        run {mismatch.run}, event #{mismatch.index}, stage: {mismatch.stage}')
        print(f'        bytecode: {mismatch.bytecode_event}  [{mismatch.bytecode_status}]')
        print(f'        hlil:     {mismatch.hlil_event}  [{mismatch.hlil_status}]')

    if report.error is not None:
        print(f'        {report.error.strip()}')


def check_file(path: Path, names, options: CheckOptions, verbose: bool,
               dump_dir: Optional[Path] = None, baseline_dir: Optional[Path] = None):
    started = time.time()
    with fileio.FileStream(str(path), encoding = default_encoding()) as fs:
        parser = ScpParser(fs, path.name)
        with contextlib.redirect_stdout(io.StringIO()):
            parser.parse()
            functions = parser.disasm_all_functions(filter_func = (lambda f: f.name in names) if names else None)

    _log(f'{path.name}: {len(functions)} functions disassembled in {time.time() - started:.1f}s')

    decompiled = {}
    if dump_dir is not None or baseline_dir is not None:
        for index, func in enumerate(functions, start = 1):
            decompiled[func.name] = decompile_function(parser, func)
            if index % PROGRESS_EVERY == 0:
                _log(f'{path.name}: decompiled {index}/{len(functions)} ({func.name})')

        _log(f'{path.name}: decompiled {len(functions)} functions in {time.time() - started:.1f}s')

    if dump_dir is not None:
        dump_dir.mkdir(parents = True, exist_ok = True)
        texts = {name: output.texts() for name, output in decompiled.items()}
        (dump_dir / f'{path.stem}{DUMP_SUFFIX}').write_text(json.dumps(texts, ensure_ascii = False), encoding = 'utf-8')

    unchanged = 0
    if baseline_dir is not None:
        baseline = _load_baseline(baseline_dir, path)
        changed = [func for func in functions if baseline.get(func.name) != decompiled[func.name].texts()]
        unchanged = len(functions) - len(changed)
        functions = changed

    checked = 0

    def progress(report: FunctionReport):
        nonlocal checked
        checked += 1
        if verbose or report.verdict != Verdict.PASS:
            _print_report(report)

        if checked % PROGRESS_EVERY == 0:
            _log(f'{path.name}: checked {checked}/{len(functions)} ({report.name}), {time.time() - started:.0f}s')

    skipped = f', {unchanged} unchanged since baseline skipped' if baseline_dir is not None else ''
    print(f'== {path} ({len(functions)} functions{skipped})', flush = True)
    return check_functions(parser, functions, options, progress, decompiled)


def main(argv = None) -> int:
    sys.setrecursionlimit(RECURSION_LIMIT)
    arg_parser = argparse.ArgumentParser(prog = 'python -m tools.diffcheck', description = 'Differential check: bytecode vs decompiled HLIL')
    arg_parser.add_argument('scp_files', nargs = '+', type = Path)
    arg_parser.add_argument('--function', action = 'append', default = [], help = 'check only this function (repeatable)')
    arg_parser.add_argument('--seed', type = int, default = 0)
    arg_parser.add_argument('--max-runs', type = int, default = MAX_RUNS)
    arg_parser.add_argument('--max-mismatches', type = int, default = DEFAULT_MAX_MISMATCHES,
                            help = 'distinct divergence sites to collect per function before stopping')
    arg_parser.add_argument('--no-localize', action = 'store_true', help = 'skip per-pass localization of failures')
    arg_parser.add_argument('--json', type = Path, help = 'write full results to this file')
    arg_parser.add_argument('--verbose', action = 'store_true', help = 'print passing functions too')
    arg_parser.add_argument('--strict-raw', action = 'store_true',
                            help = 'observe Raw words with their tag, so Raw turned into Integer is a mismatch')
    arg_parser.add_argument('--dump', type = Path, metavar = 'DIR',
                            help = 'write the HLIL and TypeScript of every function to DIR/<script>.json')
    arg_parser.add_argument('--changed-since', type = Path, metavar = 'DIR',
                            help = 'check only functions whose output differs from the dump in DIR')
    args = arg_parser.parse_args(argv)

    options = CheckOptions(
        seed = args.seed,
        max_runs = args.max_runs,
        localize = not args.no_localize,
        max_mismatches = args.max_mismatches,
        strict_raw = args.strict_raw,
    )

    started = time.time()
    results = {}
    for path in args.scp_files:
        results[str(path)] = check_file(path, set(args.function), options, args.verbose, args.dump, args.changed_since)

    print()
    for path, reports in results.items():
        counts = Counter(report.verdict for report in reports)
        stages = Counter(mismatch.stage for report in reports for mismatch in report.mismatches)
        passed = [report for report in reports if report.verdict == Verdict.PASS]
        coverage = sum(report.coverage for report in passed) / len(passed) if passed else 0.0
        print(f'{path}: PASS {counts[Verdict.PASS]}  FAIL {counts[Verdict.FAIL]}  ERROR {counts[Verdict.ERROR]}  '
              f'mean block coverage of passing functions {coverage * PERCENT:.1f}%')
        for stage, count in stages.most_common():
            print(f'    first divergent stage {stage}: {count}')

    print(f'elapsed {time.time() - started:.1f}s')

    if args.json is not None:
        payload = {path: [_report_to_dict(report) for report in reports] for path, reports in results.items()}
        args.json.write_text(json.dumps(payload, ensure_ascii = False, indent = 1, default = str), encoding = 'utf-8')

    failed = any(report.verdict != Verdict.PASS for reports in results.values() for report in reports)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

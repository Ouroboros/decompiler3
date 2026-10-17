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

from ml import fileio
from common.config import default_encoding
from falcom.ed9.parser.scp import ScpParser

from .check import DEFAULT_MAX_MISMATCHES, MAX_RUNS, CheckOptions, FunctionReport, Verdict, check_functions


# Structuring and SSA passes recurse once per nesting level
RECURSION_LIMIT = 20000
PERCENT         = 100


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


def check_file(path: Path, names, options: CheckOptions, verbose: bool):
    with fileio.FileStream(str(path), encoding = default_encoding()) as fs:
        parser = ScpParser(fs, path.name)
        with contextlib.redirect_stdout(io.StringIO()):
            parser.parse()
            functions = parser.disasm_all_functions(filter_func = (lambda f: f.name in names) if names else None)

        def progress(report: FunctionReport):
            if verbose or report.verdict != Verdict.PASS:
                _print_report(report)

        print(f'== {path} ({len(functions)} functions)')
        return check_functions(parser, functions, options, progress)


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
    args = arg_parser.parse_args(argv)

    options = CheckOptions(
        seed = args.seed,
        max_runs = args.max_runs,
        localize = not args.no_localize,
        max_mismatches = args.max_mismatches,
    )

    started = time.time()
    results = {}
    for path in args.scp_files:
        results[str(path)] = check_file(path, set(args.function), options, args.verbose)

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

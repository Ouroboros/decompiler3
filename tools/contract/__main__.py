'''Command line entry: python -m tools.contract SCP_FILE [...]'''

import argparse
import contextlib
import io
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from ml import fileio
from common.config import default_encoding
from falcom.ed9.parser.scp import ScpParser

from .audit import compare_code, compare_tables
from .extract import (
    code_from_hlil, code_from_instructions, code_from_mlil, code_from_ts, entry_from_hlil, entry_from_params,
    run_pipeline, tables_from_parser,
)
from .image import read_tables
from .model import Difference, FunctionEntry, ModuleTables


LAYER_PARSER    = 'parser'
LAYER_LLIL      = 'llil'
LAYER_MLIL      = 'mlil'
LAYER_HLIL      = 'hlil'
LAYER_TS        = 'ts'
LAYER_PIPELINE  = 'pipeline'
IR_LAYERS       = (LAYER_LLIL, LAYER_MLIL, LAYER_HLIL, LAYER_TS)
ALL_LAYERS      = (LAYER_PARSER,) + IR_LAYERS
# The TypeScript backend's file-level format is deferred (docs/REDESIGN.md section 2): code items only
TABLE_LAYERS    = (LAYER_PARSER, LAYER_LLIL, LAYER_MLIL, LAYER_HLIL)
PROGRESS_EVERY  = 100
SLOW_SECONDS    = 10.0
# Structuring and SSA passes recurse once per nesting level
RECURSION_LIMIT = 20000


def _log(message: str):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', file = sys.stderr, flush = True)


def audit_file(path: Path, layers, names) -> List[Difference]:
    data = path.read_bytes()
    reference = read_tables(data)
    script = path.name
    with fileio.FileStream(data, encoding = default_encoding()) as fs:
        parser = ScpParser(fs, script)
        with contextlib.redirect_stdout(io.StringIO()):
            parser.parse()
            functions = parser.disasm_all_functions(filter_func = (lambda f: f.name in names) if names else None)

    checked = [func.name for func in functions]
    differences: List[Difference] = []
    if LAYER_PARSER in layers:
        differences += compare_tables(LAYER_PARSER, script, reference, tables_from_parser(parser), checked)

    ir_layers = [layer for layer in layers if layer in IR_LAYERS]
    if not ir_layers:
        return differences

    entries: Dict[str, Dict[str, FunctionEntry]] = defaultdict(dict)
    started = time.time()
    for index, func in enumerate(functions, start = 1):
        function_started = time.time()
        try:
            forms = run_pipeline(parser, func)

        except Exception as e:
            differences.append(Difference(LAYER_PIPELINE, script, 'error', func.name, detail = f'{type(e).__name__}: {e}'))
            continue

        seconds = time.time() - function_started
        if seconds > SLOW_SECONDS:
            _log(f'{script}: {func.name} took {seconds:.0f}s')

        expected = code_from_instructions(func)
        layer_code = {
            LAYER_LLIL: forms.llil_code,
            LAYER_MLIL: code_from_mlil(forms.mlil),
            LAYER_HLIL: code_from_hlil(forms.hlil),
            LAYER_TS:   code_from_ts(forms.ts),
        }
        for layer in ir_layers:
            differences += compare_code(layer, script, func.name, expected, layer_code[layer], with_addresses = layer != LAYER_TS)

        entries[LAYER_LLIL][func.name] = forms.llil_entry
        entries[LAYER_MLIL][func.name] = entry_from_params(forms.mlil.name, forms.mlil.source_params, forms.mlil.is_common_func)
        entries[LAYER_HLIL][func.name] = entry_from_hlil(forms.hlil)
        if index % PROGRESS_EVERY == 0:
            _log(f'{script}: {index}/{len(functions)} ({func.name}), {time.time() - started:.0f}s')

    for layer in ir_layers:
        if layer in TABLE_LAYERS:
            # Functions are separate objects: no module, so no header, globals or function order yet
            actual = ModuleTables(functions = entries[layer])
            differences += compare_tables(layer, script, reference, actual, list(entries[layer]))

    return differences


def summarize(differences: List[Difference], examples: int):
    groups: Dict[tuple, List[Difference]] = defaultdict(list)
    for difference in differences:
        groups[(difference.layer, difference.item)].append(difference)

    print(f'{"layer":10} {"item":26} {"places":>8} {"elements":>10}')
    for (layer, item), group in sorted(groups.items()):
        print(f'{layer:10} {item:26} {len(group):>8} {sum(d.count for d in group):>10}')
        for difference in group[:examples]:
            where = f'{difference.script} {difference.function}' if difference.function else difference.script
            print(f'    {where}: {difference.detail}')


def main(argv = None) -> int:
    sys.setrecursionlimit(RECURSION_LIMIT)
    arg_parser = argparse.ArgumentParser(prog = 'python -m tools.contract', description = 'Information contract audit')
    arg_parser.add_argument('scp_files', nargs = '+', type = Path)
    arg_parser.add_argument('--layers', default = ','.join(ALL_LAYERS), help = f'comma separated, from {",".join(ALL_LAYERS)}')
    arg_parser.add_argument('--function', action = 'append', default = [], help = 'audit only this function (repeatable)')
    arg_parser.add_argument('--examples', type = int, default = 2, help = 'example differences printed per item')
    arg_parser.add_argument('--json', type = Path, help = 'write every difference to this file')
    args = arg_parser.parse_args(argv)

    layers = [layer.strip() for layer in args.layers.split(',') if layer.strip()]
    unknown = set(layers) - set(ALL_LAYERS)
    if unknown:
        arg_parser.error(f'unknown layers: {sorted(unknown)}')

    started = time.time()
    differences: List[Difference] = []
    for path in args.scp_files:
        _log(f'{path.name}: auditing')
        differences += audit_file(path, layers, set(args.function))
        _log(f'{path.name}: done, {len(differences)} differences so far')

    summarize(differences, args.examples)
    print(f'elapsed {time.time() - started:.1f}s')
    if args.json is not None:
        args.json.write_text(json.dumps([asdict(d) for d in differences], ensure_ascii = False, indent = 1, default = str), encoding = 'utf-8')

    return 1 if differences else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
'''Negative tests of the layer and pass localization of tools/diffcheck: a bug injected into one layer
or pass must be reported at exactly that stage.'''

from pathlib import Path
import contextlib
import io
import sys
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import default_encoding
from falcom.ed9.lifters.vm_lifter import ED9VMLifter
from falcom.ed9.module import ED9ModuleDecompiler
from falcom.ed9.parser.scp import ScpParser
from ir.core import RawWord
from ir.hlil import HLILConst
from ir.hlil.passes.pass_dead_code_elimination import DeadCodeEliminationPass
from ir.llil import LowLevelILConst
from ir.mlil import MLILConst
from ir.mlil.passes.pass_ssa_copy_propagation import CopyPropagationPass
from ml import fileio
from tools.diffcheck import check
from tools.diffcheck.check import CheckOptions, FunctionChecker, Verdict


DAT_PATH        = Path(__file__).parent / 'debug.dat'
# Wrapper that returns Raw 0 (PUSH_RAW 0; SET_REG 0; RETURN)
FUNCTION        = 'set_flag'
WRONG_PAYLOAD   = 1
FIRST_ITERATION = 1


def nodes(node, kind, seen = None):
    '''Every instance of kind reachable from node through attributes and lists'''
    seen = set() if seen is None else seen
    if id(node) in seen or not hasattr(node, '__dict__'):
        return []

    seen.add(id(node))
    found = [node] if isinstance(node, kind) else []
    for value in vars(node).values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            found += nodes(item, kind, seen)

    return found


def corrupt_raw(ir, kind):
    '''Turn every Raw constant of one layer (its constant class) into raw(WRONG_PAYLOAD)

    Only that class is changed: an IR function may reference the IR of the layer it came from.
    '''
    for const in nodes(ir, kind):
        if kind is LowLevelILConst:
            if const.is_raw:
                const.value = WRONG_PAYLOAD

        elif isinstance(const.value, RawWord):
            const.value = RawWord(WRONG_PAYLOAD)

    return ir


def wrapped(method, kind):
    '''Pass method that corrupts the IR it returns'''
    def run(self, ir):
        method(self, ir)
        return corrupt_raw(ir, kind)

    return run


class TestLocalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = DAT_PATH.read_bytes()
        with fileio.FileStream(data, encoding = default_encoding()) as fs:
            cls.parser = ScpParser(fs, DAT_PATH.name)
            with contextlib.redirect_stdout(io.StringIO()):
                cls.parser.parse()
                cls.func = cls.parser.disasm_all_functions(filter_func = lambda f: f.name == FUNCTION)[0]

    def check(self, per_pass: bool = False):
        options = CheckOptions(per_pass = per_pass)
        return FunctionChecker(self.parser, self.func, options, ED9ModuleDecompiler(self.parser)).check()

    def assert_stage(self, report, stage: str):
        self.assertEqual(report.verdict, Verdict.FAIL)
        self.assertEqual(report.mismatches[0].stage, stage)

    def test_unchanged_pipeline_passes_every_stage(self):
        self.assertEqual(self.check().verdict, Verdict.PASS)
        self.assertEqual(self.check(per_pass = True).verdict, Verdict.PASS)

    def test_lifter_bug_is_reported_in_llil(self):
        lift = ED9VMLifter.lift_function
        with mock.patch.object(ED9VMLifter, 'lift_function', lambda self, func: corrupt_raw(lift(self, func), LowLevelILConst)):
            self.assert_stage(self.check(), check.LAYER_LLIL)

    def test_mlil_pass_bug_is_reported_at_that_pass(self):
        stage = f'{check.LAYER_MLIL}/{CopyPropagationPass.__name__}#{FIRST_ITERATION}'
        with mock.patch.object(CopyPropagationPass, 'run', wrapped(CopyPropagationPass.run, MLILConst)):
            self.assert_stage(self.check(), stage)
            self.assert_stage(self.check(per_pass = True), stage)

    def test_hlil_pass_bug_is_reported_at_that_pass(self):
        stage = f'{check.LAYER_HLIL}/{DeadCodeEliminationPass.__name__}'
        with mock.patch.object(DeadCodeEliminationPass, 'run', wrapped(DeadCodeEliminationPass.run, HLILConst)):
            self.assert_stage(self.check(), stage)
            self.assert_stage(self.check(per_pass = True), stage)

    def test_typescript_bug_is_reported_in_typescript(self):
        generate = check.generate_typescript
        emit = lambda func: generate(func).replace(str(RawWord(0)), str(RawWord(WRONG_PAYLOAD)))
        with mock.patch.object(check, 'generate_typescript', emit):
            self.assert_stage(self.check(), check.TYPESCRIPT_STAGE)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
'''Negative tests of the information contract audit (tools/contract): every injected loss must be reported.'''

from pathlib import Path
import contextlib
import copy
import dataclasses
import io
import struct
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import default_encoding
from codegen import generate_typescript
from ir.core import RawWord
from ir.hlil import HLILComment, HLILConst
from ir.llil import LowLevelILDebug
from ml import fileio
from falcom.ed9.disasm import ED9_INSTRUCTION_TABLE, ED9Opcode
from falcom.ed9.module import ED9ModuleDecompiler
from falcom.ed9.parser.scp import ScpParser
from falcom.ed9.parser.writer import write_scp
from tools.contract.audit import compare_code, compare_tables
from tools.contract.extract import (
    code_from_hlil, code_from_instructions, code_from_llil, code_from_mlil, code_from_ts, run_pipeline, tables_from_module,
    tables_from_parser,
)
from tools.contract.image import read_tables
from tools.contract.model import TAG_INT
from tools.diffcheck.check import CheckOptions, FunctionChecker, Verdict, decompile_function


DAT_PATH            = Path(__file__).parent / 'debug.dat'
SCRIPT              = DAT_PATH.name
# Small function with several line markers and no known loss in any layer
LINE_FUNCTION       = 'FC_All_FlagSet_Final_Chapter'
# Wrapper that returns Raw 0 (PUSH_RAW 0; SET_REG 0; RETURN)
RAW_RETURN_FUNCTION = 'set_flag'
FOREIGN_VALUE       = 12345
SINGLE              = 1
# DEBUG_LOG with a negative i32 operand (no DEBUG_LOG occurs in the corpus)
DEBUG_LOG_BYTES     = struct.pack('<Bi', ED9Opcode.DEBUG_LOG, -FOREIGN_VALUE)


def load():
    data = DAT_PATH.read_bytes()
    with fileio.FileStream(data, encoding = default_encoding()) as fs:
        parser = ScpParser(fs, SCRIPT)
        with contextlib.redirect_stdout(io.StringIO()):
            parser.parse()
            functions = parser.disasm_all_functions(filter_func = lambda f: f.name in (LINE_FUNCTION, RAW_RETURN_FUNCTION))

    return data, parser, {func.name: func for func in functions}


class TestTableAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, cls.parser, _ = load()
        cls.reference = read_tables(data)

    def items(self, actual):
        return {(d.item, d.function) for d in compare_tables('test', SCRIPT, self.reference, actual)}

    def test_identical_tables_have_no_differences(self):
        self.assertEqual(self.items(copy.deepcopy(self.reference)), set())

    def test_injected_table_losses_are_reported(self):
        name = next(name for name, entry in self.reference.functions.items() if entry.debug_records and entry.defaults and any(entry.defaults))
        cases = {
            'function.debug_records': lambda entry: setattr(entry, 'debug_records', entry.debug_records[1:]),
            'function.param_flags': lambda entry: setattr(entry, 'param_flags', (FOREIGN_VALUE,) + entry.param_flags[1:]),
            'function.defaults': lambda entry: setattr(entry, 'defaults', (None,) * len(entry.defaults)),
            'function.name_hash': lambda entry: setattr(entry, 'name_hash', FOREIGN_VALUE),
            'function.is_common': lambda entry: setattr(entry, 'is_common', None),
        }
        for item, mutate in cases.items():
            with self.subTest(item = item):
                actual = copy.deepcopy(self.reference)
                mutate(actual.functions[name])
                self.assertEqual(self.items(actual), {(item, name)})

    def test_injected_module_losses_are_reported(self):
        actual = copy.deepcopy(self.reference)
        actual.globals = actual.globals + (('injected', 0),)
        actual.dword_14 = FOREIGN_VALUE
        actual.function_order = None
        self.assertEqual(self.items(actual), {('globals', None), ('dword_14', None), ('function_order', None)})

    def test_parser_matches_the_bytes(self):
        self.assertEqual(self.items(tables_from_parser(self.parser)), set())


class TestModuleTables(unittest.TestCase):
    '''File-level items reach every IL layer through the module metadata'''

    @classmethod
    def setUpClass(cls):
        data, cls.parser, functions = load()
        cls.reference = read_tables(data)
        cls.decompiler = ED9ModuleDecompiler(cls.parser)
        run_pipeline(cls.decompiler, functions[LINE_FUNCTION])
        cls.target = next(name for name, entry in cls.reference.functions.items() if entry.debug_records)

    def items(self, module):
        # Table items the IL functions represent themselves are taken from the reference
        entries = {name: copy.deepcopy(self.reference.functions[name]) for name in (LINE_FUNCTION, self.target)}
        actual = tables_from_module(module, entries)
        return {(d.item, d.function) for d in compare_tables('test', SCRIPT, self.reference, actual, list(entries))}

    def test_every_layer_has_the_file_level_items(self):
        for module in (self.decompiler.llil, self.decompiler.mlil, self.decompiler.hlil):
            with self.subTest(layer = type(module[LINE_FUNCTION]).__name__):
                self.assertEqual(self.items(module), set())

    def test_metadata_losses_are_reported(self):
        module = self.decompiler.hlil.derive()
        metadata = module.metadata
        item = metadata.functions[self.target]
        functions = dict(metadata.functions)
        functions[self.target] = dataclasses.replace(item, debug_info = item.debug_info[1:], byte06 = FOREIGN_VALUE)
        module.metadata = dataclasses.replace(metadata, dword_14 = FOREIGN_VALUE, globals = (), functions = functions)
        self.assertEqual(self.items(module), {('dword_14', None), ('globals', None), ('function.debug_records', self.target),
                                              ('function.byte06', self.target)})


class TestWriter(unittest.TestCase):
    def setUp(self):
        self.data, self.parser, _ = load()

    def test_round_trip_is_byte_identical(self):
        self.assertEqual(write_scp(self.parser), self.data)

    @classmethod
    def target(cls, parser):
        return next(func for func in parser.functions if func.debug_info and func.params)

    def test_model_changes_change_the_bytes(self):
        cases = {
            'global name': lambda parser: setattr(parser.global_vars[0], 'name', parser.global_vars[0].name + 'X'),
            'debug record': lambda parser: self.target(parser).debug_info.pop(),
            'byte06': lambda parser: setattr(self.target(parser), 'byte06', SINGLE),
            'parameter flags': lambda parser: setattr(self.target(parser).params[0].type, 'flags', FOREIGN_VALUE),
        }
        for what, mutate in cases.items():
            with self.subTest(what = what):
                data, parser, _ = load()
                mutate(parser)
                self.assertNotEqual(write_scp(parser), data)

    def test_code_is_written_from_the_instructions(self):
        data, parser, _ = load()
        func = parser.get_func_by_name(LINE_FUNCTION)
        line = next(inst for inst in func.instructions if inst.opcode == ED9Opcode.DEBUG_SET_LINENO)
        line.operands[0].value += SINGLE
        self.assertNotEqual(write_scp(parser), data)


class TestInstructionEncoding(unittest.TestCase):
    def test_debug_log_round_trip(self):
        with fileio.FileStream(DEBUG_LOG_BYTES, encoding = default_encoding()) as fs:
            inst = ED9_INSTRUCTION_TABLE.decode_instruction(fs, 0)

        self.assertEqual(inst.opcode, ED9Opcode.DEBUG_LOG)
        self.assertEqual(inst.operands[0].value, -FOREIGN_VALUE)
        self.assertEqual(inst.size, len(DEBUG_LOG_BYTES))
        self.assertEqual(ED9_INSTRUCTION_TABLE.encode_instruction(inst), DEBUG_LOG_BYTES)


class TestCodeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.parser, functions = load()
        cls.func = functions[LINE_FUNCTION]
        cls.expected = code_from_instructions(cls.func)

    def test_reference_has_line_markers(self):
        self.assertGreater(len(self.expected.lines), SINGLE)

    def test_unchanged_layers_have_no_differences(self):
        forms = run_pipeline(ED9ModuleDecompiler(self.parser), self.func)
        for layer, items in (('llil', forms.llil_code), ('mlil', code_from_mlil(forms.mlil)), ('hlil', code_from_hlil(forms.hlil))):
            with self.subTest(layer = layer):
                self.assertEqual(compare_code(layer, SCRIPT, LINE_FUNCTION, self.expected, items), [])

        self.assertEqual(compare_code('ts', SCRIPT, LINE_FUNCTION, self.expected, code_from_ts(forms.ts), with_addresses = False), [])

    def test_debug_log_reaches_every_layer(self):
        _, parser, functions = load()
        func = functions[LINE_FUNCTION]
        line_offsets = {offset for offset, _ in code_from_instructions(func).lines}
        inst = next(inst for inst in func.instructions if inst.offset in line_offsets)
        inst.opcode = ED9Opcode.DEBUG_LOG
        inst.descriptor = ED9_INSTRUCTION_TABLE.get_descriptor(ED9Opcode.DEBUG_LOG)
        expected = code_from_instructions(func)
        self.assertEqual(len(expected.logs), SINGLE)
        forms = run_pipeline(ED9ModuleDecompiler(parser), func)
        for layer, items in (('llil', forms.llil_code), ('mlil', code_from_mlil(forms.mlil)), ('hlil', code_from_hlil(forms.hlil))):
            with self.subTest(layer = layer):
                self.assertEqual(compare_code(layer, SCRIPT, LINE_FUNCTION, expected, items), [])

        self.assertEqual(compare_code('ts', SCRIPT, LINE_FUNCTION, expected, code_from_ts(forms.ts), with_addresses = False), [])

    def assert_one_line_lost(self, differences):
        self.assertEqual([(d.item, d.count) for d in differences], [('code.lines', SINGLE)])

    def test_removed_llil_line_is_reported(self):
        forms = run_pipeline(ED9ModuleDecompiler(self.parser), self.func)
        block = next(block for block in forms.llil.basic_blocks if any(isinstance(i, LowLevelILDebug) for i in block.instructions))
        block.instructions.remove(next(i for i in block.instructions if isinstance(i, LowLevelILDebug)))
        self.assert_one_line_lost(compare_code('llil', SCRIPT, LINE_FUNCTION, self.expected, code_from_llil(forms.llil)))

    def test_removed_hlil_line_is_reported(self):
        forms = run_pipeline(ED9ModuleDecompiler(self.parser), self.func)
        body = forms.hlil.body.statements
        body.remove(next(stmt for stmt in body if isinstance(stmt, HLILComment) and stmt.text.startswith('line(')))
        self.assert_one_line_lost(compare_code('hlil', SCRIPT, LINE_FUNCTION, self.expected, code_from_hlil(forms.hlil)))

    def test_moved_hlil_line_is_reported(self):
        forms = run_pipeline(ED9ModuleDecompiler(self.parser), self.func)
        comment = next(stmt for stmt in forms.hlil.body.statements if isinstance(stmt, HLILComment) and stmt.text.startswith('line('))
        comment.address += SINGLE
        differences = compare_code('hlil', SCRIPT, LINE_FUNCTION, self.expected, code_from_hlil(forms.hlil))
        self.assertEqual([(d.item, d.count) for d in differences], [('code.lines', SINGLE + SINGLE)])

    def test_removed_typescript_line_is_reported(self):
        forms = run_pipeline(ED9ModuleDecompiler(self.parser), self.func)
        line = next(value for _, value in sorted(self.expected.lines))
        text = forms.ts.replace(f'// line({line})', '', SINGLE)
        self.assert_one_line_lost(compare_code('ts', SCRIPT, LINE_FUNCTION, self.expected, code_from_ts(text), with_addresses = False))


def raw_constants(node, seen = None):
    '''Every HLILConst holding a RawWord reachable from node'''
    seen = set() if seen is None else seen
    if id(node) in seen or not hasattr(node, '__dict__'):
        return []

    seen.add(id(node))
    found = [node] if isinstance(node, HLILConst) and isinstance(node.value, RawWord) else []
    for value in vars(node).values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            found += raw_constants(item, seen)

    return found


class TestStrictRaw(unittest.TestCase):
    def setUp(self):
        _, self.parser, functions = load()
        self.func = functions[RAW_RETURN_FUNCTION]
        self.decompiler = ED9ModuleDecompiler(self.parser)

    def check(self, output, strict_raw: bool):
        return FunctionChecker(self.parser, self.func, CheckOptions(strict_raw = strict_raw), self.decompiler).check(output).verdict

    def test_raw_return_is_kept(self):
        output = decompile_function(self.decompiler, self.func)
        self.assertTrue(raw_constants(output.hlil))
        self.assertIn('raw(0)', output.ts_text)
        self.assertEqual(self.check(output, strict_raw = True), Verdict.PASS)

    def test_raw_return_turned_into_integer_is_a_mismatch(self):
        output = decompile_function(self.decompiler, self.func)
        for const in raw_constants(output.hlil):
            const.value = const.value.payload

        output.ts_text = generate_typescript(output.hlil)
        self.assertEqual(self.check(output, strict_raw = False), Verdict.PASS)
        self.assertEqual(self.check(output, strict_raw = True), Verdict.FAIL)


if __name__ == '__main__':
    unittest.main()

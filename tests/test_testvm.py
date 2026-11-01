#!/usr/bin/env python3
'''Stage 8 of docs/REDESIGN.md: a VM unlike ED9 goes through the unchanged decompiler core.

TVM (tests/testvm) implements only the target interfaces. It must round-trip, keep every contract item
(line markers, debug logs, parameter flags, module metadata) through LLIL, MLIL and HLIL, and its HLIL
must behave like its bytecode.
'''

from pathlib import Path
import copy
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parent.parent))

from codegen import generate_typescript, generate_typescript_header
from ir.hlil import HLILConst
from tests.testvm.decompiler import TVMModuleDecompiler
from tests.testvm.image import PARAM_INT, PARAM_STR, Function, Image, assemble, read_image, write_image
from tests.testvm.interp import BytecodeMachine, HLILMachine, World
from tests.testvm.isa import WORD_SIZE, Op, decode, encode
from tests.testvm.target import RESULT_REGISTER, TVMMetadata, TVMTarget
from tools.contract.audit import compare_code
from tools.contract.extract import code_from_hlil, code_from_llil, code_from_mlil
from tools.contract.model import CodeItems


VERSION         = 3
GLOBAL_COUNT    = 4
SEEDS           = 200
W               = WORD_SIZE
STRINGS         = ['branch', 'loop', 'chain', 'caller', 'logic', 'divide', 'print', 'one', 'two', 'three']
PRINT           = STRINGS.index('print')
BRANCH_THRESHOLD = 10


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


def slot(index: int, sp: int) -> int:
    '''LDL / STL operand reaching stack slot index when the stack holds sp slots'''
    return (index - sp) * W


FUNCTIONS = [
    # branch(a, b): G0 = 1 and print(a) when a + b > 10, else G1 = b - a; returns G0
    ('branch', (PARAM_INT, PARAM_INT), [
        (Op.LINE, 10), (Op.LDL, slot(0, 2)), (Op.LDL, slot(1, 3)), (Op.ADD,), (Op.PUSH, BRANCH_THRESHOLD), (Op.GT,), (Op.JZ, 'else'),
        (Op.LINE, 11), (Op.PUSH, 1), (Op.STG, 0), (Op.LDL, slot(0, 2)), (Op.HOST, PRINT, 1), (Op.JMP, 'end'),
        'else', (Op.LINE, 12), (Op.LDL, slot(1, 2)), (Op.LDL, slot(0, 3)), (Op.SUB,), (Op.STG, 1),
        'end', (Op.LINE, 13), (Op.LDG, 0), (Op.STR, RESULT_REGISTER), (Op.RET,),
    ]),
    # loop(n): sum of 0 .. n - 1 in two locals
    ('loop', (PARAM_INT,), [
        (Op.PUSH, 0), (Op.PUSH, 0),
        'top', (Op.LINE, 20), (Op.LDL, slot(2, 3)), (Op.LDL, slot(0, 4)), (Op.LT,), (Op.JZ, 'done'),
        (Op.LDL, slot(1, 3)), (Op.LDL, slot(2, 4)), (Op.ADD,), (Op.STL, slot(1, 3)),
        (Op.LDL, slot(2, 3)), (Op.PUSH, 1), (Op.ADD,), (Op.STL, slot(2, 3)), (Op.JMP, 'top'),
        'done', (Op.LINE, 21), (Op.LDL, slot(1, 3)), (Op.STR, RESULT_REGISTER), (Op.DROP, 2), (Op.RET,),
    ]),
    # chain(x): x == 1 / 2 / 3 print a word, otherwise print x
    ('chain', (PARAM_INT,), [
        (Op.LINE, 30), (Op.LDL, slot(0, 1)), (Op.PUSH, 1), (Op.EQ,), (Op.JZ, 'c2'),
        (Op.PUSHS, STRINGS.index('one')), (Op.HOST, PRINT, 1), (Op.JMP, 'end'),
        'c2', (Op.LINE, 31), (Op.LDL, slot(0, 1)), (Op.PUSH, 2), (Op.EQ,), (Op.JZ, 'c3'),
        (Op.PUSHS, STRINGS.index('two')), (Op.HOST, PRINT, 1), (Op.JMP, 'end'),
        'c3', (Op.LINE, 32), (Op.LDL, slot(0, 1)), (Op.PUSH, 3), (Op.EQ,), (Op.JZ, 'other'),
        (Op.PUSHS, STRINGS.index('three')), (Op.HOST, PRINT, 1), (Op.JMP, 'end'),
        'other', (Op.LINE, 33), (Op.LDL, slot(0, 1)), (Op.HOST, PRINT, 1),
        'end', (Op.PUSH, 7), (Op.STR, RESULT_REGISTER), (Op.RET,),
    ]),
    # caller(a, s): G2 = 2 * branch(a, 3); print(s); returns R1 mod 4 (floor modulo)
    ('caller', (PARAM_INT, PARAM_STR), [
        (Op.LINE, 40), (Op.LDL, slot(0, 2)), (Op.PUSH, 3), (Op.CALL, 0, 2),
        (Op.LDR, RESULT_REGISTER), (Op.PUSH, 2), (Op.MUL,), (Op.STG, 2), (Op.LOG, 99),
        (Op.LDL, slot(1, 2)), (Op.HOST, PRINT, 1),
        (Op.LINE, 41), (Op.LDR, 1), (Op.PUSH, 4), (Op.MOD,), (Op.STR, RESULT_REGISTER), (Op.RET,),
    ]),
    # logic(a, b): 1 when both are nonzero
    ('logic', (PARAM_INT, PARAM_INT), [
        (Op.LDL, slot(0, 2)), (Op.JZ, 'no'), (Op.LDL, slot(1, 2)), (Op.JZ, 'no'),
        (Op.PUSH, 1), (Op.STR, RESULT_REGISTER), (Op.RET,),
        'no', (Op.PUSH, 0), (Op.STR, RESULT_REGISTER), (Op.RET,),
    ]),
    # divide(a, b): a // b (faults on 0); G3 = -7 // 2 folded with floor division (-4, where ED9 truncates to -3)
    ('divide', (PARAM_INT, PARAM_INT), [
        (Op.PUSH, -7), (Op.PUSH, 2), (Op.DIV,), (Op.STG, 3),
        (Op.LDL, slot(0, 2)), (Op.LDL, slot(1, 3)), (Op.DIV,), (Op.NEG,), (Op.NOT,), (Op.STR, RESULT_REGISTER), (Op.RET,),
    ]),
]


def build_image() -> Image:
    image = Image(VERSION, GLOBAL_COUNT, list(STRINGS))
    for name, flags, lines in FUNCTIONS:
        image.functions.append(Function(name, flags, assemble(lines)))

    return image


def reference_items(func: Function) -> CodeItems:
    items = CodeItems()
    for inst in func.instructions:
        if inst.op == Op.LINE:
            items.lines.add((func.offset + inst.offset, inst.operands[0]))

        elif inst.op == Op.LOG:
            items.logs.add((func.offset + inst.offset, inst.operands[0]))

    return items


class TestTVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = write_image(build_image())
        cls.image = read_image(cls.data)
        cls.decompiler = TVMModuleDecompiler(cls.image)
        cls.layers = {func.name: cls.decompiler.decompile(func) for func in cls.image.functions}

    def test_round_trip(self):
        self.assertEqual(write_image(read_image(self.data)), self.data)
        for func in self.image.functions:
            with self.subTest(function = func.name):
                self.assertEqual(encode(decode(func.code)), func.code)

    def test_line_markers_and_logs_reach_every_layer(self):
        for func in self.image.functions:
            llil, mlil, hlil = self.layers[func.name]
            expected = reference_items(func)
            for layer, items in (('llil', code_from_llil(llil)), ('mlil', code_from_mlil(mlil)), ('hlil', code_from_hlil(hlil))):
                with self.subTest(function = func.name, layer = layer):
                    self.assertEqual(compare_code(layer, 'tvm', func.name, expected, items), [])

        self.assertTrue(any(reference_items(func).logs for func in self.image.functions))

    def test_parameter_flags_and_metadata_reach_every_layer(self):
        metadata = TVMMetadata.from_image(self.image)
        for module in (self.decompiler.llil, self.decompiler.mlil, self.decompiler.hlil):
            self.assertEqual(module.metadata, metadata)

        for func in self.image.functions:
            llil, mlil, hlil = self.layers[func.name]
            with self.subTest(function = func.name):
                self.assertEqual(tuple(param.flags for param in llil.params), func.param_flags)
                self.assertEqual(tuple(param.flags for param in mlil.source_params), func.param_flags)
                self.assertEqual(tuple(param.flags for param in hlil.parameters), func.param_flags)
                self.assertIs(hlil.target, TVMTarget)

    def test_hlil_behaves_like_the_bytecode(self):
        for func in self.image.functions:
            hlil = self.layers[func.name][-1]
            for seed in range(SEEDS):
                with self.subTest(function = func.name, seed = seed):
                    reference = BytecodeMachine(self.image, func, World(seed)).run()
                    candidate = HLILMachine(hlil, len(func.param_flags), World(seed)).run()
                    self.assertEqual(candidate, reference)

    def test_behavior_check_sees_a_wrong_constant(self):
        func = self.image.functions[self.image.function_index('branch')]
        hlil = copy.deepcopy(self.layers[func.name][-1])
        threshold = next(const for const in nodes(hlil, HLILConst) if const.value == BRANCH_THRESHOLD)
        threshold.value += 1
        mismatches = [seed for seed in range(SEEDS)
                      if HLILMachine(hlil, len(func.param_flags), World(seed)).run() != BytecodeMachine(self.image, func, World(seed)).run()]
        self.assertTrue(mismatches)

    def test_target_spelling_reaches_the_output(self):
        texts = {name: generate_typescript(layers[-1]) for name, layers in self.layers.items()}
        self.assertIn('host.print(', texts['chain'])
        self.assertIn('G[', texts['branch'])
        self.assertIn('R[1]', texts['caller'])
        self.assertIn('G[3] = -4', texts['divide'])
        self.assertIn('declare const host', generate_typescript_header(TVMTarget))


if __name__ == '__main__':
    unittest.main()

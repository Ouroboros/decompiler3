#!/usr/bin/env python3
'''The IR core and the backends know no target: ir/ and codegen/ import no target package and name no
ED9 concept (docs/REDESIGN.md stage 7). ED9 reaches them only through ir/core/target.py.'''

from pathlib import Path
import ast
import re
import unittest


ROOT                = Path(__file__).parent.parent
CORE_PACKAGES       = ('ir', 'codegen')
TARGET_PACKAGES     = ('falcom', 'tools')
# Names of ED9 concepts; case-sensitive whole words (generic words such as "global" stay allowed)
ED9_NAMES           = (
    'ED9', 'ed9', 'Falcom', 'falcom', 'FALCOM', 'Kuro', 'Kiseki', 'Scp', 'SCP', 'scp',
    'syscall', 'Syscall', 'SYSCALL', 'syscalls', 'subsystem',
    'CallScript', 'call_script', 'CALL_SCRIPT', 'caller_frame', 'CallerFrame', 'CALLER_FRAME', 'extern_call',
    'Value32', 'Nullable32', 'NullableStr', 'REGS', 'GLOBALS', 'WORD_SIZE',
)
ED9_NAME_PATTERN    = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(re.escape(name) for name in ED9_NAMES) + r')(?![A-Za-z0-9_])')


def core_sources():
    for package in CORE_PACKAGES:
        yield from sorted((ROOT / package).rglob('*.py'))


class TestTargetIsolation(unittest.TestCase):
    def test_core_imports_no_target_package(self):
        found = []
        for path in core_sources():
            for node in ast.walk(ast.parse(path.read_text(encoding = 'utf-8'))):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]

                found += [f'{path.relative_to(ROOT)}:{node.lineno}: {name}' for name in names
                          if name.split('.')[0] in TARGET_PACKAGES]

        self.assertEqual(found, [])

    def test_core_names_no_ed9_concept(self):
        found = []
        for path in core_sources():
            for number, line in enumerate(path.read_text(encoding = 'utf-8').splitlines(), start = 1):
                found += [f'{path.relative_to(ROOT)}:{number}: {match.group(1)}' for match in ED9_NAME_PATTERN.finditer(line)]

        self.assertEqual(found, [])

    def test_the_check_sees_ed9_names(self):
        self.assertTrue(ED9_NAME_PATTERN.search('REGS[0] = syscall(1, 2)'))
        self.assertFalse(ED9_NAME_PATTERN.search('global_name = register_names'))


if __name__ == '__main__':
    unittest.main()

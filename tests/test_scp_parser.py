#!/usr/bin/env python3
'''
Unit tests for SCP parser
'''

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import *
from io import BytesIO
from common import *
from falcom.ed9 import *
from falcom.ed9.lifters import ED9VMLifter
from falcom.ed9.llil_builder import FalcomLLILFormatter
from falcom.ed9.mlil_translator import translate_falcom_llil_to_mlil
from ir.llil.llil import LowLevelILFunction
from ir.mlil import MLILFormatter, optimize_mlil, generate_typescript
import unittest
import struct


class TestScpValue(unittest.TestCase):
    '''Test ScpValue serialization and deserialization'''

    def test_raw_value(self):
        '''Test Raw type value'''
        # Raw value is stored as-is
        raw_val = 0x12345678
        val = ScpValue(RawInt(raw_val))

        # Serialize
        data = val.to_bytes()
        self.assertEqual(len(data), 4)

        # Deserialize
        fs = fileio.FileStream()
        fs.OpenMemory(data)
        val2 = ScpValue(fs = fs)
        self.assertEqual(val2.value, raw_val)
        self.assertEqual(val2.type, ScpValue.Type.Raw)

    def test_integer_value(self):
        '''Test Integer type value'''
        test_values = [0, 1, -1, 100, -100, 0x1FFFFFFF]  # 0x1FFFFFFF is safe max for 30-bit signed

        for test_val in test_values:
            with self.subTest(value = test_val):
                val = ScpValue(test_val)
                self.assertEqual(val.type, ScpValue.Type.Integer)

                # Serialize
                data = val.to_bytes()
                self.assertEqual(len(data), 4)

                # Check type bits (top 2 bits should be 01)
                endian = '<' if default_endian() == 'little' else '>'
                raw = struct.unpack(f'{endian}I', data)[0]
                type_bits = raw >> 30
                self.assertEqual(type_bits, ScpValue.Type.Integer)

                # Deserialize
                fs = fileio.FileStream()
                fs.OpenMemory(data)
                val2 = ScpValue(fs = fs)
                self.assertEqual(val2.value, test_val)
                self.assertEqual(val2.type, ScpValue.Type.Integer)

    def test_float_value(self):
        '''Test Float type value'''
        test_values = [0.0, 1.0, -1.0, 3.14159, -2.71828, 100.5]

        for test_val in test_values:
            with self.subTest(value = test_val):
                val = ScpValue(test_val)
                self.assertEqual(val.type, ScpValue.Type.Float)

                # Serialize
                data = val.to_bytes()
                self.assertEqual(len(data), 4)

                # Deserialize
                fs = fileio.FileStream()
                fs.OpenMemory(data)
                val2 = ScpValue(fs = fs)
                self.assertAlmostEqual(val2.value, test_val, places = 5)
                self.assertEqual(val2.type, ScpValue.Type.Float)

    def test_roundtrip(self):
        '''Test value roundtrip (serialize then deserialize)'''
        test_cases = [
            RawInt(0x12345678),  # Must have top 2 bits as 00 for Raw type
            42,
            -123,
            3.14159,
            -2.71828,
        ]

        for original_value in test_cases:
            with self.subTest(value = original_value):
                val1 = ScpValue(original_value)
                data = val1.to_bytes()

                fs = fileio.FileStream()
                fs.OpenMemory(data)
                val2 = ScpValue(fs = fs)

                self.assertEqual(val1.type, val2.type)
                if isinstance(original_value, float):
                    self.assertAlmostEqual(val1.value, val2.value, places = 5)
                else:
                    self.assertEqual(val1.value, val2.value)

    def test_string_value_serialize(self):
        '''Test String type cannot be serialized with to_bytes (not implemented)'''
        val = ScpValue("test string")
        self.assertEqual(val.type, ScpValue.Type.String)

        # String serialization is not implemented
        with self.assertRaises(NotImplementedError):
            val.to_bytes()

    def test_string_value_deserialize(self):
        '''Test String type deserialization'''
        endian = '<' if default_endian() == 'little' else '>'

        # Create a buffer with string data at offset 0x100
        buffer = BytesIO()
        buffer.write(b'\x00' * 0x100)  # Padding
        string_offset = buffer.tell()
        test_string = "TestFunction"
        buffer.write(test_string.encode('utf-8') + b'\x00')  # Null-terminated

        # Write ScpValue at beginning pointing to string
        buffer.seek(0)
        string_value = string_offset | (ScpValue.Type.String << 30)
        buffer.write(struct.pack(f'{endian}I', string_value))

        # Parse it
        buffer.seek(0)
        fs = fileio.FileStream()
        fs.OpenMemory(buffer.getvalue())
        # Set encoding to utf-8 to avoid mbcs encoding error on Linux
        fs._encoding = 'utf-8'
        val = ScpValue(fs = fs)

        self.assertEqual(val.type, ScpValue.Type.String)
        self.assertEqual(val.value, test_string)


class TestScpParser(unittest.TestCase):
    '''Test SCP parser'''

    def test_parser_with_real_file(self):
        '''Test parser with real SCP file if available'''

        ED9_DATA_DIR = Path(r'D:\Game\Steam\steamapps\common\THE LEGEND OF HEROES KURO NO KISEKI\decrypted\tc\f\script\scena')

        # test_file = Path(__file__).parent / 'mp2000_ev.dat'
        test_file = Path(__file__).parent / 'debug.dat'
        # test_file = Path(__file__).parent / 'mp3010_01.dat'
        # test_file = ED9_DATA_DIR / 'c0600.dat'

        if not test_file.exists():
            self.skipTest(f'Test file not found: {test_file}')

        with fileio.FileStream(str(test_file), encoding = default_encoding()) as fs:
            parser = ScpParser(fs, test_file.name)
            parser.parse()

            # for i, func in enumerate(parser.functions):
            #     print(f'[{i}] {func}')
            #     print()

            # print('global vars:')

            # for i, gvar in enumerate(parser.global_vars):
            #     print(f'[{i}] {gvar}')
            #     print()

            print(parser.header)

            # Check header was parsed
            self.assertIsNotNone(parser.header)
            self.assertGreater(parser.header.function_count, 0)

            # Test disassembly and formatting
            disassembled_functions = parser.disasm_all_functions(
                filter_func = lambda f: f.name == 'EVENT_BEGIN'
            )

            formatted_lines = []
            llil_lines = []
            mlil_lines = []
            typescript_lines = []

            # Print formatted functions
            print('\n=== Disassembly ===\n')
            for func in disassembled_functions:
                print(f'{func} @ 0x{func.offset:08X}')

                formatted_lines.extend(parser.format_function(func))
                formatted_lines.append('')

                # Generate LLIL
                lifter = ED9VMLifter(parser = parser)
                llil_func = lifter.lift_function(func)
                self.assertIsInstance(llil_func, LowLevelILFunction)

                llil_lines.extend(FalcomLLILFormatter.format_llil_function(llil_func))
                llil_lines.append('')

                # Generate MLIL from LLIL
                mlil_func = translate_falcom_llil_to_mlil(llil_func)

                # Optimize MLIL (includes type inference)
                mlil_func = optimize_mlil(mlil_func, use_ssa = True, infer_types_enabled = True)

                mlil_lines.extend(MLILFormatter.format_function(mlil_func))
                mlil_lines.append('')

                # Generate TypeScript
                typescript_code = generate_typescript(mlil_func)
                typescript_lines.append(typescript_code)
                typescript_lines.append('')

            vmpy_path = test_file.with_suffix('.py')
            vmpy_path.write_text('\n'.join(formatted_lines) + '\n', encoding = 'utf-8')

            llil_path = test_file.with_suffix('.llil.asm')
            llil_path.write_text('\n'.join(llil_lines) + '\n', encoding = 'utf-8')

            mlil_path = test_file.with_suffix('.mlil.asm')
            mlil_path.write_text('\n'.join(mlil_lines) + '\n', encoding = 'utf-8')

            typescript_path = test_file.with_suffix('.ts')
            typescript_path.write_text('\n'.join(typescript_lines) + '\n', encoding = 'utf-8')

if __name__ == '__main__':
    unittest.main(buffer = False, defaultTest = 'TestScpParser')

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
from falcom.ed9.parser import *
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


class TestScpHeader(unittest.TestCase):
    '''Test SCP header parsing'''

    def create_test_header(self):
        '''Create a test SCP header'''
        endian = '<' if default_endian() == 'little' else '>'
        data = struct.pack(
            f'{endian}4s5I',
            b'#scp',          # magic
            0x1000,           # function_entry_offset
            10,               # function_count
            0x2000,           # global_var_offset
            5,                # global_var_count
            0x12345678        # dword_14
        )
        return data

    def test_header_parse(self):
        '''Test parsing SCP header'''
        data = self.create_test_header()

        fs = fileio.FileStream()
        fs.OpenMemory(data)
        header = ScpHeader(fs = fs)

        self.assertEqual(header.function_entry_offset, 0x1000)
        self.assertEqual(header.function_count, 10)
        self.assertEqual(header.global_var_offset, 0x2000)
        self.assertEqual(header.global_var_count, 5)
        self.assertEqual(header.dword_14, 0x12345678)

    def test_invalid_magic(self):
        '''Test header with invalid magic'''
        endian = '<' if default_endian() == 'little' else '>'
        data = struct.pack(
            f'{endian}4s5I',
            b'#xxx',          # wrong magic
            0, 0, 0, 0, 0
        )

        fs = fileio.FileStream()
        fs.OpenMemory(data)

        with self.assertRaises(AssertionError):
            ScpHeader(fs = fs)


class TestScenaFunctionEntry(unittest.TestCase):
    '''Test ScenaFunctionEntry parsing'''

    def create_test_function_entry(self):
        '''Create a test function entry'''
        endian = '<' if default_endian() == 'little' else '>'
        # ScenaFunctionEntry structure (0x20 bytes)
        data = struct.pack(
            f'{endian}I4B5I',
            0x5000,           # offset
            3,                # param_count
            0x01,             # byte05
            0x02,             # byte06
            1,                # default_params_count
            0x6000,           # default_params_offset
            0x7000,           # param_flags_offset
            5,                # debug_symbol_count
            0x8000,           # debug_symbol_offset
            0xABCDEF12,       # name_crc32
        )

        # Add name_offset as ScpValue (String type pointing to offset)
        # String type: top 2 bits = 11, bottom 30 bits = offset
        name_offset = 0x9000 | (ScpValue.Type.String << 30)
        data += struct.pack(f'{endian}I', name_offset)

        return data

    def test_function_entry_parse(self):
        '''Test parsing function entry'''
        data = self.create_test_function_entry()

        fs = fileio.FileStream(encoding = default_encoding())
        fs.OpenMemory(data)
        entry = ScpFunctionEntry(fs = fs)

        self.assertEqual(entry.offset, 0x5000)
        self.assertEqual(entry.param_count, 3)
        self.assertEqual(entry.byte05, 0x01)
        self.assertEqual(entry.byte06, 0x02)
        self.assertEqual(entry.default_params_count, 1)
        self.assertEqual(entry.default_params_offset, 0x6000)
        self.assertEqual(entry.param_flags_offset, 0x7000)
        self.assertEqual(entry.debug_symbol_count, 5)
        self.assertEqual(entry.debug_symbol_offset, 0x8000)
        self.assertEqual(entry.name_hash, 0xABCDEF12)
        # name_offset is read as a string through ScpValue
        self.assertEqual(entry.name_offset, '')


class TestScpParser(unittest.TestCase):
    '''Test SCP parser'''

    def test_parser_with_real_file(self):
        '''Test parser with real SCP file if available'''

        ED9_DATA_DIR = Path(r'D:\Game\Steam\steamapps\common\THE LEGEND OF HEROES KURO NO KISEKI\decrypted\tc\f\script\scena')

        test_file = Path(__file__).parent / 'mp3010_01.dat'
        # test_file = ED9_DATA_DIR / 'c0600.dat'

        if not test_file.exists():
            self.skipTest(f'Test file not found: {test_file}')

        with fileio.FileStream(str(test_file), encoding = default_encoding()) as fs:
            parser = ScpParser(fs, test_file.name)
            parser.parse()

        for i, func in enumerate(parser.functions):
            print(f'[{i}] {func}')
            print()

        print('global vars:')

        for i, gvar in enumerate(parser.global_vars):
            print(f'[{i}] {gvar}')
            print()

        print(parser.header)

        # Check header was parsed
        self.assertIsNotNone(parser.header)
        self.assertGreater(parser.header.function_count, 0)


if __name__ == '__main__':
    unittest.main(buffer = False, defaultTest = 'TestScpParser')

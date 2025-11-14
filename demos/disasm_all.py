#!/usr/bin/env python3
"""
Disassemble all functions from .dat files and output to .py files
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ml import *
from falcom.ed9.disasm import *
from falcom.ed9.parser import *
from pathlib import Path
import argparse


def disasm_file(input_path: Path, output_path: Path):
    """Disassemble all functions in a single file"""
    print(f'Processing {input_path.name}...')

    with fileio.FileStream(str(input_path), encoding = default_encoding()) as fs:
        parser = ScpParser(fs, input_path.name)
        parser.parse()

        # Create context with callbacks
        context = DisassemblerContext(
            get_func_argc = lambda func_id: parser.get_func_argc(func_id),
            optimize_instruction = ed9_optimize_instruction
        )

        disasm = Disassembler(ED9_INSTRUCTION_TABLE, context)

        # Create formatter context
        formatter_context = FormatterContext(
            get_func_name = lambda func_id: parser.get_func_name(func_id)
        )

        formatter = Formatter(formatter_context)

        # Disassemble all functions
        all_lines = []
        all_lines.append(f'# Disassembly of {input_path.name}')
        all_lines.append('')

        for func in parser.functions:
            try:
                entry = disasm.disasm_function(fs, offset = func.offset, name = func.name)
                lines = formatter.format_function(entry)
                all_lines.extend(lines)
                all_lines.append('')
            except Exception as e:
                raise RuntimeError(f'  Error disassembling {func.name}: {e}')

        # Write output
        output_path.write_text('\n'.join(all_lines), encoding = 'utf-8')
        print(f'  -> {output_path}')


def main():
    parser = argparse.ArgumentParser(description = 'Disassemble .dat files to .py files')
    parser.add_argument('path', help = 'Input file or directory')
    args = parser.parse_args()

    input_path = Path(args.path)

    if not input_path.exists():
        print(f'Error: {input_path} does not exist')
        return 1

    files = []

    if input_path.is_file():
        # Single file
        if input_path.suffix != '.dat':
            print(f'Error: {input_path} is not a .dat file')
            return 1

        files.append(input_path)

    elif input_path.is_dir():
        # Directory

        files = [Path(f) for f in fileio.getDirectoryFiles(str(input_path), '*.dat', subdir = False)]

    else:
        print(f'Error: {input_path} is neither a file nor a directory')
        return 1

    for f in files:
        dat_file = Path(f)
        output_path = dat_file.with_suffix('.py')
        try:
            disasm_file(dat_file, output_path)
        except Exception as e:
            raise RuntimeError(f'Error processing {dat_file.name}: {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

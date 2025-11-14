#!/usr/bin/env python3
"""
Disassembler Demo - Test the ED9 disassembler
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ml import *
from falcom.ed9.disasm import *
from falcom.ed9.parser import *
from pathlib import Path


def print_block(block: BasicBlock, visited=None, indent=0, all_blocks=None):
    """Recursively print basic block and its successors"""
    if visited is None:
        visited = set()

    if all_blocks is None:
        # First call: collect all blocks
        all_blocks = []
        collect_blocks(block, all_blocks, set())
        # Sort by offset
        all_blocks.sort(key=lambda b: b.offset)
        # Print in order
        for b in all_blocks:
            if b.instructions:  # Skip empty blocks
                print(f'Block {b.name}:')
                for inst in b.instructions:
                    print(f'  {inst.offset:08X}: {inst.descriptor.format_instruction(inst)}')
                if b.succs:
                    succ_names = ', '.join(br.name for br in b.succs)
                    print(f'  → {succ_names}')
                print()
        return

    # Legacy recursive implementation (not used anymore)
    if block.offset in visited:
        print(f'{"  " * indent}→ {block.name} (already visited)')
        return

    visited.add(block.offset)

    if not block.instructions:
        return

    print(f'{"  " * indent}Block {block.name}:')
    for inst in block.instructions:
        print(f'{"  " * indent}  {inst.offset:08X}: {inst.descriptor.format_instruction(inst)}')

    if block.succs:
        print(f'{"  " * indent}  Successors:')
        for succ in block.succs:
            print_block(succ, visited, indent + 1, all_blocks)


def collect_blocks(block, result, visited):
    """Collect all blocks in the CFG"""
    if block.offset in visited:
        return
    visited.add(block.offset)
    result.append(block)
    for succ in block.succs:
        collect_blocks(succ, result, visited)


def test_simple_function():
    """Test with a simple bytecode sequence"""
    print('=== Test: Simple Function ===\n')

    # Simple bytecode: PUSH_INT(1), PUSH_INT(2), ADD, SET_REG(0), RETURN
    bytecode = bytes([
        0x00, 0x01, 0x00, 0x00, 0x40,  # PUSH int(1): 0x40000001
        0x00, 0x02, 0x00, 0x00, 0x40,  # PUSH int(2): 0x40000002
        0x10,                           # ADD
        0x0A, 0x00,                     # SET_REG(0)
        0x0D,                           # RETURN
    ])

    # Disassemble
    disasm = Disassembler(ED9_INSTRUCTION_TABLE)

    entry = disasm.disasm_function(bytecode, offset=0, name='test_add')

    # Print result
    print_block(entry)
    print()


def test_conditional():
    """Test with conditional branch"""
    print('=== Test: Conditional Branch ===\n')

    # Bytecode: GET_REG(0), POP_JMP_ZERO(loc_11), PUSH_INT(1), JMP(loc_16), loc_11: PUSH_INT(0), loc_16: RETURN
    bytecode = bytes([
        0x09, 0x00,                     # 0x00: GET_REG(0)
        0x0F, 0x11, 0x00, 0x00, 0x00,  # 0x02: POP_JMP_ZERO(0x11)
        0x00, 0x01, 0x00, 0x00, 0x40,  # 0x07: PUSH int(1)
        0x0B, 0x16, 0x00, 0x00, 0x00,  # 0x0C: JMP(0x16)
        0x00, 0x00, 0x00, 0x00, 0x40,  # 0x11: PUSH int(0)
        0x0D,                           # 0x16: RETURN
    ])

    # Disassemble
    disasm = Disassembler(ED9_INSTRUCTION_TABLE)

    entry = disasm.disasm_function(bytecode, offset=0, name='test_cond')

    # Print result
    print_block(entry)
    print()


def test_caller_frame():
    """Test PUSH_CALLER_FRAME (allocates return point)"""
    print('=== Test: PUSH_CALLER_FRAME ===\n')

    # PUSH_CALLER_FRAME allocates a return point, then continues execution
    # Flow: PUSH_CALLER_FRAME(loc_return), do_work, JMP(somewhere)
    # loc_return will be used when called function returns
    bytecode = bytes([
        0x25, 0x11, 0x00, 0x00, 0x00,  # 0x00: PUSH_CALLER_FRAME(0x11)
        0x00, 0x01, 0x00, 0x00, 0x40,  # 0x05: PUSH int(1)
        0x0A, 0x00,                     # 0x0A: SET_REG(0)
        0x0B, 0x11, 0x00, 0x00, 0x00,  # 0x0C: JMP(0x11) -> jump to return point
        # loc_11 (return point allocated by PUSH_CALLER_FRAME)
        0x09, 0x00,                     # 0x11: GET_REG(0)
        0x0D,                           # 0x13: RETURN
    ])

    # Disassemble
    disasm = Disassembler(ED9_INSTRUCTION_TABLE)

    entry = disasm.disasm_function(bytecode, offset=0, name='test_frame')

    # Print result
    print_block(entry)
    print()


def test_loop():
    """Test with backward jump (loop)"""
    print('=== Test: Loop (Backward Jump) ===\n')

    # Simulates: loop with condition
    # PUSH int(0), SET_REG(0)
    # loc_loop: GET_REG(0), PUSH int(10), LT, POP_JMP_ZERO(loc_end)
    # GET_REG(0), PUSH int(1), ADD, SET_REG(0), JMP(loc_loop)
    # loc_end: RETURN
    bytecode = bytes([
        0x00, 0x00, 0x00, 0x00, 0x40,  # 0x00: PUSH int(0)
        0x0A, 0x00,                     # 0x05: SET_REG(0)
        # loc_loop (0x07)
        0x09, 0x00,                     # 0x07: GET_REG(0)
        0x00, 0x0A, 0x00, 0x00, 0x40,  # 0x09: PUSH int(10)
        0x19,                           # 0x0E: LT (i < 10)
        0x0F, 0x23, 0x00, 0x00, 0x00,  # 0x0F: POP_JMP_ZERO(0x23) -> exit loop
        # loop body
        0x09, 0x00,                     # 0x14: GET_REG(0)
        0x00, 0x01, 0x00, 0x00, 0x40,  # 0x16: PUSH int(1)
        0x10,                           # 0x1B: ADD
        0x0A, 0x00,                     # 0x1C: SET_REG(0)
        0x0B, 0x07, 0x00, 0x00, 0x00,  # 0x1E: JMP(0x07) -> back to loop
        # loc_23 (loop exit)
        0x0D,                           # 0x23: RETURN
    ])

    # Disassemble
    context = DisassemblerContext()
    disasm = Disassembler(ED9_INSTRUCTION_TABLE, context)

    entry = disasm.disasm_function(bytecode, offset = 0, name = 'loop_test')

    # Print result
    print_block(entry)
    print()


def test_jump_into_middle():
    """Test jumping into the middle of a block (should split the block)"""
    print('=== Test: Jump Into Middle of Block ===\n')

    # Backward jump into the middle of a linear block
    # Actual instruction offsets (calculated based on real sizes):
    # PUSH = 6 bytes (0x00 opcode + 0x00 size + 4-byte value)
    # SET_REG = 2 bytes (0x0A + reg)
    # GET_REG = 2 bytes (0x09 + reg)
    # LT = 1 byte (0x19)
    # ADD = 1 byte (0x10)
    # POP_JMP_ZERO = 5 bytes (0x0F + 4-byte offset)
    # JMP = 5 bytes (0x0B + 4-byte offset)
    # RETURN = 1 byte (0x0D)
    bytecode = bytes([
        # Entry block (0x00-0x0F)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,  # 0x00: PUSH int(0) [6 bytes]
        0x0A, 0x00,                           # 0x06: SET_REG(0) [2 bytes]
        0x00, 0x00, 0x01, 0x00, 0x00, 0x40,  # 0x08: PUSH int(1) [6 bytes]
        0x0A, 0x01,                           # 0x0E: SET_REG(1) [2 bytes]

        # Loop header - split point (0x10-)
        0x09, 0x01,                           # 0x10: GET_REG(1) <- Backward jump target!
        0x00, 0x00, 0x0A, 0x00, 0x00, 0x40,  # 0x12: PUSH int(10)
        0x19,                                 # 0x18: LT
        0x0F, 0x2C, 0x00, 0x00, 0x00,        # 0x19: POP_JMP_ZERO(0x2C) -> exit

        # Loop body (fallthrough from POP_JMP_ZERO)
        0x09, 0x01,                           # 0x1E: GET_REG(1)
        0x00, 0x00, 0x01, 0x00, 0x00, 0x40,  # 0x20: PUSH int(1)
        0x10,                                 # 0x26: ADD
        0x0B, 0x10, 0x00, 0x00, 0x00,        # 0x27: JMP(0x10) -> backward jump!

        # Exit
        0x0D,                                 # 0x2C: RETURN
    ])

    # Disassemble
    from falcom.ed9.disasm.ed9_optable import ed9_create_fallthrough_jump
    context = DisassemblerContext(
        create_fallthrough_jump = ed9_create_fallthrough_jump,
    )

    disasm = Disassembler(ED9_INSTRUCTION_TABLE, context)

    entry = disasm.disasm_function(bytecode, offset=0, name='test_split')

    # Print result
    print('Expected behavior:')
    print('  - Initial linear block should be split at 0x10')
    print('  - Block starting at 0x10 should exist (loop header)')
    print('  - Backward JMP(0x10) should target the split block\n')
    # Use formatter with context
    formatter_context = FormatterContext(
    )

    formatter = Formatter(formatter_context)
    lines = formatter.format_entry_block(entry)

    # Add function header manually for demo
    func_name = entry.name or f'func_{entry.offset:X}'
    print(f'def {func_name}():')
    print('\n'.join(lines))


def test_scp_parser():
    DAT_PATH = Path(__file__).parent.parent / 'tests'
    test_file = DAT_PATH / 'mp3010_01.dat'

    with fileio.FileStream(str(test_file), encoding = default_encoding()) as fs:
        parser = ScpParser(fs, test_file.name)
        parser.parse()

        # Disassemble all functions
        disassembled_functions = parser.disasm_all_functions(
            filter_func = lambda f: f.name == 'Init'
        )

        # Format and print
        for func in disassembled_functions:
            lines = parser.format_function(func)
            print('\n'.join(lines))


def main():
    # test_simple_function()
    # test_conditional()
    # test_caller_frame()
    # test_loop()
    # test_jump_into_middle()
    test_scp_parser()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Disassembler Demo - Test the ED9 disassembler
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from falcom.ed9.disasm import *


def print_block(block, visited=None, indent=0, all_blocks=None):
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
                    print(f'  {inst.offset:08X}: {inst}')
                if b.branches:
                    branch_names = ', '.join(br.name for br in b.branches)
                    print(f'  → {branch_names}')
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
        print(f'{"  " * indent}  {inst.offset:08X}: {inst}')

    if block.branches:
        print(f'{"  " * indent}  Branches:')
        for branch in block.branches:
            print_block(branch, visited, indent + 1, all_blocks)


def collect_blocks(block, result, visited):
    """Collect all blocks in the CFG"""
    if block.offset in visited:
        return
    visited.add(block.offset)
    result.append(block)
    for branch in block.branches:
        collect_blocks(branch, result, visited)


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
    disasm = Disassembler(ED9_INSTRUCTION_TABLE)

    entry = disasm.disasm_function(bytecode, offset=0, name='loop_test')

    # Print result
    print_block(entry)
    print()


def main():
    # test_simple_function()
    # test_conditional()
    # test_caller_frame()
    test_loop()  # TODO: Fix backward jump handling


if __name__ == '__main__':
    main()

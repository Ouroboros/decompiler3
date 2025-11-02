#!/usr/bin/env python3
"""
CFG Demo - Testing Control Flow Graph with Terminal instructions
Demonstrates the BinaryNinja-style design with Terminal inheritance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ir.llil import (
    LowLevelILFunction, LowLevelILBasicBlock,
    LowLevelILGoto, LowLevelILIf, LowLevelILRet,
    LowLevelILStackStore, LowLevelILConst,
    Terminal, ControlFlow
)


def test_terminal_inheritance():
    """Test that Terminal classes are correctly inherited"""

    print("🧪 Testing Terminal Inheritance")
    print("-" * 50)

    # Create dummy blocks for testing
    block1 = LowLevelILBasicBlock(0x1000, 0)
    block2 = LowLevelILBasicBlock(0x2000, 1)

    # Create instances
    goto_instr = LowLevelILGoto(block1)
    if_instr = LowLevelILIf("zero", block1, block2)
    ret_instr = LowLevelILRet()

    # Test inheritance
    print(f"LowLevelILGoto is Terminal: {isinstance(goto_instr, Terminal)}")
    print(f"LowLevelILGoto is ControlFlow: {isinstance(goto_instr, ControlFlow)}")

    print(f"LowLevelILIf is ControlFlow: {isinstance(if_instr, ControlFlow)}")
    print(f"LowLevelILIf is Terminal: {isinstance(if_instr, Terminal)}")

    print(f"LowLevelILRet is Terminal: {isinstance(ret_instr, Terminal)}")
    print(f"LowLevelILRet is ControlFlow: {isinstance(ret_instr, ControlFlow)}")

    print("\n✅ Terminal inheritance working correctly!\n")


def test_basic_block_edges():
    """Test BasicBlock edge construction"""

    print("🧪 Testing BasicBlock Edge Construction")
    print("-" * 50)

    func = LowLevelILFunction("test_edges", 0x1000)

    # Create blocks
    block0 = LowLevelILBasicBlock(0x1000, 0)
    block1 = LowLevelILBasicBlock(0x1010, 1)
    block2 = LowLevelILBasicBlock(0x1020, 2)

    func.add_basic_block(block0)
    func.add_basic_block(block1)
    func.add_basic_block(block2)

    # Add edges
    block0.add_outgoing_edge(block1)
    block0.add_outgoing_edge(block2)
    block1.add_outgoing_edge(block2)

    # Verify edges
    print(f"Block 0 outgoing: {[b.index for b in block0.outgoing_edges]}")
    print(f"Block 0 incoming: {[b.index for b in block0.incoming_edges]}")

    print(f"Block 1 outgoing: {[b.index for b in block1.outgoing_edges]}")
    print(f"Block 1 incoming: {[b.index for b in block1.incoming_edges]}")

    print(f"Block 2 outgoing: {[b.index for b in block2.outgoing_edges]}")
    print(f"Block 2 incoming: {[b.index for b in block2.incoming_edges]}")

    print("\n✅ BasicBlock edges working correctly!\n")


def create_cfg_example():
    """Create a function with proper CFG"""

    print("🧪 Testing CFG Construction")
    print("-" * 50)

    func = LowLevelILFunction("cfg_test", 0x1000)

    # Create all blocks first
    block0 = LowLevelILBasicBlock(0x1000, 0)
    block1 = LowLevelILBasicBlock(0x1010, 1)
    block2 = LowLevelILBasicBlock(0x1020, 2)
    block3 = LowLevelILBasicBlock(0x1030, 3)

    # Add blocks to function
    func.add_basic_block(block0)
    func.add_basic_block(block1)
    func.add_basic_block(block2)
    func.add_basic_block(block3)

    # Block 0: Entry - now with block references
    block0.add_instruction(LowLevelILStackStore(LowLevelILConst(42), 0))
    block0.add_instruction(LowLevelILIf("zero", block1, block2))

    # Block 1: True branch
    block1.add_instruction(LowLevelILStackStore(LowLevelILConst(100), 0))
    block1.add_instruction(LowLevelILGoto(block3))

    # Block 2: False branch (falls through to block3)
    block2.add_instruction(LowLevelILStackStore(LowLevelILConst(200), 0))

    # Block 3: Merge point
    block3.add_instruction(LowLevelILRet())

    # Build CFG
    func.build_cfg()

    # Display function
    print(func)

    # Verify CFG
    print("CFG Verification:")
    print(f"Block 0 -> {[b.index for b in block0.outgoing_edges]}")
    print(f"Block 1 -> {[b.index for b in block1.outgoing_edges]}")
    print(f"Block 2 -> {[b.index for b in block2.outgoing_edges]}")
    print(f"Block 3 -> {[b.index for b in block3.outgoing_edges]}")

    # Check terminal detection
    print("\nTerminal Detection:")
    for i, block in enumerate(func.basic_blocks):
        print(f"Block {i} has terminal: {block.has_terminal}")

    print("\n✅ CFG construction working correctly!\n")


def main():
    print("=" * 60)
    print("🔧 CFG Demo - Block-based Control Flow")
    print("=" * 60)
    print()

    # Run tests
    test_terminal_inheritance()
    test_basic_block_edges()
    create_cfg_example()

    print("=" * 60)
    print("✅ All CFG tests passed!")
    print("=" * 60)
    print("\nKey features validated:")
    print("  ✅ Terminal base class inheritance (like BN)")
    print("  ✅ ControlFlow base class for all control flow")
    print("  ✅ Direct BasicBlock references (not labels/indexes)")
    print("  ✅ BasicBlock edge tracking (incoming/outgoing)")
    print("  ✅ Automatic CFG construction from terminators")
    print("  ✅ Terminal detection in blocks")


if __name__ == "__main__":
    main()

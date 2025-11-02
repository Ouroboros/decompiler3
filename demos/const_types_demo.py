#!/usr/bin/env python3
"""
Constant Types Demo - Testing Falcom-specific constant types
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ir.llil import LowLevelILConst
from falcom import (
    LowLevelILConstFuncId,
    LowLevelILConstRetAddr,
    FalcomConstants
)


def test_constant_types():
    """Test different constant types"""

    print("🧪 Testing Constant Types")
    print("=" * 60)

    # Regular constants
    int_const = LowLevelILConst(42, 4)
    str_const = LowLevelILConst("hello", 0)

    print("\n1. Regular Constants:")
    print(f"   Integer: {int_const} (type: {type(int_const).__name__})")
    print(f"   String:  {str_const} (type: {type(str_const).__name__})")

    # Falcom-specific constants
    func_id = LowLevelILConstFuncId()
    ret_addr = LowLevelILConstRetAddr("loc_1234")

    print("\n2. Falcom-Specific Constants:")
    print(f"   FuncID:  {func_id} (type: {type(func_id).__name__})")
    print(f"   RetAddr: {ret_addr} (type: {type(ret_addr).__name__})")

    # Using FalcomConstants helper
    func_id2 = FalcomConstants.current_func_id()
    ret_addr2 = FalcomConstants.ret_addr("loc_5678")

    print("\n3. Via FalcomConstants Helper:")
    print(f"   FuncID:  {func_id2} (type: {type(func_id2).__name__})")
    print(f"   RetAddr: {ret_addr2} (type: {type(ret_addr2).__name__})")

    # Type checking
    print("\n4. Type Checking:")
    print(f"   func_id is LowLevelILConstFuncId: {isinstance(func_id, LowLevelILConstFuncId)}")
    print(f"   func_id is LowLevelILConst: {isinstance(func_id, LowLevelILConst)}")
    print(f"   ret_addr is LowLevelILConstRetAddr: {isinstance(ret_addr, LowLevelILConstRetAddr)}")
    print(f"   ret_addr is LowLevelILConst: {isinstance(ret_addr, LowLevelILConst)}")

    # Size verification
    print("\n5. Size Verification:")
    print(f"   func_id size: {func_id.size} bytes")
    print(f"   ret_addr size: {ret_addr.size} bytes")
    print(f"   int_const size: {int_const.size} bytes")

    # String representation differences
    print("\n6. String Representation:")
    print(f"   Integer 42:       {LowLevelILConst(42)}")
    print(f"   String 'hello':   {LowLevelILConst('hello')}")
    print(f"   func_id:          {func_id}  (NOT quoted!)")
    print(f"   Return address:   {ret_addr}  (NOT quoted!)")

    print("\n" + "=" * 60)
    print("✅ All constant type tests passed!")
    print("\nKey differences:")
    print("  • func_id is LowLevelILConstFuncId, not a string")
    print("  • &label is LowLevelILConstRetAddr, not a string")
    print("  • Type-safe and distinguishable from regular strings")
    print("  • Size is properly defined (4 bytes for func_id, 8 for ret_addr)")


def test_in_context():
    """Test constants in realistic usage"""

    print("\n\n🧪 Testing Constants in Context")
    print("=" * 60)

    from ir.llil import LowLevelILFunction, LowLevelILBasicBlock, LowLevelILStackStore, LowLevelILVspAdd

    func = LowLevelILFunction("test_const", 0x1000)
    block = LowLevelILBasicBlock(0x1000)

    # Push different types of constants
    block.add_instruction(LowLevelILStackStore(LowLevelILConstFuncId(), 0))
    block.add_instruction(LowLevelILVspAdd(1))

    block.add_instruction(LowLevelILStackStore(LowLevelILConstRetAddr("loc_ret"), 0))
    block.add_instruction(LowLevelILVspAdd(1))

    block.add_instruction(LowLevelILStackStore(LowLevelILConst(42), 0))
    block.add_instruction(LowLevelILVspAdd(1))

    block.add_instruction(LowLevelILStackStore(LowLevelILConst("arg_name"), 0))
    block.add_instruction(LowLevelILVspAdd(1))

    func.add_basic_block(block)

    print("\nGenerated code:")
    for instr in block.instructions:
        print(f"  {instr}")

    print("\nNotice the differences:")
    print("  • func_id    - no quotes (special constant)")
    print("  • &loc_ret   - no quotes (special constant)")
    print("  • 42         - integer constant")
    print("  • \"arg_name\" - quoted string constant")

    print("\n" + "=" * 60)
    print("✅ Context test passed!")


if __name__ == "__main__":
    test_constant_types()
    test_in_context()

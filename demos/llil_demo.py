#!/usr/bin/env python3
'''
LLIL Demo - Expression-based Architecture
Demonstrates real game functions from Kuro no Kiseki
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ir.llil import *
from ir.llil_builder import LLILFormatter
from falcom import FalcomVMBuilder


def create_AV_04_0017():
    '''
    AV_04_0017 - Strict 1:1 translation from real game bytecode
    Source: m4000.py from Kuro no Kiseki

    Original bytecode:
    # id: 0x0000 offset: 0x243C5
    @scena.Code('AV_04_0017')
    def AV_04_0017():
        DEBUG_SET_LINENO(2519)
        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_243E3')
        PUSH_INT(0)
        PUSH_STR('AV_04_0017')
        CALL(map_event_box_set_enable)

        label('loc_243E3')
        DEBUG_SET_LINENO(2520)
        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_243FB')
        PUSH_INT(427)
        CALL(avoice_play)

        label('loc_243FB')
        DEBUG_SET_LINENO(2521)
        PUSH(0x00000000)
        SET_REG(0)
        RETURN()
    '''

    function = LowLevelILFunction('AV_04_0017', 0x243C5)

    # Create all blocks upfront
    entry_block = LowLevelILBasicBlock(0x243C5, 0)
    loc_243E3 = LowLevelILBasicBlock(0x243E3, 1)
    loc_243FB = LowLevelILBasicBlock(0x243FB, 2)

    function.add_basic_block(entry_block)
    function.add_basic_block(loc_243E3)
    function.add_basic_block(loc_243FB)

    builder = FalcomVMBuilder(function)

    # === BLOCK 0: Entry - map_event_box_set_enable call ===
    builder.set_current_block(entry_block, sp = 0)
    builder.label('AV_04_0017')

    # DEBUG_SET_LINENO(2519)
    builder.debug_line(2519)
    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_243E3')
    builder.push_ret_addr('loc_243E3')
    # PUSH_INT(0)
    builder.push_int(0)
    # PUSH_STR('AV_04_0017')
    builder.push_str('AV_04_0017')
    # CALL(map_event_box_set_enable)
    builder.call('map_event_box_set_enable')
    # Falls through to BLOCK 1

    # === BLOCK 1: loc_243E3 - avoice_play call ===
    builder.set_current_block(loc_243E3)
    builder.label('loc_243E3')

    # DEBUG_SET_LINENO(2520)
    builder.debug_line(2520)
    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_243FB')
    builder.push_ret_addr('loc_243FB')
    # PUSH_INT(427)
    builder.push_int(427)
    # CALL(avoice_play)
    builder.call('avoice_play')
    # Falls through to BLOCK 2

    # === BLOCK 2: loc_243FB - Return ===
    builder.set_current_block(loc_243FB)
    builder.label('loc_243FB')

    # DEBUG_SET_LINENO(2521)
    builder.debug_line(2521)
    # PUSH(0x00000000)
    builder.push_int(0)
    # SET_REG(0)
    builder.set_reg(0)
    # RETURN()
    builder.ret()

    return function


def create_DOF_ON():
    '''
    DOF_ON - Strict 1:1 translation from real game bytecode
    Source: c0000.py from Kuro no Kiseki

    Original bytecode:
    # id: 0x003F offset: 0x1FFDB6
    @scena.Code('DOF_ON')
    def DOF_ON(arg1, arg2 = 0):
        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_1FFDCB')
        PUSH_INT(1)
        CALL(screen_dof_set_enable)

        label('loc_1FFDCB')
        LOAD_STACK(-8)
        PUSH_INT(0)
        EQ()
        POP_JMP_ZERO('loc_1FFE07')

        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_1FFE02')
        LOAD_STACK(-12)
        LOAD_STACK(-16)
        PUSH_FLOAT(0.1)
        MUL()
        ADD()
        LOAD_STACK(-16)
        CALL(screen_dof_set_focus_range)

        label('loc_1FFE02')
        JMP('loc_1FFE20')

        label('loc_1FFE07')
        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_1FFE20')
        LOAD_STACK(-12)
        LOAD_STACK(-20)
        CALL(screen_dof_set_focus_range)

        label('loc_1FFE20')
        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_1FFE35')
        PUSH_INT(3)
        CALL(screen_dof_set_blur_level)

        label('loc_1FFE35')
        PUSH(0x00000000)
        SET_REG(0)
        POP(8)
        RETURN()
    '''

    function = LowLevelILFunction('DOF_ON', 0x1FFDB6)

    # Create all blocks upfront
    entry_block = LowLevelILBasicBlock(0x1FFDB6, 0)
    loc_1FFDCB = LowLevelILBasicBlock(0x1FFDCB, 1)
    nonzero_path = LowLevelILBasicBlock(0x1FFDD8, 2)  # Implicit block after POP_JMP_ZERO
    loc_1FFE02 = LowLevelILBasicBlock(0x1FFE02, 3)
    loc_1FFE07 = LowLevelILBasicBlock(0x1FFE07, 4)    # Zero branch target
    loc_1FFE20 = LowLevelILBasicBlock(0x1FFE20, 5)    # Merge point
    loc_1FFE35 = LowLevelILBasicBlock(0x1FFE35, 6)

    function.add_basic_block(entry_block)
    function.add_basic_block(loc_1FFDCB)
    function.add_basic_block(nonzero_path)
    function.add_basic_block(loc_1FFE02)
    function.add_basic_block(loc_1FFE07)
    function.add_basic_block(loc_1FFE20)
    function.add_basic_block(loc_1FFE35)

    builder = FalcomVMBuilder(function)

    # === BLOCK 0: Entry - Enable DOF ===
    # DOF_ON has 2 parameters (arg1, arg2), so sp starts at 2
    builder.set_current_block(entry_block, sp = 2)
    builder.label('DOF_ON')

    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_1FFDCB')
    builder.push_ret_addr('loc_1FFDCB')
    # PUSH_INT(1)
    builder.push_int(1)
    # CALL(screen_dof_set_enable)
    builder.call('screen_dof_set_enable')
    # Falls through to BLOCK 1

    # === BLOCK 1: loc_1FFDCB - Check arg2 == 0 ===
    builder.set_current_block(loc_1FFDCB)
    builder.label('loc_1FFDCB')

    # LOAD_STACK(-8)
    builder.load_stack(-8)
    # PUSH_INT(0)
    builder.push_int(0)
    # EQ()
    builder.eq()
    # POP_JMP_ZERO('loc_1FFE07')
    builder.pop_jmp_zero(loc_1FFE07)

    # === BLOCK 2: Nonzero path - Calculate focus range ===
    builder.set_current_block(nonzero_path)

    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_1FFE02')
    builder.push_ret_addr('loc_1FFE02')
    # LOAD_STACK(-12)
    builder.load_stack(-12)
    # LOAD_STACK(-16)
    builder.load_stack(-16)
    # PUSH_FLOAT(0.1)
    builder.push(builder.const_float(0.1))
    # MUL()
    builder.mul()
    # ADD()
    builder.add()
    # LOAD_STACK(-16)
    builder.load_stack(-16)
    # CALL(screen_dof_set_focus_range)
    builder.call('screen_dof_set_focus_range')
    # Falls through to BLOCK 3

    # === BLOCK 3: loc_1FFE02 - Jump to merge point ===
    builder.set_current_block(loc_1FFE02)
    builder.label('loc_1FFE02')

    # JMP('loc_1FFE20')
    builder.jmp(loc_1FFE20)

    # === BLOCK 4: loc_1FFE07 - Zero path, simple focus range ===
    builder.set_current_block(loc_1FFE07)
    builder.label('loc_1FFE07')

    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_1FFE20')
    builder.push_ret_addr('loc_1FFE20')
    # LOAD_STACK(-12)
    builder.load_stack(-12)
    # LOAD_STACK(-20)
    builder.load_stack(-20)
    # CALL(screen_dof_set_focus_range)
    builder.call('screen_dof_set_focus_range')
    # Falls through to BLOCK 5

    # === BLOCK 5: loc_1FFE20 - Merge point, set blur level ===
    builder.set_current_block(loc_1FFE20)
    builder.label('loc_1FFE20')

    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_1FFE35')
    builder.push_ret_addr('loc_1FFE35')
    # PUSH_INT(3)
    builder.push_int(3)
    # CALL(screen_dof_set_blur_level)
    builder.call('screen_dof_set_blur_level')
    # Falls through to BLOCK 6

    # === BLOCK 6: loc_1FFE35 - Return ===
    builder.set_current_block(loc_1FFE35)
    builder.label('loc_1FFE35')

    # PUSH(0x00000000)
    builder.push_int(0)
    # SET_REG(0)
    builder.set_reg(0)
    # POP(8) - pops 8 bytes from stack
    builder.add_instruction(LowLevelILSpAdd(-2))  # 8 bytes = 2 words
    # RETURN()
    builder.ret()

    return function


def create_sound_play_se():
    '''
    sound_play_se - Simple function call demo
    Demonstrates basic function call pattern with single argument

    Original bytecode concept:
    def sound_play_se(se_id):
        PUSH_CURRENT_FUNC_ID()
        PUSH_RET_ADDR('loc_return')
        LOAD_STACK(-4)  # Load se_id parameter
        CALL(sound_play)

        label('loc_return')
        PUSH(0)
        SET_REG(0)
        POP(4)  # Clean up parameter
        RETURN()
    '''

    function = LowLevelILFunction('sound_play_se', 0x10000)

    # Create blocks
    entry_block = LowLevelILBasicBlock(0x10000, 0)
    loc_return = LowLevelILBasicBlock(0x10020, 1)

    function.add_basic_block(entry_block)
    function.add_basic_block(loc_return)

    builder = FalcomVMBuilder(function)

    # === BLOCK 0: Entry - sound_play call ===
    # Function has 1 parameter (se_id), so sp starts at 1
    builder.set_current_block(entry_block, sp = 1)
    builder.label('sound_play_se')

    # PUSH_CURRENT_FUNC_ID()
    builder.push_func_id()
    # PUSH_RET_ADDR('loc_return')
    builder.push_ret_addr('loc_return')
    # LOAD_STACK(-4) - Load se_id parameter from stack
    builder.load_stack(-4)
    # CALL(sound_play)
    builder.call('sound_play')

    # === BLOCK 1: Return ===
    builder.set_current_block(loc_return)
    builder.label('loc_return')

    # PUSH(0)
    builder.push_int(0)
    # SET_REG(0)
    builder.set_reg(0)
    # POP(4) - Clean up parameter (1 word = 4 bytes)
    builder.add_instruction(LowLevelILSpAdd(-1))
    # RETURN()
    builder.ret()

    return function


def main():
    print('🔧 LLIL Demo - Expression-based Architecture')
    print('=' * 60)
    print('Source: Real game functions from Kuro no Kiseki')
    print()
    print('Features:')
    print('  ✓ First-class PUSH/POP instructions (no pattern matching)')
    print('  ✓ Virtual stack tracks expressions')
    print('  ✓ Operations hold operands: EQ(lhs, rhs)')
    print('  ✓ Data flow visible for optimization')

    # Test 1: AV_04_0017 - Simple linear function
    print('\n🧪 Test 1: AV_04_0017 - Simple Linear Function')
    print('-' * 60)
    print('Source: m4000.py (id: 0x0000 offset: 0x243C5)')
    print('3 blocks, no branching, 2 function calls')

    func1 = create_AV_04_0017()
    print('\n' + '\n'.join(LLILFormatter.format_llil_function(func1)))

    # Test 2: DOF_ON - Complex control flow
    print('\n🧪 Test 2: DOF_ON - Complex Control Flow')
    print('-' * 60)
    print('Source: c0000.py (id: 0x003F offset: 0x1FFDB6)')
    print('7 blocks, conditional branching, merge points')

    func2 = create_DOF_ON()
    print('\n' + '\n'.join(LLILFormatter.format_llil_function(func2)))

    # Test 3: sound_play_se - Simple function with parameter
    print('\n🧪 Test 3: sound_play_se - Function with Parameter')
    print('-' * 60)
    print('Concept: Simple sound effect playback')
    print('1 parameter, 2 blocks, demonstrates parameter access')

    func3 = create_sound_play_se()
    print('\n' + '\n'.join(LLILFormatter.format_llil_function(func3)))

    print('\n✅ Demo completed successfully!')
    print('\nKey features demonstrated:')
    print('  ✅ First-class PUSH/POP: STACK[sp++] = value, STACK[--sp]')
    print('  ✅ No pattern matching fragility')
    print('  ✅ Expression tracking in operands: EQ(STACK[--sp], STACK[--sp])')
    print('  ✅ Clean separation: Push/Pop vs Load/Store')
    print('  ✅ Stack state tracking: [sp = N] in each block')
    print('  ✅ Parameter access: LOAD_STACK with negative offset')


if __name__ == '__main__':
    main()

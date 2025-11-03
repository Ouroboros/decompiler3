#!/usr/bin/env python3
'''Test validation logic for Falcom VM builder state management'''

from ir.llil import LowLevelILFunction, LowLevelILBasicBlock
from falcom.builder import FalcomVMBuilder


def test_missing_push_func_id():
    '''Test: call push_ret_addr without push_func_id'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)
    builder.set_current_block(entry)

    try:
        builder.push_ret_addr('loc_ret')
        print('❌ FAILED: Should have raised RuntimeError for missing push_func_id')
        return False
    except RuntimeError as e:
        if 'push_func_id' in str(e):
            print(f'✅ PASSED: Caught missing push_func_id - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_missing_push_ret_addr():
    '''Test: call() without push_ret_addr'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)
    builder.set_current_block(entry)

    builder.push_func_id()

    try:
        builder.call('some_func')
        print('❌ FAILED: Should have raised RuntimeError for missing push_ret_addr')
        return False
    except RuntimeError as e:
        if 'push_ret_addr' in str(e):
            print(f'✅ PASSED: Caught missing push_ret_addr - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_double_push_func_id():
    '''Test: call push_func_id twice without call in between'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)
    builder.set_current_block(entry)

    builder.push_func_id()

    try:
        builder.push_func_id()  # Second call without completing first
        print('❌ FAILED: Should have raised RuntimeError for duplicate push_func_id')
        return False
    except RuntimeError as e:
        if 'Previous call setup' in str(e):
            print(f'✅ PASSED: Caught duplicate push_func_id - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_double_push_ret_addr():
    '''Test: call push_ret_addr twice without call in between'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)
    builder.set_current_block(entry)

    builder.push_func_id()
    builder.push_ret_addr('loc1')

    try:
        builder.push_ret_addr('loc2')  # Second call without completing first
        print('❌ FAILED: Should have raised RuntimeError for duplicate push_ret_addr')
        return False
    except RuntimeError as e:
        if 'Previous return target' in str(e):
            print(f'✅ PASSED: Caught duplicate push_ret_addr - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_call_without_push_func_id():
    '''Test: call() with push_ret_addr but no push_func_id'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)
    builder.set_current_block(entry)

    # This should fail at push_ret_addr already
    try:
        builder.push_ret_addr('loc')
        print('❌ FAILED: Should have raised RuntimeError at push_ret_addr')
        return False
    except RuntimeError as e:
        if 'push_func_id' in str(e):
            print(f'✅ PASSED: Caught missing push_func_id at push_ret_addr - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_correct_sequence():
    '''Test: correct sequence should work'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    ret_block = LowLevelILBasicBlock(0, 'ret_block', label = 'loc_ret')
    func.add_basic_block(entry)
    func.add_basic_block(ret_block)

    builder.set_current_block(entry)

    try:
        builder.push_func_id()
        builder.push_ret_addr('loc_ret')
        builder.call('some_func')
        builder.finalize()
        print('✅ PASSED: Correct sequence works')
        return True
    except Exception as e:
        print(f'❌ FAILED: Correct sequence raised error - {e}')
        return False


def test_undefined_label():
    '''Test: call with undefined return label'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)

    builder.set_current_block(entry)

    try:
        builder.push_func_id()
        builder.push_ret_addr('undefined_label')  # Label not registered
        builder.call('some_func')
        print('❌ FAILED: Should have raised RuntimeError for undefined label')
        return False
    except RuntimeError as e:
        if 'cannot be resolved' in str(e):
            print(f'✅ PASSED: Caught undefined label - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_sp_mismatch():
    '''Test: sp mismatch when return block already built'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    ret_block = LowLevelILBasicBlock(0, 'ret_block', label = 'loc_ret')
    func.add_basic_block(entry)
    func.add_basic_block(ret_block)

    # Build entry block
    builder.set_current_block(entry)
    builder.push_func_id()  # sp_before_call = 0
    builder.push_ret_addr('loc_ret')
    # Don't call yet

    # Build return block first with different sp
    builder.set_current_block(ret_block, sp = 5)  # sp = 5, not 0!
    builder.push_int(0)

    # Now try to call from entry
    builder.set_current_block(entry)
    try:
        builder.call('some_func')  # Tries to restore sp to 0, but ret_block has sp_in=5
        print('❌ FAILED: Should have raised RuntimeError for sp mismatch')
        return False
    except RuntimeError as e:
        if 'Stack pointer mismatch' in str(e):
            print(f'✅ PASSED: Caught sp mismatch - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_pending_func_id():
    '''Test: function ends with pending push_func_id'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    func.add_basic_block(entry)

    builder.set_current_block(entry)
    builder.push_func_id()
    # Don't complete the call

    try:
        builder.finalize()
        print('❌ FAILED: Should have raised RuntimeError for pending func_id')
        return False
    except RuntimeError as e:
        if 'pending call setup' in str(e):
            print(f'✅ PASSED: Caught pending func_id - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


def test_pending_ret_addr():
    '''Test: function ends with pending push_ret_addr'''
    func = LowLevelILFunction('test', num_params = 0)
    builder = FalcomVMBuilder(func)

    entry = LowLevelILBasicBlock(0, 'entry')
    ret_block = LowLevelILBasicBlock(0, 'ret_block', label = 'loc_ret')
    func.add_basic_block(entry)
    func.add_basic_block(ret_block)

    builder.set_current_block(entry)

    builder.push_func_id()
    builder.push_ret_addr('loc_ret')
    # Don't call

    try:
        builder.finalize()
        print('❌ FAILED: Should have raised RuntimeError for pending state')
        return False
    except RuntimeError as e:
        # Either error is fine - both states are pending
        if 'pending call setup' in str(e) or 'pending return target' in str(e):
            print(f'✅ PASSED: Caught pending state - {e}')
            return True
        else:
            print(f'❌ FAILED: Wrong error message - {e}')
            return False


if __name__ == '__main__':
    print('🧪 Testing Falcom VM Builder State Validation')
    print('=' * 60)

    tests = [
        test_missing_push_func_id,
        test_missing_push_ret_addr,
        test_double_push_func_id,
        test_double_push_ret_addr,
        test_call_without_push_func_id,
        test_correct_sequence,
        test_undefined_label,
        test_sp_mismatch,
        test_pending_func_id,
        test_pending_ret_addr,
    ]

    results = []
    for test in tests:
        print(f'\n📋 {test.__doc__}')
        results.append(test())

    print('\n' + '=' * 60)
    passed = sum(results)
    total = len(results)
    print(f'📊 Results: {passed}/{total} tests passed')

    if passed == total:
        print('✅ All tests passed!')
    else:
        print(f'❌ {total - passed} test(s) failed')
        exit(1)

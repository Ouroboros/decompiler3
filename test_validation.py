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
    ret_block = LowLevelILBasicBlock(0, 'ret_block')
    func.add_basic_block(entry)
    func.add_basic_block(ret_block)

    builder.set_current_block(entry)
    builder.mark_label('loc_ret', ret_block)

    try:
        builder.push_func_id()
        builder.push_ret_addr('loc_ret')
        builder.call('some_func')
        print('✅ PASSED: Correct sequence works')
        return True
    except Exception as e:
        print(f'❌ FAILED: Correct sequence raised error - {e}')
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

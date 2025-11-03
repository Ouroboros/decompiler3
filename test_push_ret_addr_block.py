#!/usr/bin/env python3
'''Test push_ret_addr with block parameter'''

from falcom.builder import FalcomVMBuilder


def test_push_ret_addr_with_string():
    '''Test: push_ret_addr with string label (original behavior)'''
    builder = FalcomVMBuilder()
    builder.create_function('test', num_params=0)

    entry = builder.create_basic_block(0, 'entry')
    ret_block = builder.create_basic_block(10, 'loc_ret')

    builder.set_current_block(entry)
    builder.push_func_id()
    builder.push_ret_addr('loc_ret')  # String label
    builder.call('some_func')

    func = builder.finalize()
    print('✅ PASSED: push_ret_addr with string label works')
    return True


def test_push_ret_addr_with_block():
    '''Test: push_ret_addr with block reference (new behavior)'''
    builder = FalcomVMBuilder()
    builder.create_function('test', num_params=0)

    entry = builder.create_basic_block(0, 'entry')
    ret_block = builder.create_basic_block(10, 'loc_ret')

    builder.set_current_block(entry)
    builder.push_func_id()
    builder.push_ret_addr(ret_block)  # Block reference instead of string
    builder.call('some_func')

    func = builder.finalize()
    print('✅ PASSED: push_ret_addr with block reference works')
    return True


if __name__ == '__main__':
    print('🧪 Testing push_ret_addr with block parameter')
    print('=' * 60)

    tests = [
        test_push_ret_addr_with_string,
        test_push_ret_addr_with_block,
    ]

    results = []
    for test in tests:
        print(f'\n📋 {test.__doc__}')
        try:
            results.append(test())
        except Exception as e:
            print(f'❌ FAILED: {e}')
            results.append(False)

    print('\n' + '=' * 60)
    passed = sum(results)
    total = len(results)
    print(f'📊 Results: {passed}/{total} tests passed')

    if passed == total:
        print('✅ All tests passed!')
    else:
        print(f'❌ {total - passed} test(s) failed')
        exit(1)

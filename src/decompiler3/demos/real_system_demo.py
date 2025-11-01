#!/usr/bin/env python3
"""
真实系统演示

使用项目内的实际类型和组件，不隐藏任何错误
"""

def test_real_ir_system():
    """测试真实的IR系统 - 复杂示例：结构化斐波那契函数"""
    print("🔧 真实IR系统测试 - 复杂示例")
    print("=" * 40)

    # 直接导入，不隐藏错误
    from decompiler3.ir.base import IRFunction, IRBasicBlock, IRVariable, OperationType, IRType
    from decompiler3.ir.hlil import (HLILConstant, HLILBinaryOp, HLILReturn, HLILVariable,
                                    HLILAssignment, HLILCall, HLILIf, HLILBuiltinCall, HLILWhile)

    print("✅ 成功导入真实IR类型")

    # 创建复杂函数：fibonacci(n): number - 结构化版本
    function = IRFunction("fibonacci", 0x1000)
    function.return_type = IRType.NUMBER

    # 创建参数
    n_param = function.create_variable("n", 4, IRType.NUMBER)
    function.parameters.append(n_param)

    # 创建局部变量
    result_var = function.create_variable("result", 4, IRType.NUMBER)
    a_var = function.create_variable("a", 4, IRType.NUMBER)
    b_var = function.create_variable("b", 4, IRType.NUMBER)
    i_var = function.create_variable("i", 4, IRType.NUMBER)
    temp_var = function.create_variable("temp", 4, IRType.NUMBER)

    # 创建单个基本块（结构化控制流）
    main_block = IRBasicBlock(0x1000)
    function.basic_blocks.append(main_block)

    # === 创建结构化的HLIL表达式 ===

    # 常量定义
    const_0 = HLILConstant(0, 4, IRType.NUMBER)
    const_1 = HLILConstant(1, 4, IRType.NUMBER)
    const_2 = HLILConstant(2, 4, IRType.NUMBER)

    # 变量引用
    n_ref = HLILVariable(n_param, IRType.NUMBER)
    result_ref = HLILVariable(result_var, IRType.NUMBER)
    a_ref = HLILVariable(a_var, IRType.NUMBER)
    b_ref = HLILVariable(b_var, IRType.NUMBER)
    i_ref = HLILVariable(i_var, IRType.NUMBER)
    temp_ref = HLILVariable(temp_var, IRType.NUMBER)

    # === 基本情况检查：if (n <= 1) return n ===
    base_condition = HLILBinaryOp(OperationType.CMP_SLE, n_ref, const_1, 4, IRType.BOOLEAN)
    base_case_return = HLILReturn(HLILVariable(n_param, IRType.NUMBER))

    base_case_if = HLILIf(base_condition, [base_case_return], [])

    # === 循环初始化：a = 0, b = 1, i = 2 ===
    init_a = HLILAssignment(HLILVariable(a_var, IRType.NUMBER), const_0)
    init_b = HLILAssignment(HLILVariable(b_var, IRType.NUMBER), const_1)
    init_i = HLILAssignment(HLILVariable(i_var, IRType.NUMBER), const_2)

    # === 循环体：while (i <= n) ===
    loop_condition = HLILBinaryOp(OperationType.CMP_SLE,
                                  HLILVariable(i_var, IRType.NUMBER),
                                  HLILVariable(n_param, IRType.NUMBER),
                                  4, IRType.BOOLEAN)

    # 循环体内的操作
    # temp = a + b
    fibonacci_add = HLILBinaryOp(OperationType.ADD,
                                 HLILVariable(a_var, IRType.NUMBER),
                                 HLILVariable(b_var, IRType.NUMBER),
                                 4, IRType.NUMBER)
    temp_assign = HLILAssignment(HLILVariable(temp_var, IRType.NUMBER), fibonacci_add)

    # a = b
    shift_a = HLILAssignment(HLILVariable(a_var, IRType.NUMBER),
                           HLILVariable(b_var, IRType.NUMBER))

    # b = temp
    shift_b = HLILAssignment(HLILVariable(b_var, IRType.NUMBER),
                           HLILVariable(temp_var, IRType.NUMBER))

    # i = i + 1
    i_increment = HLILBinaryOp(OperationType.ADD,
                              HLILVariable(i_var, IRType.NUMBER),
                              const_1, 4, IRType.NUMBER)
    i_assign = HLILAssignment(HLILVariable(i_var, IRType.NUMBER), i_increment)

    # 组装循环体
    loop_body = [temp_assign, shift_a, shift_b, i_assign]
    while_loop = HLILWhile(loop_condition, loop_body)

    # === Built-in调用演示 ===
    debug_call = HLILBuiltinCall("debug_print", [HLILVariable(b_var, IRType.NUMBER)], 4, IRType.VOID)

    # === 最终返回 ===
    final_return = HLILReturn(HLILVariable(b_var, IRType.NUMBER))

    # === 将所有指令添加到基本块 ===
    main_block.add_instruction(base_case_if)    # if (n <= 1) return n
    main_block.add_instruction(init_a)          # a = 0
    main_block.add_instruction(init_b)          # b = 1
    main_block.add_instruction(init_i)          # i = 2
    main_block.add_instruction(while_loop)      # while循环
    main_block.add_instruction(debug_call)      # debug输出
    main_block.add_instruction(final_return)    # return b

    # === 输出详细信息 ===
    print(f"✅ 复杂函数: {function.name}")
    print(f"   参数: {[p.name for p in function.parameters]}")
    print(f"   变量: {list(function.variables.keys())}")
    print(f"   基本块数量: {len(function.basic_blocks)}")
    print(f"   返回类型: {function.return_type.to_string()}")
    print(f"   指令数量: {len(main_block.instructions)}")

    print(f"\n💡 关键表达式演示:")
    print(f"   基本情况条件: {base_condition}")
    print(f"   斐波那契计算: {fibonacci_add}")
    print(f"   循环条件: {loop_condition}")
    print(f"   变量递增: {i_increment}")
    print(f"   Built-in调用: {debug_call}")
    print(f"   返回语句: {final_return}")

    print(f"\n🏗️ 控制流结构:")
    print(f"   • if语句 (基本情况)")
    print(f"   • while循环 (迭代计算)")
    print(f"   • 复杂赋值链")
    print(f"   • Built-in函数调用")
    print(f"   • 结构化返回")

    return function, fibonacci_add

def test_real_typescript_generator():
    """测试真实的TypeScript生成器"""
    print("\n🎯 真实TypeScript生成器测试")
    print("=" * 35)

    # 不隐藏导入错误
    from decompiler3.typescript.generator import TypeScriptGenerator

    function, add_expr = test_real_ir_system()

    generator = TypeScriptGenerator("pretty")
    print(f"✅ 创建真实TypeScript生成器: {generator.__class__.__name__}")
    print(f"   样式: {generator.style}")

    # 不隐藏生成错误
    ts_code = generator.generate_function(function)
    print("生成的TypeScript:")
    print(ts_code)

def test_real_target_system():
    """测试真实的目标系统"""
    print("\n🏭 真实目标系统测试")
    print("=" * 25)

    from decompiler3.target.capability import get_target_capability, TargetCapability
    from decompiler3.target.capability import X86Capability, FalcomVMCapability

    print("✅ 成功导入真实目标能力类")

    # 测试真实的目标能力
    x86_cap = get_target_capability("x86")
    falcom_cap = get_target_capability("falcom_vm")

    print(f"✅ X86能力: {type(x86_cap).__name__}")
    print(f"   名称: {x86_cap.name}")
    print(f"   栈机器: {x86_cap.is_stack_machine}")
    print(f"   寄存器类: {list(x86_cap.register_classes.keys())}")
    print(f"   支持操作: {len(x86_cap.supported_operations)}")

    print(f"✅ Falcom VM能力: {type(falcom_cap).__name__}")
    print(f"   名称: {falcom_cap.name}")
    print(f"   栈机器: {falcom_cap.is_stack_machine}")
    print(f"   支持操作: {len(falcom_cap.supported_operations)}")

def test_real_builtin_system():
    """测试真实的Built-in系统"""
    print("\n🔧 真实Built-in系统测试")
    print("=" * 30)

    from decompiler3.builtin.registry import builtin_registry, get_builtin

    print("✅ 成功导入真实Built-in系统")

    # 测试真实的built-in函数
    abs_builtin = get_builtin("abs")
    print(f"✅ abs函数: {type(abs_builtin).__name__}")
    print(f"   签名: {abs_builtin.signature.name}")
    print(f"   参数: {abs_builtin.signature.parameters}")
    print(f"   返回: {abs_builtin.signature.return_type}")
    print(f"   目标映射: {list(abs_builtin.mappings.keys())}")

    # 测试类别
    math_builtins = builtin_registry.list_by_category("math")
    print(f"✅ 数学built-ins: {math_builtins}")

    categories = builtin_registry.get_all_categories()
    print(f"✅ 所有类别: {categories}")

def test_real_instruction_selection():
    """测试真实的指令选择"""
    print("\n⚙️  真实指令选择测试")
    print("=" * 25)

    from decompiler3.target.instruction_selection import MachineInstruction, InstructionSelector

    print("✅ 成功导入真实指令选择类")

    # 创建真实的机器指令
    instr = MachineInstruction("add", ["eax", "ebx"], cost=1)
    print(f"✅ 机器指令: {instr}")
    print(f"   类型: {type(instr).__name__}")
    print(f"   操作码: {instr.opcode}")
    print(f"   操作数: {instr.operands}")
    print(f"   成本: {instr.cost}")

    # 创建指令选择器
    from decompiler3.target.capability import get_target_capability
    x86_cap = get_target_capability("x86")
    selector = InstructionSelector(x86_cap)
    print(f"✅ 指令选择器: {type(selector).__name__}")
    print(f"   目标: {selector.target.name}")

def main():
    """主演示函数 - 不隐藏任何错误"""
    print("🏗️  真实系统演示 - 暴露所有错误")
    print("=" * 50)
    print("所有错误都会直接抛出，不再隐藏")
    print()

    test_real_ir_system()
    test_real_typescript_generator()
    test_real_target_system()
    test_real_builtin_system()
    test_real_instruction_selection()

    print("\n🎉 真实系统演示完成!")

if __name__ == "__main__":
    main()
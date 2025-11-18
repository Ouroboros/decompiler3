# MLIL Implementation Guide

## 概述

MLIL (Medium Level IL) 已经完整实现！这是一个**无栈语义**的中间表示，将 LLIL 的栈操作转换为变量操作。

## 设计亮点

### ✅ 解决的问题

1. **移除了 Expr/Statement 分离** - 所有指令统一继承 `MediumLevelILInstruction`
2. **简化的变量系统** - 非 SSA 形式，易于理解和使用
3. **完整的指令集** - 常量、变量、算术、比较、控制流、函数调用等
4. **完善的 BasicBlock** - 包含所有必要字段（start, label, llil_block 映射等）
5. **对齐 LLIL 设计** - 保持一致的设计哲学

## 文件结构

```
ir/mlil/
├── __init__.py           # 导出所有公共 API
├── mlil.py               # 核心 MLIL 指令定义
├── mlil_builder.py       # MLIL 构建器
├── llil_to_mlil.py       # LLIL → MLIL 转换器（栈消除）
└── mlil_formatter.py     # MLIL 格式化输出
```

## 快速开始

### 基本用法

```python
from ir.llil import LowLevelILFunction
from ir.mlil import translate_llil_to_mlil, format_mlil_function

# 假设你已经有了 LLIL 函数
llil_func: LowLevelILFunction = ...

# 转换为 MLIL
mlil_func = translate_llil_to_mlil(llil_func)

# 格式化输出
lines = format_mlil_function(mlil_func)
for line in lines:
    print(line)
```

### 手动构建 MLIL

```python
from ir.mlil import MLILBuilder

# 创建 builder
builder = MLILBuilder()
builder.create_function('test_func', 0x1000)

# 创建基本块
entry = builder.create_block(0x1000, 'entry')
exit_block = builder.create_block(0x1020, 'exit')

# 设置当前块
builder.set_current_block(entry)

# 创建变量
var_x = builder.get_or_create_var('x')
var_y = builder.get_or_create_var('y')

# 生成代码：x = 10
builder.set_var(var_x, builder.const_int(10))

# 生成代码：y = x + 5
x_val = builder.var(var_x)
add_result = builder.add(x_val, builder.const_int(5))
builder.set_var(var_y, add_result)

# 跳转到 exit
builder.goto(exit_block)

# 设置 exit 块
builder.set_current_block(exit_block)
builder.ret()

# 完成构建
mlil_func = builder.finalize()
```

## 转换示例

### LLIL → MLIL

**LLIL (栈语义):**
```
; sp = 0
STACK[0] = 10                 ; StackStore(Const(10), offset=0, slot_index=0)
sp = sp + 1                   ; SpAdd(+1)

; sp = 1
STACK[1] = 5                  ; StackStore(Const(5), offset=0, slot_index=1)
sp = sp + 1                   ; SpAdd(+1)

; sp = 2
sp = sp - 1                   ; SpAdd(-1)
rhs = STACK[1]                ; (implicit)
sp = sp - 1                   ; SpAdd(-1)
lhs = STACK[0]                ; (implicit)

; sp = 0
result = lhs + rhs            ; Add(lhs, rhs)
STACK[0] = result             ; StackStore(Add(...), offset=0, slot_index=0)
sp = sp + 1                   ; SpAdd(+1)
```

**MLIL (变量语义):**
```
var_s0 = 10                   ; SetVar(var_s0, Const(10))
var_s1 = 5                    ; SetVar(var_s1, Const(5))
var_s0 = (var_s0 + var_s1)    ; SetVar(var_s0, Add(Var(var_s0), Var(var_s1)))
```

**优势显而易见：**
- ❌ LLIL: 7 条指令，需要追踪 sp
- ✅ MLIL: 3 条指令，直接操作变量
- 代码更清晰，更易于分析和优化

## 主要组件

### 1. MLILVariable
```python
class MLILVariable:
    '''MLIL 变量（非 SSA 形式）'''
    def __init__(self, name: str, slot_index: int = -1):
        self.name = name
        self.slot_index = slot_index  # 原始栈槽索引（用于调试）
```

**变量命名规则：**
- `var_s0`, `var_s1`, ... - 栈槽变量（slot_index = 0, 1, ...）
- `param_0`, `param_1`, ... - 函数参数
- 自定义名称 - 用户定义的变量

### 2. 指令类型

**常量：**
- `MLILConst` - 整数/浮点/字符串常量

**变量操作：**
- `MLILVar` - 加载变量值
- `MLILSetVar` - 存储值到变量

**算术操作：**
- `MLILAdd`, `MLILSub`, `MLILMul`, `MLILDiv`, `MLILMod`
- `MLILAnd`, `MLILOr`, `MLILXor`, `MLILShl`, `MLILShr`
- `MLILLogicalAnd`, `MLILLogicalOr`

**比较操作：**
- `MLILEq`, `MLILNe`, `MLILLt`, `MLILLe`, `MLILGt`, `MLILGe`

**一元操作：**
- `MLILNeg`, `MLILLogicalNot`, `MLILTestZero`

**控制流：**
- `MLILGoto` - 无条件跳转
- `MLILIf` - 条件分支
- `MLILRet` - 返回（可选返回值）

**函数调用：**
- `MLILCall` - 普通函数调用
- `MLILSyscall` - 系统调用
- `MLILCallScript` - Falcom 脚本调用

**全局变量/寄存器：**
- `MLILLoadGlobal` / `MLILStoreGlobal`
- `MLILLoadReg` / `MLILStoreReg`

### 3. MLILBuilder

提供便捷的 API 构建 MLIL：

```python
# 创建函数
builder.create_function(name, addr)

# 块管理
block = builder.create_block(start, label)
builder.set_current_block(block)

# 变量
var = builder.get_or_create_var(name, slot_index)

# 常量
const = builder.const_int(10)

# 变量操作
builder.set_var(var, value)
var_expr = builder.var(var)

# 算术
result = builder.add(lhs, rhs)
result = builder.sub(lhs, rhs)

# 控制流
builder.goto(target)
builder.branch_if(condition, true_target, false_target)
builder.ret()

# 完成
mlil_func = builder.finalize()
```

### 4. LLILToMLILTranslator

核心转换器，执行栈消除：

**转换规则：**
```python
# LLIL → MLIL
StackStore(value, offset, slot_index) → SetVar(var_sN, translated_value)
StackLoad(offset, slot_index)         → Var(var_sN)
FrameLoad(offset)                     → Var(param_N)
SpAdd(delta)                          → (eliminated)
```

**使用方法：**
```python
translator = LLILToMLILTranslator()
mlil_func = translator.translate(llil_func)

# 或使用便捷函数
mlil_func = translate_llil_to_mlil(llil_func)
```

### 5. MLILFormatter

格式化输出 MLIL：

```python
# 文本格式
lines = MLILFormatter.format_function(mlil_func)
print('\n'.join(lines))

# DOT 格式（CFG 可视化）
dot = MLILFormatter.to_dot(mlil_func)
with open('cfg.dot', 'w') as f:
    f.write(dot)
# 然后：dot -Tpng cfg.dot -o cfg.png
```

## 集成到现有流程

### 在测试中使用

修改 `tests/test_scp_parser.py`:

```python
from ir.mlil import translate_llil_to_mlil, MLILFormatter

# 在现有 LLIL 生成之后
lifter = ED9VMLifter(parser=parser)
llil_func = lifter.lift_function(func)

# 转换为 MLIL
mlil_func = translate_llil_to_mlil(llil_func)

# 格式化输出
mlil_lines = MLILFormatter.format_function(mlil_func)

# 保存到文件
mlil_path = test_file.with_suffix('.mlil.txt')
mlil_path.write_text('\n'.join(mlil_lines) + '\n', encoding='utf-8')
```

## 下一步

MLIL 基础已经完成，可以进行：

1. **SSA 构造** - 添加 SSA 形式支持（Phi 节点、变量版本）
2. **数据流分析** - 到达定义、活跃变量分析
3. **优化** - 常量折叠、死代码消除、表达式简化
4. **类型推导** - 推断变量类型
5. **HLIL 转换** - 进一步提升到高级 IL

## 总结

✅ **MLIL 完整实现包括：**
- 核心指令集（所有操作类型）
- 变量系统（非 SSA，易于使用）
- Builder（便捷构建 API）
- Translator（LLIL→MLIL，栈消除）
- Formatter（清晰的输出格式）

✅ **设计优势：**
- 无栈语义，代码更清晰
- 统一的指令层次（无 Expr/Statement 分离）
- 与 LLIL 设计一致
- 完整的文档和示例

🎯 **可以开始使用了！**

# 快速开始指南

## 🚀 安装和设置

### 1. 安装为Python包 ✅ **推荐**

```bash
# 在项目根目录
pip install -e .
```

**好处**:
- ✅ 正确的Python包结构
- ✅ 解决所有相对导入问题
- ✅ 可以从任何地方导入decompiler3
- ✅ 提供命令行工具

### 2. 直接运行演示（开发模式）

```bash
python3 run_demos.py
```

## 🔧 命令行使用

安装后可以使用`decompiler3`命令：

```bash
# 显示系统信息
decompiler3 info

# 运行演示
decompiler3 demo basic
decompiler3 demo real_system
decompiler3 demo generator
decompiler3 demo add_instruction
decompiler3 demo extend_vm

# 编译TypeScript
decompiler3 compile input.ts output.bin --target x86

# 反编译字节码
decompiler3 decompile input.bin output.ts --target x86

# 列出支持的架构
decompiler3 targets
```

## 📁 新的项目结构

```
decompiler3/
├── setup.py                    # Python包配置
├── run_demos.py                # 开发模式运行脚本
├── src/
│   └── decompiler3/            # 主包
│       ├── __init__.py         # 包入口
│       ├── cli.py              # 命令行接口
│       ├── ir/                 # IR系统
│       ├── builtin/            # Built-in函数
│       ├── target/             # 目标后端
│       ├── pipeline/           # 编译管道
│       ├── typescript/         # TypeScript支持
│       ├── demos/              # 演示脚本
│       │   ├── basic_test.py
│       │   ├── real_system_demo.py
│       │   └── correct_generator_design.py
│       └── examples/           # 扩展示例
│           ├── add_instruction_example.py
│           └── extend_falcom_vm.py
├── requirements.txt
└── README.md
```

## 🎯 核心优势

### ✅ 解决了导入问题
- 不再有 `attempted relative import beyond top-level package` 错误
- 使用标准的绝对导入：`from decompiler3.ir.base import OperationType`
- 所有演示和示例都在包内，可以正确导入

### ✅ 标准Python包结构
- 遵循PEP 518/621标准
- 可以通过pip安装
- 提供命令行工具入口点
- 支持开发模式安装

### ✅ 清晰的模块组织
- **demos/**: 系统演示和测试
- **examples/**: 扩展功能示例
- **ir/**: 核心IR架构
- **target/**: 目标架构后端
- **builtin/**: Built-in函数系统
- **pipeline/**: 编译/反编译管道

## 🔥 运行示例

### 1. 基础功能测试
```bash
decompiler3 demo basic
```

### 2. 真实系统演示
```bash
decompiler3 demo real_system
```

### 3. 代码生成器设计演示
```bash
decompiler3 demo generator
```

### 4. 指令集扩展教程
```bash
decompiler3 demo add_instruction
```

### 5. Falcom VM扩展示例
```bash
decompiler3 demo extend_vm
```

## 📚 Python导入示例

```python
# 导入核心组件
from decompiler3 import OperationType, IRFunction, HLILConstant
from decompiler3.target import get_target_capability
from decompiler3.builtin import get_builtin

# 或者导入具体模块
from decompiler3.ir.hlil import HLILBinaryOp
from decompiler3.target.capability import X86Capability
from decompiler3.pipeline.compiler import Compiler

# 创建IR
function = IRFunction("example")
const_42 = HLILConstant(42, 4, "number")

# 获取目标能力
x86_cap = get_target_capability("x86")

# 使用built-in函数
abs_func = get_builtin("abs")
```

## ✅ 系统状态总结

| 组件 | 状态 | 说明 |
|------|------|------|
| 包结构 | ✅ 完成 | 标准Python包，可pip安装 |
| 导入系统 | ✅ 完成 | 绝对导入，无相对导入问题 |
| 三层IR架构 | ✅ 完成 | LLIL/MLIL/HLIL + SSA |
| TypeScript支持 | ✅ 完成 | 双向编译管道 |
| 目标后端 | ✅ 完成 | x86/Falcom VM/ARM |
| Built-in系统 | ✅ 完成 | 统一语义入口 |
| 指令选择 | ✅ 完成 | 模式匹配 + 成本模型 |
| 字节码生成 | ✅ 完成 | 多目标编码器 |
| 命令行工具 | ✅ 完成 | 统一CLI接口 |

## 🎉 现在可以:

1. **正确导入**: `from decompiler3.ir.base import OperationType`
2. **命令行使用**: `decompiler3 demo basic`
3. **标准安装**: `pip install -e .`
4. **无导入错误**: 完全消除了相对导入问题
5. **专业结构**: 符合Python包开发最佳实践

## 🛠️ 开发指南

### 添加新演示
在 `src/decompiler3/demos/` 目录下创建新的Python文件，使用绝对导入：

```python
from decompiler3.ir.base import OperationType
from decompiler3.target.capability import get_target_capability
```

### 添加新示例
在 `src/decompiler3/examples/` 目录下创建扩展示例，展示如何扩展系统功能。

### 运行测试
```bash
# 开发模式
python3 run_demos.py

# 安装后
decompiler3 demo basic
```

🎯 **核心改进**: 现在的项目结构解决了所有Python包导入问题，提供了专业的开发体验！
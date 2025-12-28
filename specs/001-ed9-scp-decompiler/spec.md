# Feature Specification: ED9 SCP Script Decompiler

**Feature Branch**: `001-ed9-scp-decompiler`
**Created**: 2025-12-28
**Status**: Implemented
**Input**: ED9/ED61st 引擎的 SCP 脚本文件反编译为可读的 TypeScript 源码

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 反编译单个脚本文件 (Priority: P1)

用户希望将一个 .dat 脚本文件反编译为可读的 TypeScript 代码，以便理解游戏脚本逻辑。

**Why this priority**: 这是反编译器的核心功能，没有这个功能其他一切都没有意义。

**Independent Test**: 运行 `python tests/test_scp_parser.py`，输入 debug.dat，验证输出 debug.ts 包含可读的 TypeScript 代码。

**Acceptance Scenarios**:

1. **Given** 一个有效的 .dat 脚本文件, **When** 运行反编译器, **Then** 输出包含所有函数的 TypeScript 代码
2. **Given** 脚本包含控制流（if/while/switch）, **When** 反编译, **Then** 输出结构化的控制流语句
3. **Given** 脚本包含函数调用和系统调用, **When** 反编译, **Then** 输出正确的函数调用表达式

---

### User Story 2 - 查看中间表示 (Priority: P2)

用户希望查看反编译过程中的中间表示（LLIL/MLIL/HLIL），以便调试或理解反编译过程。

**Why this priority**: 对于调试和理解反编译器行为很重要，但不是最终用户的主要需求。

**Independent Test**: 启用调试输出选项后，验证生成 .llil.asm、.mlil.asm、.hlil.ts 文件。

**Acceptance Scenarios**:

1. **Given** 一个 .dat 脚本文件且启用调试输出, **When** 反编译, **Then** 可选生成 LLIL、MLIL、HLIL 中间表示文件
2. **Given** 生成的 LLIL 文件, **When** 查看, **Then** 可以看到低级指令和基本块结构

**Note**: IL 文件输出是可选的调试功能，TypeScript 输出是必须的最终产物。

---

### User Story 3 - 使用签名数据库美化输出 (Priority: P3)

用户希望通过签名数据库将系统调用替换为有意义的函数名，并格式化参数（如枚举值、十六进制）。

**Why this priority**: 提升输出可读性，但需要先有基础反编译功能。

**Independent Test**: 配置签名 YAML 文件，验证 `syscall(5, 0, ...)` 被替换为 `mes_message_talk(...)`。

**Acceptance Scenarios**:

1. **Given** 签名数据库定义了 syscall, **When** 反编译包含该 syscall 的脚本, **Then** 输出使用定义的函数名
2. **Given** 签名定义了枚举格式, **When** 参数值匹配枚举, **Then** 输出枚举名而非数字
3. **Given** 签名定义了 variadic 参数, **When** syscall 有可变数量参数, **Then** 正确处理所有参数

---

### Edge Cases

- 脚本文件损坏或格式不正确时，应报告明确的错误信息
- 遇到未知的操作码时，立刻抛出异常并输出具体错误位置
- 空函数（无指令）应正确处理
- 嵌套控制流（多层 if/while）应正确还原

## Requirements *(mandatory)*

### Functional Requirements

**解析阶段**:
- **FR-001**: 系统必须能解析 ED9/ED61st 引擎的 .dat 脚本文件头
- **FR-002**: 系统必须能解析 ScpValue 的四种类型：Raw、Integer、Float、String
- **FR-003**: 系统必须能解析脚本中的所有函数定义和参数

**反汇编阶段**:
- **FR-004**: 系统必须能将 VM 字节码反汇编为可读的指令格式
- **FR-005**: 系统必须能识别基本块边界和控制流

**IR 转换阶段**:
- **FR-006**: 系统必须能将 VM 指令提升为 LLIL（低级中间语言）
- **FR-007**: 系统必须能将 LLIL 转换为 MLIL（中级中间语言），包括 SSA 分析
- **FR-008**: 系统必须能将 MLIL 转换为 HLIL（高级中间语言），包括控制流结构化
- **FR-009**: 每个 IR 层必须支持独立的格式化输出

**代码生成阶段**:
- **FR-010**: 系统必须能将 HLIL 生成为 TypeScript 代码
- **FR-011**: 生成的代码必须包含类型注解（number、string、boolean 等）
- **FR-012**: 生成的代码必须包含函数参数默认值

**签名数据库**:
- **FR-013**: 系统必须支持从 YAML 文件加载签名数据库
- **FR-014**: 签名数据库必须支持枚举定义
- **FR-015**: 签名数据库必须支持函数签名（参数名、类型、格式化提示）
- **FR-016**: 签名数据库必须支持 syscall 签名（subsystem/cmd 标识）
- **FR-017**: 签名数据库必须支持 variadic 参数
- **FR-018**: 签名数据库必须支持联合类型（如 `[string, number]`）

### Key Entities

- **ScpValue**: 脚本值，支持四种类型（Raw、Integer、Float、String），使用高 2 位标识类型
- **ScpFunction**: 脚本函数，包含名称、偏移量、参数列表、基本块
- **LowLevelIL**: 低级中间语言，接近 VM 指令的表示
- **MediumLevelIL**: 中级中间语言，包含 SSA 形式和类型信息
- **HighLevelIL**: 高级中间语言，包含结构化控制流（if/while/switch）
- **FormatSignatureDB**: 签名数据库，存储枚举、函数和 syscall 的格式化信息

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 能够成功反编译测试脚本集中 100% 的函数
- **SC-002**: 生成的 TypeScript 代码语法正确（无解析错误）
- **SC-003**: 控制流结构（if/while/switch）与原始脚本逻辑一致
- **SC-004**: 签名数据库能正确替换已定义的 syscall 为命名函数

## Assumptions

- 输入文件使用 ED9/ED61st 引擎的标准 SCP 格式
- 脚本使用小端字节序
- 字符串使用 UTF-8 编码（或系统默认编码）
- 用户有权访问游戏脚本文件

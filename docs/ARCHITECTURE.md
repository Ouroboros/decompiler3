# Falcom ED9 Decompiler Architecture

## Overview

分层架构设计，从字节码到高级IR的渐进式转换：

```
SCP File → Parser → Bytecode → Disassembler → LLIL → MLIL → HLIL → Decompiled Code
```

## Layer 1: Parser (已完成)

**位置**: `falcom/ed9/parser/`

**职责**:
- 解析SCP文件格式
- 提取函数、全局变量、字符串等元数据
- 提供字节码原始数据访问

**关键组件**:
- `ScpParser`: 主解析器
- `ScpHeader`: 文件头
- `Function`: 函数元数据
- `GlobalVar`: 全局变量

## Layer 2: Disassembler (新增)

**位置**: `falcom/ed9/disasm/`

**职责**:
- 将字节码转换为可读的指令序列
- 识别基本块边界
- 解析操作数

**关键组件**:

### 1. `optable.py` - VM操作码表

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Callable

class Opcode(IntEnum):
    """VM操作码"""
    PUSH                = 0x00
    POP                 = 0x01
    LOAD_STACK          = 0x02
    LOAD_STACK_DEREF    = 0x03
    # ... 更多操作码
    CALL_SCRIPT         = 0x22  # 注意：这里改为CALL_SCRIPT
    SYSCALL             = 0x24
    # ...

class OperandType(IntEnum):
    """操作数类型"""
    NONE    = 0  # 无操作数
    BYTE    = 1  # C: 单字节
    SHORT   = 2  # H: 短整数
    INT     = 3  # i: 整数
    FLOAT   = 4  # f: 浮点数
    STRING  = 5  # S: 字符串偏移
    OFFSET  = 6  # O: 代码偏移（跳转目标）
    FUNC    = 7  # F: 函数名
    VALUE   = 8  # V: ScpValue（多类型值）

@dataclass
class InstructionDescriptor:
    """指令描述符"""
    opcode: int
    mnemonic: str
    operand_format: str  # 操作数格式字符串，如 'VVC', 'O', 'i'
    flags: int = 0
    handler: Optional[Callable] = None

    @property
    def is_branch(self) -> bool:
        return self.opcode in (Opcode.JMP, Opcode.POP_JMP_ZERO, Opcode.POP_JMP_NOT_ZERO)

    @property
    def is_call(self) -> bool:
        return self.opcode in (Opcode.CALL, Opcode.CALL_SCRIPT)

    @property
    def is_return(self) -> bool:
        return self.opcode == Opcode.RETURN

# 操作码表
OPTABLE: dict[int, InstructionDescriptor] = {
    0x00: InstructionDescriptor(0x00, 'PUSH', 'V', handler=push_handler),
    0x01: InstructionDescriptor(0x01, 'POP', 'C'),
    0x02: InstructionDescriptor(0x02, 'LOAD_STACK', 'i'),
    # ... 更多指令
    0x22: InstructionDescriptor(0x22, 'CALL_SCRIPT', 'VVC'),
    0x24: InstructionDescriptor(0x24, 'SYSCALL', 'CBB'),
    # ...
}
```

### 2. `instruction.py` - 指令定义

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Operand:
    """操作数"""
    type: OperandType
    value: Any  # 可以是 int, float, str, ScpValue 等

    def __str__(self) -> str:
        match self.type:
            case OperandType.OFFSET:
                return f'loc_{self.value:X}'
            case OperandType.STRING:
                return f'"{self.value}"'
            case _:
                return str(self.value)

@dataclass
class Instruction:
    """指令"""
    offset: int                      # 在字节码中的偏移
    opcode: int                      # 操作码
    descriptor: InstructionDescriptor
    operands: list[Operand]

    @property
    def mnemonic(self) -> str:
        return self.descriptor.mnemonic

    def __str__(self) -> str:
        if not self.operands:
            return self.mnemonic

        ops = ', '.join(str(op) for op in self.operands)
        return f'{self.mnemonic}({ops})'
```

### 3. `disassembler.py` - 反汇编器

```python
from io import BytesIO
from typing import Iterator
import ml.fileio as fileio

class Disassembler:
    """字节码反汇编器"""

    def __init__(self, bytecode: bytes):
        self.fs = fileio.FileStream(BytesIO(bytecode))
        self.instructions: list[Instruction] = []
        self.labels: set[int] = set()  # 跳转目标位置

    def disassemble(self) -> list[Instruction]:
        """反汇编整个字节码"""
        self.instructions = []
        self.labels = set()

        # First pass: 识别所有跳转目标
        self._identify_labels()

        # Second pass: 反汇编指令
        self.fs.Position = 0
        while self.fs.Position < len(self.fs.BaseStream.getvalue()):
            inst = self._disassemble_instruction()
            self.instructions.append(inst)

        return self.instructions

    def _disassemble_instruction(self) -> Instruction:
        """反汇编单条指令"""
        offset = self.fs.Position
        opcode = self.fs.ReadByte()

        descriptor = OPTABLE.get(opcode)
        if not descriptor:
            raise ValueError(f'Unknown opcode 0x{opcode:02X} at offset 0x{offset:X}')

        # 解析操作数
        operands = self._parse_operands(descriptor.operand_format)

        return Instruction(offset, opcode, descriptor, operands)

    def _parse_operands(self, format_str: str) -> list[Operand]:
        """根据格式字符串解析操作数"""
        operands = []

        for fmt in format_str:
            match fmt:
                case 'C':  # Byte
                    val = self.fs.ReadByte()
                    operands.append(Operand(OperandType.BYTE, val))

                case 'H':  # Short
                    val = self.fs.ReadUShort()
                    operands.append(Operand(OperandType.SHORT, val))

                case 'i':  # Int
                    val = self.fs.ReadLong()
                    operands.append(Operand(OperandType.INT, val))

                case 'O':  # Offset (跳转目标)
                    val = self.fs.ReadULong()
                    self.labels.add(val)  # 记录跳转目标
                    operands.append(Operand(OperandType.OFFSET, val))

                case 'V':  # ScpValue
                    val = ScpValue(fs=self.fs)
                    operands.append(Operand(OperandType.VALUE, val))

                # ... 更多类型

        return operands

    def _identify_labels(self):
        """第一遍扫描，识别所有跳转目标"""
        # 实现逻辑...
        pass
```

## Layer 3: Lifter (新增)

**位置**: `falcom/ed9/lifter/`

**职责**:
- 将反汇编后的指令提升为LLIL
- 管理虚拟栈和寄存器状态
- 识别高级结构（函数调用、条件分支等）
- LLIL Function 会为每一条指令分配全局 `inst_index`，可通过 `get_instruction_by_index()`、
  `get_instruction_block_by_index()` 和 `iter_instructions()` 查询，用于后续 MLIL/HLIL pass 做数据流分析。

### 核心设计

```python
class BytecodeLifter:
    """字节码到LLIL的提升器"""

    def __init__(self, function: Function):
        self.function = function
        self.builder = FalcomVMBuilder()
        self.instructions: list[Instruction] = []

    def lift(self) -> LowLevelILFunction:
        """提升整个函数到LLIL"""

        # 1. 反汇编字节码
        disasm = Disassembler(self.function.bytecode)
        self.instructions = disasm.disassemble()

        # 2. 创建LLIL函数
        self.builder.create_function(
            self.function.name,
            self.function.offset,
            num_params=len(self.function.params)
        )

        # 3. 创建基本块
        blocks = self._create_basic_blocks()

        # 4. 提升每个基本块
        for block in blocks:
            self._lift_block(block)

        return self.builder.finalize()

    def _lift_instruction(self, inst: Instruction):
        """提升单条指令到LLIL"""

        match inst.opcode:
            case Opcode.PUSH:
                self._lift_push(inst)

            case Opcode.LOAD_STACK:
                offset = inst.operands[0].value
                self.builder.load_stack(offset)

            case Opcode.CALL:
                target = inst.operands[0].value
                self.builder.call(target)

            case Opcode.CALL_SCRIPT:
                module = inst.operands[0].value.value
                func = inst.operands[1].value.value
                argc = inst.operands[2].value
                self.builder.call_script(module, func, argc)

            case Opcode.ADD:
                self.builder.add()

            # ... 更多指令处理

    def _lift_push(self, inst: Instruction):
        """处理PUSH指令的不同变体"""
        value = inst.operands[0].value

        if isinstance(value, ScpValue):
            match value.type:
                case ScpValue.Type.Integer:
                    self.builder.push_int(value.value)

                case ScpValue.Type.Float:
                    self.builder.push(self.builder.const_float(value.value))

                case ScpValue.Type.String:
                    self.builder.push_str(value.value)
```

## 数据流

```
1. SCP File
   ↓
2. ScpParser.parse()
   ↓
3. Function objects (with bytecode)
   ↓
4. Disassembler.disassemble()
   ↓
5. Instruction list
   ↓
6. BytecodeLifter.lift()
   ↓
7. LowLevelILFunction
   ↓
8. (Future) MLIL/HLIL transformations
```

## 关键设计原则

1. **分离关注点**:
   - Parser只关心文件格式
   - Disassembler只关心字节码到指令
   - Lifter只关心指令到LLIL

2. **可测试性**:
   - 每层都可以独立测试
   - 使用明确定义的接口

3. **可扩展性**:
   - 操作码表驱动设计
   - 新指令只需添加到表中

4. **类型安全**:
   - 使用dataclass和类型注解
   - 明确的枚举类型

## 实现顺序

1. ✅ Parser (已完成)
2. ✅ LLIL Builder (已完成)
3. 🔄 Disassembler (下一步)
   - 实现optable.py
   - 实现instruction.py
   - 实现disassembler.py
4. 🔄 Lifter
   - 实现bytecode_lifter.py
   - 为每个指令实现提升逻辑
5. 🔄 Integration
   - 编写端到端测试
   - 优化性能

## 示例用法

```python
# 解析SCP文件
parser = ScpParser(fs, 'c0000.dat')
parser.parse()

# 获取函数
func = parser.functions[0]

# 反汇编
disasm = Disassembler(func.bytecode)
instructions = disasm.disassemble()

# 打印反汇编
for inst in instructions:
    print(f'{inst.offset:08X}: {inst}')

# 提升到LLIL
lifter = BytecodeLifter(func)
llil_func = lifter.lift()

# 打印LLIL
print(FalcomLLILFormatter.format_llil_function(llil_func))
```

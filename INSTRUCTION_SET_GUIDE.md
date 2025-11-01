# 指令集添加指南

本指南详细说明如何在系统的不同层次添加新的指令集。

## 🎯 **指令集层次结构**

```
1. IR操作类型 (OperationType) - 抽象操作
   ↓
2. 目标架构能力 (TargetCapability) - 架构支持的操作
   ↓
3. 指令选择模式 (InstructionPattern) - IR到机器指令的映射
   ↓
4. 字节码编码 (Assembler) - 机器指令到字节码
```

## 📍 **1. 添加新的IR操作类型**

**位置**: `src/ir/base.py` 中的 `OperationType` 枚举

```python
class OperationType(Enum):
    # 现有操作...

    # 添加新操作
    ROTATE_LEFT = auto()    # 循环左移
    ROTATE_RIGHT = auto()   # 循环右移
    POPCOUNT = auto()       # 计算1的个数
    BSWAP = auto()         # 字节序交换

    # 向量操作
    VECTOR_ADD = auto()
    VECTOR_MUL = auto()

    # 特殊操作
    SYSCALL = auto()       # 系统调用
    ATOMIC_LOAD = auto()   # 原子加载
    ATOMIC_STORE = auto()  # 原子存储
```

## 📍 **2. 添加新目标架构**

**位置**: `src/target/capability.py`

### 示例：添加RISC-V架构

```python
class RISCVCapability(TargetCapability):
    """RISC-V architecture capability model"""

    def __init__(self):
        super().__init__("riscv")
        self.pointer_size = 8  # RV64
        self.word_size = 8

        # RISC-V寄存器
        self.add_register_class(RegisterClass(
            "general", 32, 8,
            [DataType.INT64, DataType.POINTER],
            [f"x{i}" for i in range(32)]
        ))

        # 浮点寄存器
        self.add_register_class(RegisterClass(
            "float", 32, 8,
            [DataType.FLOAT64],
            [f"f{i}" for i in range(32)]
        ))

        # 特殊寄存器
        self.special_registers = {
            "zero": "x0",
            "ra": "x1",
            "sp": "x2",
            "gp": "x3",
            "tp": "x4"
        }

        # RISC-V指令能力
        self._add_riscv_instructions()

    def _add_riscv_instructions(self):
        """添加RISC-V特定指令"""

        # 基础算术指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.ADD,
            [DataType.INT32, DataType.INT64],
            [AddressingMode.REGISTER, AddressingMode.IMMEDIATE],
            latency=1, throughput=1
        ))

        # 新增：位操作指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.ROTATE_LEFT,
            [DataType.INT32, DataType.INT64],
            [AddressingMode.REGISTER, AddressingMode.IMMEDIATE],
            latency=1, throughput=1
        ))

        # 新增：原子操作指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.ATOMIC_LOAD,
            [DataType.INT32, DataType.INT64],
            [AddressingMode.MEMORY],
            latency=5, throughput=1, has_side_effects=True
        ))

# 注册新架构
TARGET_CAPABILITIES["riscv"] = RISCVCapability()
```

## 📍 **3. 添加指令选择模式**

**位置**: `src/target/instruction_selection.py`

### 扩展指令选择器

```python
class InstructionSelector:
    def _add_riscv_patterns(self):
        """添加RISC-V特定模式"""

        # 基础指令模式
        self.patterns.append(InstructionPattern(
            "riscv_add_immediate",
            lambda expr: (isinstance(expr, LLILBinaryOp) and
                         expr.operation == OperationType.ADD and
                         isinstance(expr.right, LLILConstant)),
            ["addi $dest $left ${right_imm}"],
            cost=1
        ))

        # 新增：位操作模式
        self.patterns.append(InstructionPattern(
            "riscv_rotate_left",
            lambda expr: (isinstance(expr, LLILUnaryOp) and
                         expr.operation == OperationType.ROTATE_LEFT),
            ["rol $dest $operand $amount"],
            cost=1
        ))

        # 新增：原子操作模式
        self.patterns.append(InstructionPattern(
            "riscv_atomic_load",
            lambda expr: (isinstance(expr, LLILLoad) and
                         hasattr(expr, 'is_atomic') and expr.is_atomic),
            ["lr.w $dest ($address)"],
            cost=5
        ))

    def _select_riscv_instruction(self, expr: LLILExpression) -> List[MachineInstruction]:
        """RISC-V特定指令选择"""

        if isinstance(expr, LLILBinaryOp):
            if expr.operation == OperationType.ROTATE_LEFT:
                left_reg = self.resolve_operand("$left", expr)
                amount = self.resolve_operand("$right", expr)
                return [MachineInstruction("rol", [left_reg, amount])]

        elif isinstance(expr, LLILLoad) and getattr(expr, 'is_atomic', False):
            address_reg = self.resolve_operand("$address", expr)
            dest_reg = self.register_allocator.allocate_register()
            return [MachineInstruction("lr.w", [dest_reg, f"({address_reg})"])]

        return []
```

## 📍 **4. 添加字节码编码**

**位置**: `src/pipeline/compiler.py` 中的汇编器

### 扩展字节码编码

```python
def _assemble_riscv_bytecode(self, instructions: List[MachineInstruction]) -> bytes:
    """汇编RISC-V字节码"""
    bytecode = bytearray()

    # RISC-V指令编码映射
    opcode_map = {
        # R型指令 (寄存器-寄存器)
        "add": 0x33,      # ADD rd, rs1, rs2
        "sub": 0x33,      # SUB rd, rs1, rs2
        "rol": 0x33,      # 自定义：循环左移

        # I型指令 (立即数)
        "addi": 0x13,     # ADDI rd, rs1, imm
        "lw": 0x03,       # LW rd, offset(rs1)

        # S型指令 (存储)
        "sw": 0x23,       # SW rs2, offset(rs1)

        # 原子指令
        "lr.w": 0x2F,     # LR.W rd, (rs1)
        "sc.w": 0x2F,     # SC.W rd, rs2, (rs1)
    }

    for instruction in instructions:
        if instruction.opcode.endswith(":"):
            continue

        opcode = opcode_map.get(instruction.opcode, 0x00)

        if instruction.opcode in ["add", "sub", "rol"]:
            # R型指令编码: [31:25]funct7 [24:20]rs2 [19:15]rs1 [14:12]funct3 [11:7]rd [6:0]opcode
            encoded = self._encode_r_type(opcode, instruction.operands)

        elif instruction.opcode in ["addi", "lw"]:
            # I型指令编码: [31:20]imm [19:15]rs1 [14:12]funct3 [11:7]rd [6:0]opcode
            encoded = self._encode_i_type(opcode, instruction.operands)

        elif instruction.opcode == "sw":
            # S型指令编码: [31:25]imm[11:5] [24:20]rs2 [19:15]rs1 [14:12]funct3 [11:7]imm[4:0] [6:0]opcode
            encoded = self._encode_s_type(opcode, instruction.operands)

        else:
            # 未知指令
            encoded = [0x00, 0x00, 0x00, 0x00]

        bytecode.extend(encoded)

    return bytes(bytecode)

def _encode_r_type(self, opcode: int, operands: List[str]) -> List[int]:
    """编码R型指令"""
    if len(operands) < 3:
        return [0x00, 0x00, 0x00, 0x00]

    rd = self._reg_to_num(operands[0])
    rs1 = self._reg_to_num(operands[1])
    rs2 = self._reg_to_num(operands[2])

    # RISC-V R型指令格式
    instruction = (
        (0 << 25) |        # funct7
        (rs2 << 20) |      # rs2
        (rs1 << 15) |      # rs1
        (0 << 12) |        # funct3
        (rd << 7) |        # rd
        opcode             # opcode
    )

    return [(instruction >> i) & 0xFF for i in [0, 8, 16, 24]]
```

## 📍 **5. 添加虚拟机指令集**

**位置**: 直接在目标架构中定义

### 示例：扩展Falcom VM指令集

```python
class ExtendedFalcomVMCapability(TargetCapability):
    """扩展的Falcom VM指令集"""

    def _add_extended_instructions(self):
        """添加扩展指令"""

        # 新增：字符串操作指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.STRING_CONCAT,
            [DataType.POINTER],
            [AddressingMode.STACK_RELATIVE],
            latency=5, throughput=1, has_side_effects=True
        ))

        # 新增：图形操作指令
        self.add_instruction_capability(InstructionCapability(
            OperationType.DRAW_SPRITE,
            [DataType.INT32, DataType.INT32, DataType.POINTER],
            [AddressingMode.IMMEDIATE, AddressingMode.STACK_RELATIVE],
            latency=10, throughput=1, has_side_effects=True
        ))

# 对应的字节码编码
def _assemble_extended_falcom_bytecode(self, instructions):
    opcode_map = {
        # 现有指令...

        # 新增指令
        "STR_CONCAT": 0x30,    # 字符串连接
        "DRAW_SPRITE": 0x40,   # 绘制精灵
        "PLAY_BGM": 0x41,      # 播放背景音乐
        "LOAD_SCENE": 0x42,    # 加载场景
        "SAVE_GAME": 0x43,     # 保存游戏
    }

    # 按相同模式编码...
```

## 🔧 **实际添加步骤**

### 步骤1：定义新操作类型
在 `src/ir/base.py` 中添加到 `OperationType`

### 步骤2：扩展目标能力
在 `src/target/capability.py` 中添加到对应的架构类

### 步骤3：添加指令选择
在 `src/target/instruction_selection.py` 中添加选择模式

### 步骤4：实现字节码编码
在 `src/pipeline/compiler.py` 中添加编码逻辑

### 步骤5：更新Built-in映射
在 `src/builtin/registry.py` 中添加目标特定映射

### 步骤6：测试验证
创建测试用例验证新指令集工作正常

## 📋 **完整示例**

让我创建一个完整的示例，展示如何添加一个新的位操作指令：
# Contract: High-Level IL

**Module**: `ir/hlil/hlil.py`

## HighLevelILFunction

Container for structured HLIL representation.

### Constructor

```python
HighLevelILFunction(name: str, parameters: List[IRParameter] = None)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Function name |
| parameters | List[IRParameter] | Function parameters |
| body | HLILBlock | Function body |
| variables | List[HLILVariable] | Local variables |
| return_type | Optional[str] | Return type hint |

## HLILBlock

Container for a sequence of statements.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| statements | List[HLILStatement] | Ordered statements |

### Methods

#### append()

```python
def append(self, stmt: HLILStatement) -> None
```

## HLILVariable

Variable representation in HLIL.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Variable name |
| type_hint | Optional[str] | Type annotation |
| default_value | Optional[str] | Default value |
| kind | VariableKind | LOCAL/PARAM/GLOBAL/REG |
| index | Optional[int] | Index for globals/regs |

### VariableKind

| Kind | Description |
|------|-------------|
| LOCAL | Local variable |
| PARAM | Function parameter |
| GLOBAL | Global variable |
| REG | Register |

## Control Flow Statements

### HLILIf

If-then-else statement.

```python
HLILIf(
    condition: HLILExpression,
    then_body: HLILBlock,
    else_body: Optional[HLILBlock] = None
)
```

**TypeScript output:**
```typescript
if (condition) {
    // then_body
} else {
    // else_body
}
```

### HLILWhile

While loop.

```python
HLILWhile(
    condition: HLILExpression,
    body: HLILBlock
)
```

**TypeScript output:**
```typescript
while (condition) {
    // body
}
```

### HLILDoWhile

Do-while loop.

```python
HLILDoWhile(
    body: HLILBlock,
    condition: HLILExpression
)
```

**TypeScript output:**
```typescript
do {
    // body
} while (condition);
```

### HLILFor

For loop.

```python
HLILFor(
    init: Optional[HLILStatement],
    condition: Optional[HLILExpression],
    update: Optional[HLILStatement],
    body: HLILBlock
)
```

**TypeScript output:**
```typescript
for (init; condition; update) {
    // body
}
```

### HLILSwitch

Switch statement.

```python
HLILSwitch(
    scrutinee: HLILExpression,
    cases: List[HLILSwitchCase],
    default_case: Optional[HLILBlock] = None
)
```

```python
HLILSwitchCase(
    value: int,
    body: HLILBlock
)
```

**TypeScript output:**
```typescript
switch (scrutinee) {
    case 0:
        // body
        break;
    case 1:
        // body
        break;
    default:
        // default_case
}
```

### HLILBreak / HLILContinue

Loop control statements.

```python
HLILBreak()
HLILContinue()
```

## Expression Statements

### HLILAssign

Assignment statement.

```python
HLILAssign(
    target: HLILVariable,
    value: HLILExpression
)
```

### HLILReturn

Return statement.

```python
HLILReturn(value: Optional[HLILExpression] = None)
```

### HLILExprStmt

Expression as statement.

```python
HLILExprStmt(expr: HLILExpression)
```

## Expressions

### HLILConst

Constant value.

```python
HLILConst(value: Union[int, float, str, bool])
```

### HLILBinaryOp

Binary operation.

```python
HLILBinaryOp(
    op: BinaryOp,
    lhs: HLILExpression,
    rhs: HLILExpression
)
```

**Binary operators:**

| BinaryOp | Symbol |
|----------|--------|
| ADD | + |
| SUB | - |
| MUL | * |
| DIV | / |
| MOD | % |
| EQ | == |
| NE | != |
| LT | < |
| LE | <= |
| GT | > |
| GE | >= |
| AND | && |
| OR | \|\| |
| BIT_AND | & |
| BIT_OR | \| |
| BIT_XOR | ^ |
| SHL | << |
| SHR | >> |

### HLILUnaryOp

Unary operation.

```python
HLILUnaryOp(
    op: UnaryOp,
    operand: HLILExpression
)
```

**Unary operators:**

| UnaryOp | Symbol |
|---------|--------|
| NEG | - |
| NOT | ! |
| BIT_NOT | ~ |

### HLILCall

Function call.

```python
HLILCall(
    func_name: str,
    args: List[HLILExpression]
)
```

### HLILSyscall

System call.

```python
HLILSyscall(
    subsystem: int,
    cmd: int,
    args: List[HLILExpression]
)
```

### HLILGlobalRef / HLILRegRef

Global variable or register reference.

```python
HLILGlobalRef(index: int)  # GLOBAL[index]
HLILRegRef(index: int)     # REG[index]
```

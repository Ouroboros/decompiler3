# Contract: TypeScript Code Generator

**Module**: `codegen/typescript.py`

## TypeScriptGenerator

Generates TypeScript source code from HLIL.

### Class Methods

#### set_signature_db()

Set the signature database for formatting.

```python
@classmethod
def set_signature_db(cls, db: FormatSignatureDB) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| db | FormatSignatureDB | Signature database |

**Note:** Must be called before `generate_function()` if signatures are needed.

#### generate_function()

Generate TypeScript code for a function.

```python
@classmethod
def generate_function(cls, func: HighLevelILFunction) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| func | HighLevelILFunction | HLIL function |
| **Returns** | str | TypeScript source code |

**Output format:**
```typescript
function FunctionName(
    arg0: number = 0,
    arg1: string = ""
): number {
    // body
}
```

#### generate_statement()

Generate code for a single statement.

```python
@classmethod
def generate_statement(cls, stmt: HLILStatement, indent: int = 0) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| stmt | HLILStatement | Statement to generate |
| indent | int | Indentation level |
| **Returns** | str | Generated code |

#### generate_expression()

Generate code for an expression.

```python
@classmethod
def generate_expression(cls, expr: HLILExpression) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| expr | HLILExpression | Expression to generate |
| **Returns** | str | Generated code |

## Type Mapping

| HLIL Type | TypeScript Type |
|-----------|-----------------|
| INT | number |
| FLOAT | number |
| STRING | string |
| BOOL | boolean |
| VOID | void |
| UNKNOWN | any |
| VARIANT | any |

## Output Conventions

### Function Signature

```typescript
function name(param: type = default): returnType {
    // body
}
```

### Control Flow

**If statement:**
```typescript
if (condition) {
    // then
} else {
    // else
}
```

**While loop:**
```typescript
while (condition) {
    // body
}
```

**Switch statement:**
```typescript
switch (value) {
    case 0:
        // body
        break;
    case 1:
        // body
        break;
    default:
        // default
        break;
}
```

### Syscall Formatting

**With signature:**
```typescript
mes_message_talk(SPEAKER_PLAYER, "Hello");
```

**Without signature:**
```typescript
syscall(5, 0, 1, "Hello");
```

### Variable Naming

| Source | Output |
|--------|--------|
| `var_s0` | `var_s0` |
| `arg0` | `arg0` |
| `GLOBAL[0]` | `GLOBAL[0]` |
| `REG[0]` | `REG[0]` |

## Operator Precedence

Parentheses are added based on operator precedence:

| Precedence | Operators |
|------------|-----------|
| 1 (highest) | `!`, `~`, unary `-` |
| 2 | `*`, `/`, `%` |
| 3 | `+`, `-` |
| 4 | `<<`, `>>` |
| 5 | `<`, `<=`, `>`, `>=` |
| 6 | `==`, `!=` |
| 7 | `&` |
| 8 | `^` |
| 9 | `\|` |
| 10 | `&&` |
| 11 (lowest) | `\|\|` |

## Indentation

- 4 spaces per indentation level
- Consistent across all constructs

# Contract: Format Signature Database

**Module**: `falcom/ed9/format_signatures.py`

## FormatSignatureDB

Database for function/syscall formatting metadata.

### Constructor

```python
FormatSignatureDB()
```

Creates empty database.

### Methods

#### load_yaml()

Load signatures from YAML file.

```python
def load_yaml(self, path: Path) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| path | Path | Path to YAML file |

**Raises:**
- `FileNotFoundError` if file doesn't exist
- `yaml.YAMLError` if YAML is invalid

**YAML Schema:**
```yaml
enums:
  EnumName:
    0: VALUE_ZERO
    1: VALUE_ONE

functions:
  function_name:
    params:
      - name: arg1
        type: number
        format: hex
      - name: arg2
        type: string
    return:
      type: number

syscalls:
  syscall_name:
    id: [subsystem, cmd]
    params:
      - name: arg1
        type: number
      - name: args
        type: "[string, number]"
        variadic: true
    return:
      type: void
```

#### get_function()

Get function signature.

```python
def get_function(self, name: str) -> Optional[FunctionSig]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Function name |
| **Returns** | Optional[FunctionSig] | Signature or None |

#### get_syscall()

Get syscall signature by ID.

```python
def get_syscall(self, subsystem: int, cmd: int) -> Optional[SyscallSig]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| subsystem | int | Subsystem ID |
| cmd | int | Command ID |
| **Returns** | Optional[SyscallSig] | Signature or None |

#### get_enum_value()

Look up enum value name.

```python
def get_enum_value(self, enum_name: str, value: int) -> Optional[str]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| enum_name | str | Enum type name |
| value | int | Numeric value |
| **Returns** | Optional[str] | Value name or None |

#### format_value()

Format value using hint.

```python
def format_value(self, value: int, format_hint: str) -> Optional[str]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| value | int | Value to format |
| format_hint | str | "hex" or enum name |
| **Returns** | Optional[str] | Formatted string or None |

**Examples:**
```python
db.format_value(255, "hex")        # → "0xff"
db.format_value(1, "MyEnum")       # → "VALUE_ONE"
db.format_value(999, "MyEnum")     # → None (not in enum)
```

## ParamHint

Parameter formatting hint.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Parameter name |
| type | Optional[str] | Type hint |
| format | Optional[str] | Format hint |
| variadic | bool | Is variadic |

### Type Values

| Type | Description |
|------|-------------|
| `"number"` | Numeric value |
| `"string"` | String value |
| `"boolean"` | Boolean value |
| `"[string, number]"` | Union type |

### Format Values

| Format | Description |
|--------|-------------|
| `"hex"` | Hexadecimal |
| `"EnumName"` | Enum lookup |
| `None` | Decimal (default) |

## FunctionSig

Function signature.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| params | List[ParamHint] | Parameter hints |
| return_hint | Optional[ReturnHint] | Return type hint |

## SyscallSig

Syscall signature.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| name | str | Display name |
| subsystem | int | Subsystem ID |
| cmd | int | Command ID |
| params | List[ParamHint] | Parameter hints |
| return_hint | Optional[ReturnHint] | Return hint |

## ReturnHint

Return value hint.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| type | str | Return type |
| format | Optional[str] | Format hint |

## YAML Examples

### Basic Syscall

```yaml
syscalls:
  mes_message_talk:
    id: [5, 0]
    params:
      - name: speaker_id
        type: number
      - name: message
        type: string
    return:
      type: void
```

### With Enum Formatting

```yaml
enums:
  SpeakerType:
    0: SPEAKER_NONE
    1: SPEAKER_PLAYER
    2: SPEAKER_NPC

syscalls:
  mes_set_speaker:
    id: [5, 1]
    params:
      - name: speaker_type
        type: number
        format: SpeakerType
```

### Variadic Parameters

```yaml
syscalls:
  debug_print:
    id: [0, 10]
    params:
      - name: format
        type: string
      - name: args
        type: "[string, number]"
        variadic: true
```

### Hex Formatting

```yaml
syscalls:
  set_flags:
    id: [2, 5]
    params:
      - name: flags
        type: number
        format: hex
```

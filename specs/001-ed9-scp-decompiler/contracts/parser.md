# Contract: SCP Parser

**Module**: `falcom/ed9/parser/scp.py`

## ScpParser

Main parser for SCP script files.

### Constructor

```python
ScpParser(fs: FileStream, filename: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| fs | FileStream | File stream positioned at start |
| filename | str | Filename for error messages |

### Methods

#### parse()

Parse the SCP file header and function table.

```python
def parse(self) -> None
```

**Preconditions:**
- File stream is valid and readable
- File starts with `#scp` magic

**Postconditions:**
- `self.header` is populated
- `self.functions` contains all function entries
- `self.globals` contains all global variables

**Raises:**
- `ValueError` if magic is invalid
- `IOError` if file is truncated

#### get_func_by_name()

Look up a function by name.

```python
def get_func_by_name(self, name: str) -> Optional[ScpFunction]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Function name to find |
| **Returns** | Optional[ScpFunction] | Function or None |

#### disasm_all_functions()

Disassemble all functions with stack simulation.

```python
def disasm_all_functions(self) -> List[ScpFunction]
```

**Returns:** List of functions with populated basic blocks and instructions.

**Side effects:**
- Performs stack simulation for CALL pattern recognition
- Populates basic block CFG edges

## ScpValue

Typed value representation.

### Constructor

```python
ScpValue(value: Union[int, float, str, RawInt], fs: FileStream = None)
```

**From value:**
```python
val = ScpValue(42)           # Integer
val = ScpValue(3.14)         # Float
val = ScpValue("hello")      # String
val = ScpValue(RawInt(0xFF)) # Raw
```

**From stream:**
```python
val = ScpValue(fs=file_stream)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| type | ScpValue.Type | Value type (Raw/Integer/Float/String) |
| value | int \| float \| str | Decoded value |

### Methods

#### to_bytes()

Serialize to 4 bytes.

```python
def to_bytes(self) -> bytes
```

**Returns:** 4-byte little-endian encoded value with type tag.

## ScpParamFlags

Parameter flag enumeration.

| Flag | Value | Description |
|------|-------|-------------|
| Value32 | 0x00 | Regular value |
| Nullable32 | 0x04 | Nullable value |
| NullableStr | 0x08 | Nullable string |
| Offset | 0x0C | Code offset |
| Pointer | 0x10 | Pointer |

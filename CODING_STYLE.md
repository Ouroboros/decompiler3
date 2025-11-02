# Coding Style Rules

## Spacing
- ALL `=` must have spaces: `param = None`, `func(key = value)`, `x = 10`

## Strings
- Use single quotes `'` by default: `'string'`
- Use double quotes `"` only when necessary (e.g., string contains single quote)

## Naming
- Use full names: `STACK` not `S`, `REG` not `R`, `sp` not `vsp`
- Use specific methods: `eq()`, `ne()`, `add()` - NOT generic `compare(op_type: str)`

## Type Safety
- No hardcoded strings: use enums/types, not `if op_name == "EQ"`
- Use instruction types: `condition: LowLevelILInstruction` not `condition: str`

## Return Values
- Use NamedTuple for multi-value returns, NOT bare tuples

## Formatting
- Pattern functions return lines WITHOUT indentation - caller controls indent

## Binary Operations
- Both-or-neither rule: `lhs` and `rhs` must BOTH be None or BOTH be provided
- Never allow one None and one value

## Architecture
- `ir/` = VM-agnostic generic LLIL
- `falcom/` = Falcom VM-specific code

## Examples
- Use ONLY real game code examples, NOT synthetic/abstract examples

## Emojis
- Do NOT use in code unless user explicitly requests

# Information Contract Audit (`tools/contract`)

Checks that every layer keeps the script information that cannot be regenerated
(`docs/REDESIGN.md` section 3). Behavior is checked separately by `tools/diffcheck`.

```bash
python -m tools.contract tests/debug.dat
python -m tools.contract --layers parser,hlil --examples 5 --json audit.json tests/*.dat
```

Exit code is 0 only when no difference is found.

## Reference

- **Tables** are decoded from the raw bytes by `tools/contract/image.py`, independently of
  `falcom/ed9/parser`: header `dword_14`, globals (name, type), function order, and per function
  `name_hash`, `is_common`, `byte06`, raw parameter flags, typed defaults and every call debug record
  (callee index or none, call type, arguments with their type).
- **Code items** come from the reachable instructions of the disassembled function: line markers
  (`DEBUG_SET_LINENO`) and debug logs (`DEBUG_LOG`) with their bytecode address. Unreachable code is not
  part of the IR contract.

## Layers

| Layer | Tables | Code items |
| --- | --- | --- |
| `writer` | Parsing and writing back with the original layout (`falcom/ed9/parser/writer.py`) must give the original bytes; the code range is generated only by the instruction encoder | — |
| `code` | Per function, the instructions decoded linearly from the entry to the next function (the last one to the string pool), dead code included, must re-encode to the original bytes | — |
| `parser` | Parser model (`ScpParser`) | — |
| `llil` | Function name, `is_common_func`, parameter types and defaults; file-level items, `name_hash`, `byte06` and debug records from the LLIL module metadata | `LowLevelILDebug` (`line`, `log`) with address |
| `mlil` | As LLIL, from `source_params` and the MLIL module | `MLILDebug` (`line`, `log`) with address |
| `hlil` | Function name, `is_common_func`, defaults; module items as LLIL, from the HLIL module | Line and log statements with address |
| `ts` | Not audited: the TypeScript file-level format is deferred | `// line(N)` / `// log(N)` comments, values only (no addresses) |

Items a layer does not represent are reported as `missing`. Values are compared with their tag, so a
Raw default that became an Integer is a difference. Parameter types of LLIL and MLIL are Python type names
mapped back to the raw flags (`Value32` 1, `str` 2, `Pointer` 5, `Nullable32` 9, `NullableStr` 10).
`name_hash` is regenerable from the name (`falcom/ed9/parser/crc32.py`); the parser layer reports the
computed value and the audit compares it with the stored one.

Code items are compared as sets: a pass may copy a line marker (shared code), but every marker must keep
its address and none may appear at a new address.

## Raw values

Raw versus Integer inside code is not audited statically: a `PUSH_RAW 0` that only reserves a stack slot
is regenerable, and only the behavior check can tell it from a Raw value that reaches a call, a global or a
return. `python -m tools.diffcheck --strict-raw` observes Raw words with their tag.

## Negative tests

`tests/test_contract_audit.py` injects losses (table fields, module items, removed or moved line markers
in LLIL, HLIL and TypeScript, Raw returned as Integer) and requires each to be reported.

# ED9 Script Toolchain Redesign

Status: proposal, not implemented. The current pipeline stays the reference until the new one reaches
parity (section 9).

## 1. Goal

A toolchain whose every representation of an ED9 script is complete, so a script can be decompiled, edited
and compiled back. This document covers decompilation; every interface is designed with its inverse so the
compiler can be added without changing the representations.

Inputs that stay authoritative: `falcom/ed9/spec/vm_spec.md` (checked against `ed9_crack.exe`),
`falcom/ed9/vm_values.py`, the bytecode reference interpreter of `tools/diffcheck`, and the 88 `scena`
scripts plus the 3 test DATs.

## 2. Principles

| # | Principle | Consequence |
| --- | --- | --- |
| P1 | Every representation is complete with respect to the information contract (section 3) | A backend can be generated from any layer without the parser |
| P2 | Every transformation has an inverse or a verifier | read/write, decode/encode, lift/lower, structure/flatten |
| P3 | Each fact is defined once | One binary schema for reading and writing, one opcode table for decoding, encoding and interpretation, one value semantics, one expression node set for flat and structured IR |
| P4 | No node kind is ignorable | Line markers and debug logs are statements; no pass treats them as no-ops or drops them |
| P5 | Verification is part of the pipeline | Each stage has a round trip, an audit or an interpreter; a check mode audits after every pass |
| P6 | Generic and game-specific code are separate | `ir/`-style generic containers, ED9 metadata and semantics in the ED9 package |

## 3. Information Contract

Information that cannot be regenerated and must be present in every layer:

| Scope | Item |
| --- | --- |
| File | header field `dword_14`; global table (name, type) |
| Function table | order, name, `name_hash`, `is_common_func`, `byte06`, raw 32-bit parameter flags, typed default values |
| Function | call debug records verbatim, including those with zero arguments |
| Code | behavior; line markers with their bytecode address; `DEBUG_LOG` values; Raw versus Integer tags; exact constant values |

Regenerable, not stored above the code layer: code and stack layout (offsets, labels, slots, stack values
never read), the fixed pushes of call sequences (`PUSH_CURRENT_FUNC_ID`, `PUSH_RET_ADDR`,
`PUSH_CALLER_FRAME`), string and table placement. Unreachable code is kept by the image and code layers
only (decision D1).

## 4. Layers

```
bytes ⇄ L0 Image ⇄ L1 Code ⇄ L2 Flat IR (⇄ SSA) ⇄ L3 Structured IR → backends (TypeScript, ...)
```

All layers hang off one `Module`, which owns the contract metadata of section 3 once; functions carry
their L1, L2 and L3 forms.

### L0 Image

`ScriptImage`: header, function records, global records, parameter flags, default values, debug records,
string pool, per-function code bytes, plus a separate `Layout` (section order, offsets, string placement).

- One declarative schema per record drives both `read` and `write`.
- `write(read(b)).bytes == b` with the original layout; `read(write(image, fresh layout)) == image`
  with a regenerated one.

### L1 Code

Per function, the instructions of its whole byte range, decoded linearly from the entry to the next
function, with branch targets turned into labels; the CFG marks unreachable instructions.

- One opcode table (operand types, sizes, stack effect, control flow) drives decode, encode, formatting
  and the reference interpreter.
- `encode(decode(code)) == code` for every function.

### L2 Flat IR

Stack-free, CFG-based. Statements: `Assign`, `Store` (global, register, through pointer), `Call` (local,
script, syscall; result and `REGS[0]` effect explicit), `If`, `Goto`, `Return`, `Line`, `Log`.
Expressions: typed constants (`Raw`, `Int`, `Float`, `Str`, stack address), variables, VM operators.

- Lifting simulates the stack once and resolves call frames; there is no separate stack IR, because L1
  already is the exact stack-level form.
- An SSA form of the same nodes (versioned variables, phi) is used for analysis and leaves through
  deconstruction.
- Constant folding, type tests and identities go through `vm_values` only.
- Lowering L2 → L1 is defined from the start (needed by the compiler, and checked by interpretation).

### L3 Structured IR

A tree of `Block`, `If`, `While`, `DoWhile`, `Switch` (multi-label cases), `Break`, `Continue`, `Return`
whose leaves are L2 statements and expressions — the same node classes, not copies.

- Structuring uses post-dominators (current `structural_analysis.py` algorithm) and copies shared code
  explicitly; copied `Line` statements keep their address.
- Flattening L3 → L2 is defined, so structuring can be verified and compiled back.
- Readability passes (temporary elimination, naming, condition simplification, switch rebuilding) run here
  or in L2 and are subject to the same audits.

### Backends

A backend translates a `Module` at L3 and must state how it represents every contract item. The
TypeScript backend keeps the parse-back check (`ts_parse.py`): its output parsed back must equal the L3
it came from in behavior and contract content.

## 5. Passes

```
class Pass:
    preserves: set of contract items      # audited after the pass in check mode
    def run(self, module) -> module
```

- A pass may move or duplicate `Line` and `Log` statements but never removes them; audits compare sets of
  (address, line) and multisets of logs.
- Check mode runs the contract audit and, optionally, the interpreter after every pass, so a loss or
  miscompile is attributed to one pass.

## 6. Verification

| Check | Scope | Proves |
| --- | --- | --- |
| V0 image round trip | 88 scripts | L0 is complete; independent field audit against raw bytes (`tmp/dat_audit.py` prototype) |
| V1 code round trip | every function | L1 is complete, including unreachable bytes |
| V2 contract audit | L1, L2, L3, backend | No contract item is lost; one extractor serves L2 and L3 because nodes are shared |
| V3 behavior | every function | bytecode ≡ L2 ≡ L3 ≡ backend under one input plan; the existing bytecode interpreter stays the independent oracle; Raw is compared by tag |
| V4 per-pass check mode | every pass | Localizes a failure to the pass that introduced it |
| V5 negative tests | each layer | Deleting a contract item or miscompiling one node is detected and localized |

Incremental runs: each function's forms are hashed per layer; only functions with changed hashes are
rechecked. Full runs are for checker changes and acceptance.

## 7. Comparison With the Current Design

Measured values are from the current code and the 88 `scena` scripts. Values for the new design are design
targets until the prototype measures them (section 8).

| Dimension | Current (measured) | New (target) |
| --- | --- | --- |
| Completeness | Global names wrong (5 of 5), 112,461 debug records dropped, Raw tag lost after LLIL, ≥ 572 line markers lost in HLIL, parameter flags lost in HLIL, `DEBUG_LOG` lost in the lifter | Zero contract differences in every layer (V2) |
| Round trip | No writer, no encoder: 0 of 88 | 88 of 88 byte-identical (V0, V1) |
| Places that convert between node sets | 6 (model → instructions → LLIL → MLIL ⇄ SSA → HLIL → TypeScript), each a possible loss | 3 (image → code → L2; L2 → L3 shares nodes; L3 → backend) |
| Node classes | LLIL 50, MLIL 57, HLIL 32 = 139 | One expression set shared by L2/L3; estimate under 60 in total |
| Behavior verification | bytecode ≡ HLIL ≡ TypeScript; 22,118 of 22,118 PASS, 99.66 % block coverage | bytecode ≡ L2 ≡ L3 ≡ backend, same pass rate and coverage at least |
| Failure localization | HLIL passes; MLIL and earlier reported as "MLILToHLIL (or MLIL)" | Exact layer and pass (V4) |
| Value semantics | `vm_values` in SCCP and the checker; other passes reason on Python values | `vm_values` everywhere |
| Compiler readiness | Read-only; every layer would need an inverse | Inverses defined per layer (P2) |
| Instruction fidelity | The disassembler appends synthetic size-0 `JMP`s at `offset - 1` when it splits blocks (595 in the three largest scripts), so the instruction list is not the bytecode | L1 holds exactly the decoded bytes; fallthrough is a CFG edge, not an instruction |
| I/O | `FileStream` on the file seeks for nearly every value, and each seek refills the buffer from `/mnt/d` over 9P | Read the whole file once and decode from memory |
| Readability (TypeScript) | 58,764 `let`, 54,317 of them register temporaries; 22,719 returns through a temporary | Not worse; temporaries removed in L2 |
| Speed | section 7.1 | Not slower; parsing from memory |
| Development cost | Exists | New work until parity — the one dimension the new design loses |

### 7.1 Baseline Measurements

Single process, current code (`b75a696..78b31c6` pipeline), scripts on `/mnt/d`:

| Measurement | Result |
| --- | --- |
| Decompile to HLIL and TypeScript, file-backed | `c1010` (19,876 B) 1.8 s, `m7200` (28,722 B) 2.2 s, `t3400` (1,590,061 B) 132.9 s |
| Differential check | `c1010` 1.0 s, `m7200` 1.2 s, `t3400` 230.5 s; all 88 scripts about 9.9 h of CPU |
| Parse tables of `c0000` (5,539,286 B), file-backed | not finished after 10 min; stack dumps show `ScpValue` reads of debug records |
| Parse and disassemble `c0000`, file read into memory first | 1 s tables, 4 to 5 s for 964 functions |
| Linear decode coverage, `c0000` / `c0400` / `c0600` | 2,587 functions, none overlapping the next; 34 functions hold 228 unreachable bytes, all inside their own range and decodable (dead `JMP`, `PUSH`) |
| Code size | parser 931, disassembler 1,239, lifter 310, other ED9 1,614, `ir/core` 77, LLIL 1,636, MLIL 3,454 + passes 3,357, HLIL 3,139 + passes 1,702, codegen 714, checker 2,224 lines |

The file-backed versus in-memory difference is an I/O cost of the current parser, independent of the
pipeline design; the decompile and check timings above include it.

## 8. Evaluation Before Rebuilding

1. Prototype the whole chain (L0 to TypeScript) on `c1010`, `c1020`, `m7200` (167 functions).
2. Measure every row of section 7 for both designs on the same functions.
3. Rebuild only if the new design is not worse on any row except development cost and is better on
   completeness, round trip and localization. Otherwise fix the current pipeline in place.

## 9. Migration

- New package next to the current one in this repository; the current pipeline, checker and corpus stay
  usable throughout.
- Algorithms already validated are ported, not reinvented: post-dominator merge points, shared-code
  copying, switch rebuilding with multi-label cases, VM constant folding, VM-order parameter numbering.
- Parity: all 88 `scena` scripts pass V0 to V3 and the 3 test DATs pass; then the current pipeline is
  removed.

## 10. Open Decisions

| # | Question | Default |
| --- | --- | --- |
| D1 | Keep unreachable code above L1? | No: kept in L0/L1 only (linear decode covers it, section 7.1) |
| D2 | Package location and name | New directory in this repository |
| D3 | Contract representation in the TypeScript backend | Deferred until a compiler is planned |

## 11. Risks

- The prototype may show that a flat plus structured IR cannot express something the current three-layer
  pipeline handles; section 8 exists to find that before rebuilding.
- Unknown semantics of `name_hash` and debug records: stored verbatim, but a compiler will need rules to
  regenerate them after edits.
- Comparing Raw by tag may expose new failures in both designs.
- 0.34 % of blocks are not reached by the checker in either design.

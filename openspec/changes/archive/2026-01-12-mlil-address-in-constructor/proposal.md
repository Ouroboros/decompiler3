# Proposal: mlil-address-in-constructor

**Change ID**: `mlil-address-in-constructor`
**Status**: Draft
**Created**: 2026-01-12

## Summary

Refactor MLIL instruction classes to accept `address` parameter in constructors, eliminating the need for manual post-construction address assignment.

## Problem

Current approach to address preservation requires manual assignment after construction:

```python
new_inst = MLILCall(target, new_args)
new_inst.address = inst.address  # Easy to forget!
```

This pattern is:
1. **Error-prone**: Easy to forget the second line
2. **Not scalable**: New instruction types will need the same treatment
3. **Verbose**: Two lines instead of one

## Solution

Add `address` as a keyword-only parameter to all MLIL instruction constructors:

```python
# Before
new_inst = MLILCall(target, new_args)
new_inst.address = inst.address

# After
new_inst = MLILCall(target, new_args, address=inst.address)
```

## Design

### Approach: Keyword argument in base class + `**kwargs` propagation

1. **Base class** (`MediumLevelILInstruction`):
   ```python
   def __init__(self, operation: MediumLevelILOperation, *, address: int = 0):
       super().__init__()
       self.operation = operation
       self.address = address
       ...
   ```

2. **All subclasses** accept `**kwargs` and pass to parent:
   ```python
   class MLILCall(MediumLevelILStatement):
       def __init__(self, target: str, args: List[...], **kwargs):
           super().__init__(MediumLevelILOperation.MLIL_CALL, **kwargs)
           self.target = target
           self.args = args
   ```

### Why `**kwargs` instead of explicit `address` parameter?

- Future-proof: Can add more inheritable attributes (e.g., `llil_index`, `options`) without changing all subclass signatures
- Less boilerplate: Subclasses don't need to repeat the parameter
- Consistent pattern: All subclasses use the same `**kwargs` pattern

## Scope

- **Primary**: `ir/mlil/mlil.py` - All MLIL instruction classes (~30 classes)
- **Secondary**: `ir/mlil/mlil_ssa.py` - SSA instruction classes (~5 classes)
- **Cleanup**: Remove manual `new_inst.address = ...` lines from:
  - `ir/mlil/mlil_ssa.py`
  - `ir/mlil/passes/*.py`

## Impact

- **Low risk**: Backward compatible (address defaults to 0)
- **No API breaking changes**: Existing code without address parameter still works
- **Cleaner code**: Single-line instruction creation with address

## Alternatives Considered

1. **Factory method `with_address()`**: More verbose, requires method on every class
2. **Copy constructor**: Doesn't fit the use case (creating new instruction, not copying)
3. **Post-construction hook**: Still requires remembering to call it

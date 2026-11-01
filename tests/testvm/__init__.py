'''TVM: a small VM, deliberately unlike ED9, that reaches the IR only through ir/core/target.py

It exists to prove the decompiler core needs no change for another VM (docs/REDESIGN.md stage 8):
8-byte stack slots, parameter 1 pushed first, 32-bit wrapping integers with floor division, results in
register 3, host calls by name, and its own image format.
'''

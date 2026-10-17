'''Differential equivalence checker: ED9 bytecode vs decompiled HLIL

A function is equivalent when, for the same inputs, the bytecode reference
interpreter and the HLIL interpreter produce the same observable trace.
See docs/DIFFCHECK.md for the equivalence contract.
'''

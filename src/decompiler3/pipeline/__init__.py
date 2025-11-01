"""
Decompilation and compilation pipeline

Main entry point for the bidirectional IR system.
Coordinates the entire process from bytecode to TypeScript and back.
"""

from .decompiler import *
from .compiler import *
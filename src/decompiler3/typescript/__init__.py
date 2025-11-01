"""
TypeScript compilation pipeline

Bidirectional compilation between HLIL and TypeScript.
Supports both decompilation (HLIL → TS) and compilation (TS → HLIL).
"""

from .generator import *
from .parser import *
from .pipeline import *
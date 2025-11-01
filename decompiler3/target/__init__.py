"""
Target backend system

Pluggable backend system with capability models for different target architectures.
Supports legalization, instruction selection, and code generation.
"""

from .capability import *
from .backend import *
from .legalization import *
from .instruction_selection import *
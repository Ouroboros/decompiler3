'''Falcom script parsers for different binary formats'''

from .scp import ScpParser
from .types_scp import *
from .types_parser import *
from .writer import ScpWriter, write_scp

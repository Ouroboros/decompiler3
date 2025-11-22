'''
Code Generators

Generate source code from HLIL in various target languages.
'''

from .typescript import *

__all__ = [
    'TypeScriptGenerator',
    'generate_typescript',
    'generate_typescript_header',
]

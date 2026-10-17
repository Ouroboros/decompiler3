'''Parse generated TypeScript back into HLIL, so the HLIL interpreter can execute the emitted text

Only the subset codegen/typescript.py emits is accepted; anything else raises TSParseError. Mapping:

- GLOBALS[i] / REGS[i] are storage variables; other names must be parameters or declared with let
- deref(p) / deref_set(p, v) / addr_of(x) are pointer operations, int(x) is x
- syscall(subsystem, cmd, ...) / extern_call('module:func', ...) are external calls, other calls script calls
- true / false are the integers 1 / 0, numeric literals with '.' or an exponent are floats
'''

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ir.hlil import (
    BinaryOp, HighLevelILFunction, HLILAddressOf, HLILAssign, HLILBinaryOp, HLILBlock, HLILBreak, HLILCall,
    HLILComment, HLILConst, HLILContinue, HLILDeref, HLILDoWhile, HLILExpression, HLILExprStmt, HLILExternCall,
    HLILIf, HLILReturn, HLILStatement, HLILSwitch, HLILSwitchCase, HLILSyscall, HLILUnaryOp, HLILVar, HLILVariable,
    HLILWhile, UnaryOp, VariableKind,
)


class TSParseError(Exception):
    pass


@dataclass
class Token:
    kind: str
    text: str
    position: int


TOKEN_PATTERN = re.compile(r'''
    (?P<space>\s+)
  | (?P<comment>//[^\n]*)
  | (?P<number>0[xX][0-9A-Fa-f]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
  | (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<op>===|!==|==|!=|<=|>=|&&|\|\||<<|>>|[-+*/%&|^!~<>=(){}\[\],;:?.])
''', re.VERBOSE)

STRING_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '0': '\0', '\\': '\\', '"': '"', "'": "'"}

# Binary operators by JavaScript precedence, loosest first
BINARY_LEVELS = (
    {'||': BinaryOp.OR},
    {'&&': BinaryOp.AND},
    {'|': BinaryOp.BIT_OR},
    {'^': BinaryOp.BIT_XOR},
    {'&': BinaryOp.BIT_AND},
    {'==': BinaryOp.EQ, '!=': BinaryOp.NE},
    {'<': BinaryOp.LT, '<=': BinaryOp.LE, '>': BinaryOp.GT, '>=': BinaryOp.GE},
    {'<<': BinaryOp.SHL, '>>': BinaryOp.SHR},
    {'+': BinaryOp.ADD, '-': BinaryOp.SUB},
    {'*': BinaryOp.MUL, '/': BinaryOp.DIV, '%': BinaryOp.MOD},
)

UNARY_OPS = {'-': UnaryOp.NEG, '!': UnaryOp.NOT, '~': UnaryOp.BIT_NOT}

BOOLEAN_LITERALS = {'true': 1, 'false': 0}
STORAGE_KINDS = {'GLOBALS': VariableKind.GLOBAL, 'REGS': VariableKind.REG}
EXTERN_SEPARATOR = ':'


def tokenize(source: str) -> List[Token]:
    tokens = []
    position = 0
    while position < len(source):
        match = TOKEN_PATTERN.match(source, position)
        if match is None:
            raise TSParseError(f'unexpected character {source[position]!r} at {position}')

        kind = match.lastgroup
        if kind not in ('space', 'comment'):
            tokens.append(Token(kind, match.group(), position))

        position = match.end()

    tokens.append(Token('end', '', position))
    return tokens


def unescape(literal: str) -> str:
    body = literal[1:-1]
    result = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == '\\' and index + 1 < len(body):
            escaped = body[index + 1]
            if escaped not in STRING_ESCAPES:
                raise TSParseError(f'unsupported escape \\{escaped}')

            result.append(STRING_ESCAPES[escaped])
            index += 2
            continue

        result.append(char)
        index += 1

    return ''.join(result)


class TypeScriptParser:
    '''Recursive descent parser for one generated function'''

    def __init__(self, source: str, start_addr: int = 0):
        self.tokens = tokenize(source)
        self.index = 0
        self.start_addr = start_addr
        self.variables: Dict[str, HLILVariable] = {}

    # ------------------------------------------------------------- tokens --

    def _peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def _next(self) -> Token:
        token = self._peek()
        self.index += 1
        return token

    def _at(self, text: str, offset: int = 0) -> bool:
        token = self._peek(offset)
        return token.kind in ('op', 'name') and token.text == text

    def _accept(self, text: str) -> bool:
        if self._at(text):
            self.index += 1
            return True

        return False

    def _expect(self, text: str) -> Token:
        token = self._next()
        if token.text != text or token.kind not in ('op', 'name'):
            raise TSParseError(f'expected {text!r}, found {token.text!r} at {token.position}')

        return token

    def _expect_name(self) -> str:
        token = self._next()
        if token.kind != 'name':
            raise TSParseError(f'expected a name, found {token.text!r} at {token.position}')

        return token.text

    # ----------------------------------------------------------- function --

    def parse_function(self) -> HighLevelILFunction:
        self._expect('function')
        func = HighLevelILFunction(self._expect_name(), self.start_addr)
        self._expect('(')
        while not self._at(')'):
            name = self._expect_name()
            self._parse_type()
            if self._accept('='):
                self._parse_expression()

            var = HLILVariable(name, kind = VariableKind.PARAM)
            self.variables[name] = var
            func.parameters.append(var)
            if not self._accept(','):
                break

        self._expect(')')
        self._parse_type()
        func.body = self._parse_block()
        if self._peek().kind != 'end':
            raise TSParseError(f'trailing input at {self._peek().position}')

        func.variables = [var for var in self.variables.values() if var.kind == VariableKind.LOCAL]
        return func

    def _parse_type(self):
        if not self._accept(':'):
            return

        self._expect_name()
        while self._accept('['):
            self._expect(']')

    # --------------------------------------------------------- statements --

    def _parse_block(self) -> HLILBlock:
        self._expect('{')
        block = HLILBlock()
        while not self._accept('}'):
            statement = self._parse_statement()
            if statement is not None:
                block.add_statement(statement)

        return block

    def _parse_statement(self) -> Optional[HLILStatement]:
        if self._accept('let'):
            name = self._expect_name()
            self._parse_type()
            self._expect(';')
            self.variables.setdefault(name, HLILVariable(name, kind = VariableKind.LOCAL))
            return None

        if self._at('{'):
            return self._parse_block()

        if self._accept('if'):
            return self._parse_if()

        if self._peek().kind == 'name' and self._at(':', 1) and self._at('while', 2):
            label = self._expect_name()
            self._expect(':')
            self._expect('while')
            return self._parse_while(label)

        if self._accept('while'):
            return self._parse_while(None)

        if self._accept('do'):
            body = self._parse_block()
            self._expect('while')
            condition = self._parse_parenthesized()
            self._expect(';')
            return HLILDoWhile(condition, body)

        if self._accept('switch'):
            return self._parse_switch()

        if self._at('break') or self._at('continue'):
            keyword = self._next().text
            label = self._expect_name() if self._peek().kind == 'name' else None
            self._expect(';')
            return HLILBreak(label) if keyword == 'break' else HLILContinue(label)

        if self._accept('return'):
            value = None if self._at(';') else self._parse_expression()
            self._expect(';')
            return HLILReturn(value)

        if self._at('deref_set'):
            self._next()
            self._expect('(')
            pointer = self._parse_expression()
            self._expect(',')
            value = self._parse_expression()
            self._expect(')')
            self._expect(';')
            return HLILAssign(HLILDeref(pointer), value)

        expression = self._parse_expression()
        if self._accept('='):
            if not isinstance(expression, HLILVar):
                raise TSParseError(f'cannot assign to {type(expression).__name__}')

            value = self._parse_expression()
            self._expect(';')
            return HLILAssign(expression, value)

        self._expect(';')
        return HLILExprStmt(expression)

    def _parse_parenthesized(self) -> HLILExpression:
        self._expect('(')
        expression = self._parse_expression()
        self._expect(')')
        return expression

    def _parse_if(self) -> HLILIf:
        condition = self._parse_parenthesized()
        true_block = self._parse_block()
        false_block = None
        if self._accept('else'):
            if self._accept('if'):
                false_block = HLILBlock([self._parse_if()])

            else:
                false_block = self._parse_block()

        return HLILIf(condition, true_block, false_block)

    def _parse_while(self, label: Optional[str]) -> HLILWhile:
        condition = self._parse_parenthesized()
        return HLILWhile(condition, self._parse_block(), label)

    def _parse_switch(self) -> HLILSwitch:
        scrutinee = self._parse_parenthesized()
        self._expect('{')
        cases = []
        while not self._accept('}'):
            if self._accept('default'):
                value = None

            else:
                self._expect('case')
                value = self._parse_expression()

            self._expect(':')
            cases.append(HLILSwitchCase(value, self._parse_block()))

        return HLILSwitch(scrutinee, cases)

    # -------------------------------------------------------- expressions --

    def _parse_expression(self, level: int = 0) -> HLILExpression:
        if level == len(BINARY_LEVELS):
            return self._parse_unary()

        operators = BINARY_LEVELS[level]
        lhs = self._parse_expression(level + 1)
        while self._peek().kind == 'op' and self._peek().text in operators:
            op = operators[self._next().text]
            lhs = HLILBinaryOp(op, lhs, self._parse_expression(level + 1))

        return lhs

    def _parse_unary(self) -> HLILExpression:
        token = self._peek()
        if token.kind == 'op' and token.text in UNARY_OPS:
            self._next()
            return HLILUnaryOp(UNARY_OPS[token.text], self._parse_unary())

        return self._parse_primary()

    def _parse_arguments(self) -> List[HLILExpression]:
        self._expect('(')
        args = []
        while not self._at(')'):
            args.append(self._parse_expression())
            if not self._accept(','):
                break

        self._expect(')')
        return args

    def _parse_primary(self) -> HLILExpression:
        token = self._next()
        if token.kind == 'number':
            return HLILConst(self._number(token.text))

        if token.kind == 'string':
            return HLILConst(unescape(token.text))

        if token.kind == 'op' and token.text == '(':
            expression = self._parse_expression()
            self._expect(')')
            return expression

        if token.kind != 'name':
            raise TSParseError(f'unexpected {token.text!r} at {token.position}')

        name = token.text
        if name in BOOLEAN_LITERALS:
            return HLILConst(BOOLEAN_LITERALS[name])

        if name in STORAGE_KINDS:
            self._expect('[')
            index_token = self._next()
            if index_token.kind != 'number':
                raise TSParseError(f'non-constant {name} index at {index_token.position}')

            self._expect(']')
            index = int(index_token.text, 0)
            return HLILVar(HLILVariable(f'{name}[{index}]', kind = STORAGE_KINDS[name], index = index))

        if self._at('('):
            return self._call(name, self._parse_arguments())

        return HLILVar(self._variable(name))

    def _call(self, name: str, args: List[HLILExpression]) -> HLILExpression:
        if name == 'deref':
            return HLILDeref(self._single(name, args))

        if name == 'addr_of':
            return HLILAddressOf(self._single(name, args))

        if name == 'int':
            return self._single(name, args)

        if name == 'syscall':
            subsystem, cmd = (self._literal(arg) for arg in args[:2])
            return HLILSyscall(subsystem, cmd, args[2:])

        if name == 'extern_call':
            target = self._literal(args[0])
            if not isinstance(target, str) or EXTERN_SEPARATOR not in target:
                raise TSParseError(f'bad extern_call target {target!r}')

            return HLILExternCall(target, args[1:])

        return HLILCall(name, args)

    @classmethod
    def _single(cls, name: str, args: List[HLILExpression]) -> HLILExpression:
        if len(args) != 1:
            raise TSParseError(f'{name} takes one argument, got {len(args)}')

        return args[0]

    @classmethod
    def _literal(cls, expr: HLILExpression):
        if not isinstance(expr, HLILConst):
            raise TSParseError(f'expected a literal, got {type(expr).__name__}')

        return expr.value

    @classmethod
    def _number(cls, text: str):
        if text.lower().startswith('0x'):
            return int(text, 16)

        if any(char in text for char in '.eE'):
            return float(text)

        return int(text)

    def _variable(self, name: str) -> HLILVariable:
        # JavaScript throws ReferenceError on a name that is neither a parameter nor declared with let
        if name not in self.variables:
            raise TSParseError(f'undeclared variable {name}')

        return self.variables[name]


def parse_typescript(source: str, start_addr: int = 0) -> HighLevelILFunction:
    return TypeScriptParser(source, start_addr).parse_function()

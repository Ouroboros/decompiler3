"""
TypeScript parser

Parses TypeScript code into HLIL for compilation back to bytecode.
Uses Node.js TypeScript compiler API for accurate parsing.
"""

import json
import subprocess
import tempfile
import os
from typing import List, Dict, Optional, Any, Union
from ..ir.base import IRFunction, IRVariable, IRBasicBlock, IRContext
from ..ir.hlil import *


class TypeScriptParser:
    """Parses TypeScript code into HLIL"""

    def __init__(self):
        self.context = None
        self.current_function = None
        self.variable_counter = 0
        self.block_counter = 0

    def parse_function(self, typescript_code: str, function_name: Optional[str] = None) -> IRFunction:
        """Parse TypeScript function into HLIL"""
        # Use Node.js to parse TypeScript and get AST
        ast = self._parse_with_nodejs(typescript_code)

        # Convert AST to HLIL
        if not ast:
            raise ValueError("Failed to parse TypeScript code")

        # Find function declaration
        function_node = self._find_function_node(ast, function_name)
        if not function_node:
            raise ValueError(f"Function '{function_name}' not found")

        # Create IR function
        ir_function = IRFunction(function_name or "anonymous")
        self.current_function = ir_function

        # Parse parameters
        if "parameters" in function_node:
            for param_node in function_node["parameters"]:
                param_name = param_node["name"]["text"]
                param_type = self._extract_type(param_node.get("type"))
                variable = IRVariable(param_name, 4, param_type)
                ir_function.parameters.append(variable)
                ir_function.variables[param_name] = variable

        # Parse return type
        if "type" in function_node:
            ir_function.return_type = self._extract_type(function_node["type"])

        # Parse function body
        if "body" in function_node:
            body_block = IRBasicBlock()
            ir_function.basic_blocks.append(body_block)
            self._parse_statement_list(function_node["body"]["statements"], body_block)

        return ir_function

    def parse_expression(self, typescript_expr: str) -> HLILExpression:
        """Parse a single TypeScript expression into HLIL"""
        # Wrap expression in a function for parsing
        wrapper_code = f"function temp() {{ return {typescript_expr}; }}"
        ast = self._parse_with_nodejs(wrapper_code)

        if not ast or not ast.get("statements"):
            raise ValueError("Failed to parse TypeScript expression")

        # Extract the return statement expression
        function_node = ast["statements"][0]
        return_stmt = function_node["body"]["statements"][0]
        expr_node = return_stmt["expression"]

        return self._parse_expression_node(expr_node)

    def _parse_with_nodejs(self, typescript_code: str) -> Optional[Dict]:
        """Use Node.js TypeScript compiler to parse code"""
        # Create a temporary file with TypeScript code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
            f.write(typescript_code)
            temp_file = f.name

        try:
            # Create a Node.js script to parse the TypeScript
            parser_script = '''
const ts = require('typescript');
const fs = require('fs');

const filename = process.argv[2];
const sourceCode = fs.readFileSync(filename, 'utf-8');

const sourceFile = ts.createSourceFile(
    filename,
    sourceCode,
    ts.ScriptTarget.Latest,
    true
);

// Convert AST to JSON (simplified)
function astToJson(node) {
    const result = {
        kind: ts.SyntaxKind[node.kind],
        text: node.getText ? node.getText() : undefined
    };

    // Add specific properties based on node type
    if (ts.isIdentifier(node)) {
        result.text = node.text;
    } else if (ts.isFunctionDeclaration(node)) {
        result.name = node.name ? node.name.text : undefined;
        result.parameters = node.parameters ? node.parameters.map(astToJson) : [];
        result.type = node.type ? astToJson(node.type) : undefined;
        result.body = node.body ? astToJson(node.body) : undefined;
    } else if (ts.isBlock(node)) {
        result.statements = node.statements.map(astToJson);
    } else if (ts.isReturnStatement(node)) {
        result.expression = node.expression ? astToJson(node.expression) : undefined;
    } else if (ts.isBinaryExpression(node)) {
        result.left = astToJson(node.left);
        result.operator = ts.SyntaxKind[node.operatorToken.kind];
        result.right = astToJson(node.right);
    } else if (ts.isCallExpression(node)) {
        result.expression = astToJson(node.expression);
        result.arguments = node.arguments.map(astToJson);
    } else if (ts.isNumericLiteral(node)) {
        result.value = parseFloat(node.text);
    } else if (ts.isStringLiteral(node)) {
        result.value = node.text.slice(1, -1); // Remove quotes
    } else if (ts.isParameter(node)) {
        result.name = astToJson(node.name);
        result.type = node.type ? astToJson(node.type) : undefined;
    }

    // Recursively process children for other node types
    if (!result.statements && !result.parameters && !result.arguments) {
        ts.forEachChild(node, (child) => {
            const childKey = ts.SyntaxKind[child.kind].toLowerCase();
            if (!result[childKey]) {
                result[childKey] = astToJson(child);
            }
        });
    }

    return result;
}

const ast = astToJson(sourceFile);
console.log(JSON.stringify(ast, null, 2));
            '''

            # Write parser script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as parser_file:
                parser_file.write(parser_script)
                parser_script_path = parser_file.name

            try:
                # Run Node.js parser
                result = subprocess.run(
                    ['node', parser_script_path, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    return json.loads(result.stdout)
                else:
                    print(f"TypeScript parsing error: {result.stderr}")
                    return None

            finally:
                os.unlink(parser_script_path)

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            # Fallback to simple parsing if Node.js is not available
            return self._simple_parse(typescript_code)

        finally:
            os.unlink(temp_file)

    def _simple_parse(self, typescript_code: str) -> Dict:
        """Simple fallback parser without Node.js"""
        # This is a very basic parser for demonstration
        # Real implementation would need a proper TypeScript parser
        lines = typescript_code.strip().split('\n')

        if lines[0].startswith('function'):
            # Parse function declaration
            func_line = lines[0]
            name_start = func_line.find('function') + 8
            name_end = func_line.find('(')
            function_name = func_line[name_start:name_end].strip()

            return {
                "kind": "SourceFile",
                "statements": [{
                    "kind": "FunctionDeclaration",
                    "name": {"text": function_name},
                    "parameters": [],
                    "body": {
                        "kind": "Block",
                        "statements": []
                    }
                }]
            }

        return {"kind": "SourceFile", "statements": []}

    def _find_function_node(self, ast: Dict, function_name: Optional[str]) -> Optional[Dict]:
        """Find function declaration in AST"""
        if "statements" in ast:
            for stmt in ast["statements"]:
                if stmt.get("kind") == "FunctionDeclaration":
                    if function_name is None or stmt.get("name", {}).get("text") == function_name:
                        return stmt
        return None

    def _extract_type(self, type_node: Optional[Dict]) -> Optional[str]:
        """Extract type information from AST node"""
        if not type_node:
            return None

        kind = type_node.get("kind")
        if kind == "NumberKeyword":
            return "number"
        elif kind == "StringKeyword":
            return "string"
        elif kind == "BooleanKeyword":
            return "boolean"
        elif kind == "VoidKeyword":
            return "void"
        elif kind == "AnyKeyword":
            return "any"
        else:
            return "any"

    def _parse_statement_list(self, statements: List[Dict], block: IRBasicBlock):
        """Parse a list of statements"""
        for stmt_node in statements:
            stmt = self._parse_statement_node(stmt_node)
            if stmt:
                block.add_instruction(stmt)

    def _parse_statement_node(self, node: Dict) -> Optional[HLILExpression]:
        """Parse a statement node"""
        kind = node.get("kind")

        if kind == "ReturnStatement":
            expr = None
            if "expression" in node:
                expr = self._parse_expression_node(node["expression"])
            return HLILReturn(expr)

        elif kind == "ExpressionStatement":
            return self._parse_expression_node(node["expression"])

        elif kind == "VariableStatement":
            # Handle variable declarations
            return self._parse_variable_declaration(node)

        elif kind == "IfStatement":
            return self._parse_if_statement(node)

        elif kind == "WhileStatement":
            return self._parse_while_statement(node)

        elif kind == "ForStatement":
            return self._parse_for_statement(node)

        elif kind == "BlockStatement":
            # Handle block statements
            statements = []
            for stmt in node.get("statements", []):
                parsed_stmt = self._parse_statement_node(stmt)
                if parsed_stmt:
                    statements.append(parsed_stmt)
            # Return first statement for now - proper block handling needed
            return statements[0] if statements else None

        return None

    def _parse_expression_node(self, node: Dict) -> HLILExpression:
        """Parse an expression node"""
        kind = node.get("kind")

        if kind == "NumericLiteral":
            value = node.get("value", 0)
            return HLILConstant(value, 4, "number")

        elif kind == "StringLiteral":
            value = node.get("value", "")
            return HLILConstant(value, len(value), "string")

        elif kind == "TrueKeyword":
            return HLILConstant(True, 1, "boolean")

        elif kind == "FalseKeyword":
            return HLILConstant(False, 1, "boolean")

        elif kind == "Identifier":
            name = node.get("text", "unknown")
            variable = self._get_or_create_variable(name)
            return HLILVariable(variable)

        elif kind == "BinaryExpression":
            return self._parse_binary_expression(node)

        elif kind == "UnaryExpression":
            return self._parse_unary_expression(node)

        elif kind == "CallExpression":
            return self._parse_call_expression(node)

        elif kind == "PropertyAccessExpression":
            return self._parse_property_access(node)

        elif kind == "ElementAccessExpression":
            return self._parse_element_access(node)

        else:
            # Unknown expression - create placeholder
            return HLILConstant(f"unknown_expr_{kind}", 4, "any")

    def _parse_binary_expression(self, node: Dict) -> HLILBinaryOp:
        """Parse binary expression"""
        left = self._parse_expression_node(node["left"])
        right = self._parse_expression_node(node["right"])

        operator = node.get("operator", "")
        op_map = {
            "PlusToken": OperationType.ADD,
            "MinusToken": OperationType.SUB,
            "AsteriskToken": OperationType.MUL,
            "SlashToken": OperationType.DIV,
            "PercentToken": OperationType.MOD,
            "AmpersandToken": OperationType.AND,
            "BarToken": OperationType.OR,
            "CaretToken": OperationType.XOR,
            "LessThanLessThanToken": OperationType.LSL,
            "GreaterThanGreaterThanToken": OperationType.ASR,
            "EqualsEqualsEqualsToken": OperationType.CMP_E,
            "ExclamationEqualsEqualsToken": OperationType.CMP_NE,
            "LessThanToken": OperationType.CMP_SLT,
            "LessThanEqualsToken": OperationType.CMP_SLE,
        }

        operation = op_map.get(operator, OperationType.ADD)
        return HLILBinaryOp(operation, left, right)

    def _parse_unary_expression(self, node: Dict) -> HLILUnaryOp:
        """Parse unary expression"""
        operand = self._parse_expression_node(node["operand"])
        operator = node.get("operator", "")

        if operator == "MinusToken":
            return HLILUnaryOp(OperationType.NEG, operand)
        elif operator == "ExclamationToken":
            return HLILUnaryOp(OperationType.NOT, operand)
        else:
            return HLILUnaryOp(OperationType.NOT, operand)  # Default

    def _parse_call_expression(self, node: Dict) -> HLILExpression:
        """Parse function call"""
        target = self._parse_expression_node(node["expression"])
        arguments = [self._parse_expression_node(arg) for arg in node.get("arguments", [])]

        # Check if it's a built-in function
        if isinstance(target, HLILVariable):
            builtin_name = self._map_to_builtin(target.variable.name)
            if builtin_name:
                return HLILBuiltinCall(builtin_name, arguments)

        return HLILCall(target, arguments)

    def _parse_property_access(self, node: Dict) -> HLILFieldAccess:
        """Parse property access (obj.field)"""
        base = self._parse_expression_node(node["expression"])
        field_name = node["name"]["text"]
        return HLILFieldAccess(base, field_name)

    def _parse_element_access(self, node: Dict) -> HLILArrayAccess:
        """Parse element access (obj[index])"""
        base = self._parse_expression_node(node["expression"])
        index = self._parse_expression_node(node["argumentExpression"])
        return HLILArrayAccess(base, index)

    def _parse_variable_declaration(self, node: Dict) -> HLILAssignment:
        """Parse variable declaration"""
        # Simplified variable declaration parsing
        # Real implementation would handle the full declaration syntax
        declarations = node.get("declarationList", {}).get("declarations", [])
        if declarations:
            decl = declarations[0]
            name = decl["name"]["text"]
            variable = self._get_or_create_variable(name)

            if "initializer" in decl:
                initializer = self._parse_expression_node(decl["initializer"])
                return HLILAssignment(HLILVariable(variable), initializer)

        # Return dummy assignment if parsing fails
        dummy_var = self._get_or_create_variable("temp")
        return HLILAssignment(HLILVariable(dummy_var), HLILConstant(0))

    def _parse_if_statement(self, node: Dict) -> HLILIf:
        """Parse if statement"""
        condition = self._parse_expression_node(node["expression"])

        true_body = []
        if "thenStatement" in node:
            stmt = self._parse_statement_node(node["thenStatement"])
            if stmt:
                true_body.append(stmt)

        false_body = []
        if "elseStatement" in node:
            stmt = self._parse_statement_node(node["elseStatement"])
            if stmt:
                false_body.append(stmt)

        return HLILIf(condition, true_body, false_body if false_body else None)

    def _parse_while_statement(self, node: Dict) -> HLILWhile:
        """Parse while statement"""
        condition = self._parse_expression_node(node["expression"])

        body = []
        if "statement" in node:
            stmt = self._parse_statement_node(node["statement"])
            if stmt:
                body.append(stmt)

        return HLILWhile(condition, body)

    def _parse_for_statement(self, node: Dict) -> HLILFor:
        """Parse for statement"""
        init = None
        if "initializer" in node:
            init = self._parse_expression_node(node["initializer"])

        condition = None
        if "condition" in node:
            condition = self._parse_expression_node(node["condition"])

        update = None
        if "incrementor" in node:
            update = self._parse_expression_node(node["incrementor"])

        body = []
        if "statement" in node:
            stmt = self._parse_statement_node(node["statement"])
            if stmt:
                body.append(stmt)

        return HLILFor(init, condition, update, body)

    def _get_or_create_variable(self, name: str) -> IRVariable:
        """Get or create a variable"""
        if self.current_function and name in self.current_function.variables:
            return self.current_function.variables[name]

        # Create new variable
        variable = IRVariable(name, 4, "any")
        if self.current_function:
            self.current_function.variables[name] = variable
        return variable

    def _map_to_builtin(self, function_name: str) -> Optional[str]:
        """Map TypeScript function to built-in"""
        mapping = {
            "Math.abs": "abs",
            "Math.pow": "pow",
            "Math.sqrt": "sqrt",
            "Math.sin": "sin",
            "Math.cos": "cos",
            "Math.log": "log",
            "console.log": "print",
            "typeof": "typeof",
        }
        return mapping.get(function_name)
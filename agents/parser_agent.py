import ast
import json
from typing import Dict, List, Any, Optional, Union
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Try importing tree-sitter with more specific error handling
TREE_SITTER_AVAILABLE = False
tree_sitter_error = None

try:
    from tree_sitter import Parser, Language
    TREE_SITTER_AVAILABLE = True
except ImportError as e:
    tree_sitter_error = f"tree_sitter not available: {e}"

try:
    from tree_sitter_languages import get_language
    TREE_SITTER_LANGUAGES_AVAILABLE = True
except ImportError as e:
    TREE_SITTER_LANGUAGES_AVAILABLE = False
    if tree_sitter_error is None:
        tree_sitter_error = f"tree_sitter_languages not available: {e}"

logger = logging.getLogger(__name__)

class LanguageType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    PHP = "php"
    RUBY = "ruby"
    TEXT = "text"

@dataclass
class ParseResult:
    """Structured result from parsing a file"""
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    exports: List[Dict[str, Any]]
    lines: int
    size: int
    error: Optional[str] = None
    tree_sitter_used: bool = False

class ParserAgent:
    """Enhanced parser agent with better error handling and type safety"""
    
    def __init__(self):
        self.parsers = {}
        self.supported_extensions = {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.java': LanguageType.JAVA,
            '.go': LanguageType.GO,
            '.rs': LanguageType.RUST,
            '.cpp': LanguageType.CPP,
            '.cc': LanguageType.CPP,
            '.cxx': LanguageType.CPP,
            '.c': LanguageType.C,
            '.php': LanguageType.PHP,
            '.rb': LanguageType.RUBY,
        }
        self._setup_parsers()
        logger.info("Enhanced ParserAgent initialized")

    def _setup_parsers(self):
        """Setup tree-sitter parsers with better error handling"""
        if not TREE_SITTER_AVAILABLE or not TREE_SITTER_LANGUAGES_AVAILABLE:
            logger.warning(f"tree-sitter not fully available: {tree_sitter_error}")
            logger.warning("Falling back to AST-only parsing for Python and regex-based parsing for other languages")
            return

        # List of languages to try loading
        language_map = {
            'python': LanguageType.PYTHON,
            'javascript': LanguageType.JAVASCRIPT,
            'typescript': LanguageType.TYPESCRIPT,
            'java': LanguageType.JAVA,
            'go': LanguageType.GO,
            'rust': LanguageType.RUST,
            'cpp': LanguageType.CPP,
            'c': LanguageType.C,
            'php': LanguageType.PHP,
            'ruby': LanguageType.RUBY
        }
        
        successful_loads = []
        failed_loads = []
        
        for ts_lang_name, lang_type in language_map.items():
            try:
                logger.debug(f"Attempting to load tree-sitter parser for: {ts_lang_name}")
                ts_lang = get_language(ts_lang_name)
                parser = Parser()
                parser.set_language(ts_lang)
                self.parsers[lang_type.value] = parser
                successful_loads.append(ts_lang_name)
                logger.debug(f"Successfully loaded parser for {ts_lang_name}")
            except Exception as e:
                failed_loads.append((ts_lang_name, str(e)))
                logger.debug(f"Could not load parser for {ts_lang_name}: {e}")
                continue
        
        if successful_loads:
            logger.info(f"Successfully loaded tree-sitter parsers for: {successful_loads}")
        else:
            logger.warning("No tree-sitter parsers could be loaded")
            
        if failed_loads:
            logger.debug(f"Failed to load parsers for: {[name for name, _ in failed_loads]}")

    def parse_files(self, code_files: Dict[str, str]) -> Dict[str, ParseResult]:
        """Parse all code files and return structured results"""
        parsed_results = {}
        
        for file_path, content in code_files.items():
            try:
                language_type = self._detect_language(file_path)
                parsed_data = self._parse_file(content, language_type, file_path)
                parsed_results[file_path] = parsed_data
                logger.debug(f"Successfully parsed {file_path} ({language_type.value})")
            except Exception as e:
                logger.error(f"Error parsing file {file_path}: {e}")
                parsed_results[file_path] = ParseResult(
                    language=self._detect_language(file_path).value,
                    functions=[],
                    classes=[],
                    imports=[],
                    variables=[],
                    exports=[],
                    lines=len(content.split('\n')) if content else 0,
                    size=len(content) if content else 0,
                    error=str(e)
                )
        
        logger.info(f"Parsed {len(parsed_results)} files")
        return parsed_results

    def _parse_file(self, content: str, language_type: LanguageType, file_path: str) -> ParseResult:
        """Parse a single file with enhanced error handling"""
        if not content.strip():
            return ParseResult(
                language=language_type.value,
                functions=[],
                classes=[],
                imports=[],
                variables=[],
                exports=[],
                lines=0,
                size=0
            )

        try:
            # Always try Python AST parsing first for Python files
            if language_type == LanguageType.PYTHON:
                return self._parse_python(content)
            
            # Try tree-sitter if available for this language
            elif language_type.value in self.parsers:
                return self._parse_with_tree_sitter(content, language_type)
            
            # Fallback to regex-based parsing for JS/TS
            elif language_type in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                return self._parse_javascript(content, language_type)
            
            # Generic parsing for other languages
            else:
                return self._parse_generic(content, language_type)
                
        except Exception as e:
            logger.error(f"Parsing failed for {file_path}: {e}")
            return ParseResult(
                language=language_type.value,
                functions=[],
                classes=[],
                imports=[],
                variables=[],
                exports=[],
                lines=len(content.split('\n')),
                size=len(content),
                error=str(e)
            )

    def _parse_with_tree_sitter(self, content: str, language_type: LanguageType) -> ParseResult:
        """Enhanced tree-sitter parsing with better node handling"""
        try:
            parser = self.parsers[language_type.value]
            tree = parser.parse(content.encode('utf-8'))
            
            result = ParseResult(
                language=language_type.value,
                functions=[],
                classes=[],
                imports=[],
                variables=[],
                exports=[],
                lines=len(content.split('\n')),
                size=len(content),
                tree_sitter_used=True
            )
            
            self._traverse_tree_sitter_node(tree.root_node, content, result)
            return result
            
        except Exception as e:
            logger.error(f"Tree-sitter parsing failed for {language_type.value}: {e}")
            # Fallback to regex-based parsing
            if language_type in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                return self._parse_javascript(content, language_type)
            else:
                return self._parse_generic(content, language_type)

    def _traverse_tree_sitter_node(self, node, content: str, result: ParseResult):
        """Traverse tree-sitter AST and extract information"""
        node_extractors = {
            'function_definition': self._extract_function_info,
            'function_declaration': self._extract_function_info,
            'method_definition': self._extract_function_info,
            'class_definition': self._extract_class_info,
            'class_declaration': self._extract_class_info,
            'import_statement': self._extract_import_info,
            'import_from_statement': self._extract_import_info,
        }
        
        if node.type in node_extractors:
            try:
                info = node_extractors[node.type](node, content)
                if info:
                    if node.type in ['function_definition', 'function_declaration', 'method_definition']:
                        result.functions.append(info)
                    elif node.type in ['class_definition', 'class_declaration']:
                        result.classes.append(info)
                    elif 'import' in node.type:
                        result.imports.append(info)
            except Exception as e:
                logger.debug(f"Error extracting info from {node.type}: {e}")
        
        # Recursively process children
        for child in node.children:
            self._traverse_tree_sitter_node(child, content, result)

    def _extract_function_info(self, node, content: str) -> Optional[Dict[str, Any]]:
        """Extract function information from tree-sitter node"""
        try:
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            
            return {
                "name": content[name_node.start_byte:name_node.end_byte],
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
                "type": node.type,
                "byte_start": node.start_byte,
                "byte_end": node.end_byte
            }
        except Exception as e:
            logger.debug(f"Error extracting function info: {e}")
            return None

    def _extract_class_info(self, node, content: str) -> Optional[Dict[str, Any]]:
        """Extract class information from tree-sitter node"""
        try:
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            
            return {
                "name": content[name_node.start_byte:name_node.end_byte],
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
                "type": node.type,
                "methods": []  # Could be populated by analyzing child nodes
            }
        except Exception as e:
            logger.debug(f"Error extracting class info: {e}")
            return None

    def _extract_import_info(self, node, content: str) -> Optional[Dict[str, Any]]:
        """Extract import information from tree-sitter node"""
        try:
            return {
                "line": node.start_point[0] + 1,
                "type": node.type,
                "raw": content[node.start_byte:node.end_byte].strip()
            }
        except Exception as e:
            logger.debug(f"Error extracting import info: {e}")
            return None

    def _parse_python(self, content: str) -> ParseResult:
        """Enhanced Python parsing with better error handling"""
        try:
            tree = ast.parse(content)
            
            result = ParseResult(
                language=LanguageType.PYTHON.value,
                functions=[],
                classes=[],
                imports=[],
                variables=[],
                exports=[],
                lines=len(content.split('\n')),
                size=len(content)
            )
            
            for node in ast.walk(tree):
                try:
                    self._process_python_node(node, result)
                except Exception as e:
                    logger.debug(f"Error processing Python node {type(node).__name__}: {e}")
                    continue
            
            return result
            
        except SyntaxError as e:
            logger.error(f"Python syntax error at line {e.lineno}: {e.msg}")
            return ParseResult(
                language=LanguageType.PYTHON.value,
                functions=[],
                classes=[],
                imports=[],
                variables=[],
                exports=[],
                lines=len(content.split('\n')),
                size=len(content),
                error=f"Syntax error at line {e.lineno}: {e.msg}"
            )

    def _process_python_node(self, node, result: ParseResult):
        """Process individual Python AST nodes"""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = {
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "line_start": node.lineno,
                "line_end": getattr(node, 'end_lineno', node.lineno),
                "docstring": ast.get_docstring(node),
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "complexity": self._estimate_complexity(node)
            }
            result.functions.append(func_info)
        
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "bases": [self._get_base_name(base) for base in node.bases],
                "line_start": node.lineno,
                "line_end": getattr(node, 'end_lineno', node.lineno),
                "docstring": ast.get_docstring(node),
                "methods": [],
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
            }
            
            # Extract methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = {
                        "name": item.name,
                        "args": [arg.arg for arg in item.args.args],
                        "line_start": item.lineno,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                        "decorators": [self._get_decorator_name(d) for d in item.decorator_list]
                    }
                    class_info["methods"].append(method_info)
            
            result.classes.append(class_info)
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "type": "import",
                        "line": node.lineno
                    })
            else:  # ImportFrom
                for alias in node.names:
                    result.imports.append({
                        "module": node.module or "",
                        "name": alias.name,
                        "alias": alias.asname,
                        "type": "from_import",
                        "line": node.lineno
                    })

    def _parse_javascript(self, content: str, language_type: LanguageType) -> ParseResult:
        """Enhanced JavaScript/TypeScript parsing"""
        import re
        
        result = ParseResult(
            language=language_type.value,
            functions=[],
            classes=[],
            imports=[],
            variables=[],
            exports=[],
            lines=len(content.split('\n')),
            size=len(content)
        )
        
        lines = content.split('\n')
        
        # Enhanced patterns
        patterns = {
            'function': re.compile(r'(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\(|(\w+)\s*:\s*(?:async\s+)?function|\(\s*\)\s*=>\s*|(\w+)\s*=\s*(?:async\s+)?\(.*?\)\s*=>)'),
            'class': re.compile(r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'),
            'import': re.compile(r'import\s+(?:(?:\{[^}]*\}|\w+|\*\s+as\s+\w+)\s+from\s+)?[\'"]([^\'"]+)[\'"]'),
            'export': re.compile(r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)'),
        }
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Functions
            func_match = patterns['function'].search(line)
            if func_match:
                func_name = next((g for g in func_match.groups() if g), None)
                if func_name:
                    result.functions.append({
                        "name": func_name,
                        "line": i,
                        "is_async": "async" in line,
                        "is_arrow": "=>" in line,
                        "is_exported": line.startswith("export")
                    })
            
            # Classes
            class_match = patterns['class'].search(line)
            if class_match:
                result.classes.append({
                    "name": class_match.group(1),
                    "extends": class_match.group(2),
                    "line": i,
                    "is_exported": line.startswith("export")
                })
            
            # Imports
            import_match = patterns['import'].search(line)
            if import_match:
                result.imports.append({
                    "module": import_match.group(1),
                    "line": i,
                    "raw": line
                })
            
            # Exports
            export_match = patterns['export'].search(line)
            if export_match:
                result.exports.append({
                    "name": export_match.group(1),
                    "line": i
                })
        
        return result

    def _parse_generic(self, content: str, language_type: LanguageType) -> ParseResult:
        """Generic parsing for unsupported languages"""
        lines = content.split('\n')
        
        return ParseResult(
            language=language_type.value,
            functions=[],
            classes=[],
            imports=[],
            variables=[],
            exports=[],
            lines=len(lines),
            size=len(content)
        )

    def _detect_language(self, file_path: str) -> LanguageType:
        """Detect programming language from file extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        return self.supported_extensions.get(extension, LanguageType.TEXT)

    def _get_decorator_name(self, decorator_node) -> str:
        """Extract decorator name from AST node"""
        if isinstance(decorator_node, ast.Name):
            return decorator_node.id
        elif isinstance(decorator_node, ast.Attribute):
            return decorator_node.attr
        else:
            return str(decorator_node)

    def _get_base_name(self, base_node) -> str:
        """Extract base class name from AST node"""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return f"{self._get_base_name(base_node.value)}.{base_node.attr}"
        else:
            return str(base_node)

    def _estimate_complexity(self, node) -> int:
        """Simple cyclomatic complexity estimation"""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def get_file_statistics(self, parsed_results: Dict[str, ParseResult]) -> Dict[str, Any]:
        """Generate statistics from parsed results"""
        total_files = len(parsed_results)
        total_lines = sum(result.get('lines', 0) if isinstance(result, dict) else result.lines 
                        for result in parsed_results.values())
        total_functions = sum(len(result.get('functions', []) if isinstance(result, dict) else result.functions) 
                            for result in parsed_results.values())
        total_classes = sum(len(result.get('classes', []) if isinstance(result, dict) else result.classes) 
                        for result in parsed_results.values())
        
        languages = {}
        for result in parsed_results.values():
            lang = result.get('language', 'unknown') if isinstance(result, dict) else result.language
            if lang not in languages:
                languages[lang] = {"files": 0, "lines": 0}
            languages[lang]["files"] += 1
            languages[lang]["lines"] += result.get('lines', 0) if isinstance(result, dict) else result.lines
        
        errors = [f for f, result in parsed_results.items() if 
                (result.get('error') if isinstance(result, dict) else result.error)]
        tree_sitter_usage = sum(1 for result in parsed_results.values() if 
                            (result.get('tree_sitter_used', False) if isinstance(result, dict) else result.tree_sitter_used))
        
        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "languages": languages,
            "files_with_errors": len(errors),
            "error_files": errors,
            "tree_sitter_files": tree_sitter_usage,
            "parsers_loaded": list(self.parsers.keys())
        }
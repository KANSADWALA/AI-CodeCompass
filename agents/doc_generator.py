from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

class DocumentationGenerator:
    def __init__(self, model_name: str = "llama2"):
        # Updated to use Ollama instead of ChatOpenAI
        self.llm = Ollama(model=model_name, temperature=0.1)
    
    def generate_documentation(self, code_files: Dict[str, str], parsed_data: Dict[str, Any]) -> str:
        """Generate comprehensive documentation for the codebase"""
        try:
            # Analyze codebase for documentation
            doc_structure = self._analyze_for_documentation(code_files, parsed_data)
            
            # Generate different sections
            readme = self._generate_readme(doc_structure)
            api_docs = self._generate_api_documentation(doc_structure)
            
            # Combine documentation
            full_docs = f"""# Project Documentation

            {readme}

            {api_docs}
            """
            return full_docs
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return f"Error generating documentation: {e}"
    
    def _analyze_for_documentation(self, code_files: Dict[str, str], parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codebase structure for documentation generation"""
        structure = {
            "project_name": self._infer_project_name(code_files),
            "main_modules": [],
            "classes": [],
            "functions": [],
            "dependencies": set(),
            "file_structure": {}
        }
        
        for file_path, data in parsed_data.items():
            if isinstance(data, dict) and "error" not in data:
                # Collect classes with their methods
                for cls in data.get("classes", []):
                    cls_info = {
                        "name": cls["name"],
                        "file": file_path,
                        "docstring": cls.get("docstring"),
                        "methods": cls.get("methods", []),
                        "line_start": cls.get("line_start")
                    }
                    structure["classes"].append(cls_info)
                
                # Collect standalone functions
                for func in data.get("functions", []):
                    func_info = {
                        "name": func["name"],
                        "file": file_path,
                        "args": func.get("args", []),
                        "docstring": func.get("docstring"),
                        "line_start": func.get("line_start")
                    }
                    structure["functions"].append(func_info)
                
                # Collect dependencies
                for imp in data.get("imports", []):
                    if imp.get("module"):
                        structure["dependencies"].add(imp["module"])
                
                # File structure
                structure["file_structure"][file_path] = {
                    "language": data.get("language"),
                    "lines": data.get("lines", 0),
                    "functions": len(data.get("functions", [])),
                    "classes": len(data.get("classes", []))
                }
        
        structure["dependencies"] = list(structure["dependencies"])
        return structure
    
    def _generate_readme(self, structure: Dict[str, Any]) -> str:
        """Generate README.md content"""
        prompt = f"""
        Generate a comprehensive README.md for a project with the following structure:

        **Project Name:** {structure['project_name']}

        **File Structure:**
        {self._format_file_structure(structure['file_structure'])}

        **Key Dependencies:** {', '.join(structure['dependencies'][:10])}

        **Classes Found:** {len(structure['classes'])}
        **Functions Found:** {len(structure['functions'])}

        Generate a README.md that includes:
        1. Project title and description
        2. Installation instructions
        3. Usage examples
        4. Project structure overview
        5. API reference (brief)
        6. Contributing guidelines

        Make it professional and informative.
        """
        
        # Updated to use Ollama's direct invoke method
        response = self.llm.invoke(prompt)
        return response
    
    def _generate_api_documentation(self, structure: Dict[str, Any]) -> str:
        """Generate API documentation"""
        if not structure["classes"] and not structure["functions"]:
            return "## API Documentation\n\nNo public API found in the codebase."
        
        prompt = f"""
        You are a technical writer creating API documentation.
        
        Generate API documentation for the following code elements:

        **Classes:**
        {self._format_classes_for_docs(structure['classes'])}

        **Functions:**
        {self._format_functions_for_docs(structure['functions'])}

        Create clear API documentation with:
        1. Class descriptions and methods
        2. Function signatures and descriptions
        3. Parameter descriptions
        4. Return value descriptions
        5. Usage examples where appropriate

        Use proper markdown formatting.
        """
        
        # Updated to use Ollama's direct invoke method
        response = self.llm.invoke(prompt)
        return f"## API Documentation\n\n{response}"
    
    def _infer_project_name(self, code_files: Dict[str, str]) -> str:
        """Infer project name from files"""
        # Look for setup.py, package.json, etc.
        for file_path in code_files:
            if "setup.py" in file_path:
                return "Python Project"
            elif "package.json" in file_path:
                return "JavaScript/Node.js Project"
            elif "pom.xml" in file_path:
                return "Java Maven Project"
            elif "Cargo.toml" in file_path:
                return "Rust Project"
        
        return "Code Project"
    
    def _format_file_structure(self, file_structure: Dict[str, Any]) -> str:
        """Format file structure for display"""
        lines = []
        for file_path, info in file_structure.items():
            lang = info.get("language", "unknown")
            lines_count = info.get("lines", 0)
            lines.append(f"- {file_path} ({lang}, {lines_count} lines)")
        return "\n".join(lines[:20])  # Limit to first 20 files
    
    def _format_classes_for_docs(self, classes: List[Dict]) -> str:
        """Format classes for documentation prompt"""
        if not classes:
            return "None"
        
        lines = []
        for cls in classes[:10]:  # Limit to first 10 classes
            methods = ", ".join([m["name"] for m in cls.get("methods", [])[:5]])
            lines.append(f"- {cls['name']} (in {cls['file']}): methods: {methods}")
        return "\n".join(lines)
    
    def _format_functions_for_docs(self, functions: List[Dict]) -> str:
        """Format functions for documentation prompt"""
        if not functions:
            return "None"
        
        lines = []
        for func in functions[:15]:  # Limit to first 15 functions
            args = ", ".join(func.get("args", []))
            lines.append(f"- {func['name']}({args}) in {func['file']}")
        return "\n".join(lines)
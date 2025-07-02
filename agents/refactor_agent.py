from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

class RefactorAgent:
    def __init__(self, model_name: str = "llama2"):
        # Updated to use Ollama instead of ChatOpenAI
        self.llm = Ollama(model=model_name, temperature=0.1)
    
    def analyze_code(self, code_files: Dict[str, str], parsed_data: Dict[str, Any]) -> str:
        """Analyze code and suggest refactoring improvements"""
        try:
            # Analyze code quality issues
            issues = self._identify_code_issues(code_files, parsed_data)
            
            # Generate refactoring suggestions
            suggestions = self._generate_refactor_suggestions(issues, code_files)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return f"Error analyzing code: {e}"
    
    def _identify_code_issues(self, code_files: Dict[str, str], parsed_data: Dict[str, Any]) -> Dict[str, List]:
        """Identify potential code quality issues"""
        issues = {
            "long_functions": [],
            "large_classes": [],
            "duplicate_code": [],
            "complex_functions": [],
            "naming_issues": [],
            "missing_docs": [],
            "code_smells": []
        }
        
        for file_path, data in parsed_data.items():
            if isinstance(data, dict) and "error" not in data:
                # Check for long functions (>50 lines)
                for func in data.get("functions", []):
                    if func.get("line_end", 0) - func.get("line_start", 0) > 50:
                        issues["long_functions"].append({
                            "file": file_path,
                            "function": func["name"],
                            "lines": func.get("line_end", 0) - func.get("line_start", 0)
                        })
                    
                    # Check for missing docstrings
                    if not func.get("docstring"):
                        issues["missing_docs"].append({
                            "file": file_path,
                            "function": func["name"],
                            "type": "function"
                        })
                    
                    # Check for complex parameter lists
                    if len(func.get("args", [])) > 5:
                        issues["complex_functions"].append({
                            "file": file_path,
                            "function": func["name"],
                            "parameter_count": len(func.get("args", []))
                        })
                
                # Check for large classes (>500 lines)
                for cls in data.get("classes", []):
                    if cls.get("line_end", 0) - cls.get("line_start", 0) > 500:
                        issues["large_classes"].append({
                            "file": file_path,
                            "class": cls["name"],
                            "lines": cls.get("line_end", 0) - cls.get("line_start", 0)
                        })
                    
                    # Check for missing class docstrings
                    if not cls.get("docstring"):
                        issues["missing_docs"].append({
                            "file": file_path,
                            "class": cls["name"],
                            "type": "class"
                        })
        
        # Check for potential duplicate code (simplified)
        self._check_duplicate_code(code_files, issues)
        
        return issues
    
    def _check_duplicate_code(self, code_files: Dict[str, str], issues: Dict[str, List]):
        """Simple duplicate code detection"""
        file_lines = {}
        
        for file_path, content in code_files.items():
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            file_lines[file_path] = lines
        
        # Look for identical functions or blocks (simplified)
        for file1, lines1 in file_lines.items():
            for file2, lines2 in file_lines.items():
                if file1 != file2:
                    # Check for similar line sequences
                    for i in range(len(lines1) - 5):
                        chunk1 = lines1[i:i+5]
                        for j in range(len(lines2) - 5):
                            chunk2 = lines2[j:j+5]
                            if chunk1 == chunk2 and len('\n'.join(chunk1)) > 100:
                                issues["duplicate_code"].append({
                                    "file1": file1,
                                    "file2": file2,
                                    "lines": 5,
                                    "content_preview": chunk1[0][:50] + "..."
                                })
                                break
    
    def _generate_refactor_suggestions(self, issues: Dict[str, List], code_files: Dict[str, str]) -> str:
        """Generate refactoring suggestions based on identified issues"""
        prompt = f"""
        You are a senior software engineer and code quality expert. Provide practical, actionable refactoring advice.
        
        Analyze the following code quality issues and provide specific refactoring suggestions:

        **Long Functions (>50 lines):**
        {self._format_issues(issues['long_functions'])}

        **Large Classes (>500 lines):**
        {self._format_issues(issues['large_classes'])}

        **Complex Functions (>5 parameters):**
        {self._format_issues(issues['complex_functions'])}

        **Missing Documentation:**
        {self._format_issues(issues['missing_docs'])}

        **Potential Duplicate Code:**
        {self._format_issues(issues['duplicate_code'])}

        **Project Overview:**
        - Total files: {len(code_files)}
        - Languages detected: {self._get_languages(code_files)}

        Please provide:
        1. **Priority Issues**: Most critical issues to address first
        2. **Specific Refactoring Suggestions**: Concrete steps for each issue type
        3. **Code Quality Improvements**: General recommendations
        4. **Architecture Recommendations**: High-level structural improvements
        5. **Best Practices**: Coding standards and patterns to adopt

        Format the response clearly with actionable recommendations.
        """
        
        # Updated to use Ollama's direct invoke method
        response = self.llm.invoke(prompt)
        return response
    
    def _format_issues(self, issues: List[Dict]) -> str:
        """Format issues for display in prompt"""
        if not issues:
            return "None found"
        
        lines = []
        for issue in issues[:10]:  # Limit to first 10 issues
            if "function" in issue:
                lines.append(f"- {issue['file']}: {issue['function']} ({issue.get('lines', 'N/A')} lines)")
            elif "class" in issue:
                lines.append(f"- {issue['file']}: {issue['class']} ({issue.get('lines', 'N/A')} lines)")
            elif "file1" in issue:
                lines.append(f"- Duplicate between {issue['file1']} and {issue['file2']}")
            else:
                lines.append(f"- {issue}")
        
        return "\n".join(lines)
    
    def _get_languages(self, code_files: Dict[str, str]) -> str:
        """Get list of programming languages in the codebase"""
        extensions = set()
        for file_path in code_files.keys():
            if '.' in file_path:
                ext = file_path.split('.')[-1]
                extensions.add(ext)
        
        return ", ".join(sorted(extensions))
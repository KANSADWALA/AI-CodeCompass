from typing import Dict, Any, List
from langchain_community.llms import Ollama
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import logging

# Local imports
from agents.parser_agent import ParseResult

logger = logging.getLogger(__name__)

class SummaryAgent:
    def __init__(self, model_name: str = "llama2"):
        # Updated to use Ollama instead of ChatOpenAI
        self.llm = Ollama(model=model_name, temperature=0.1)
        
    def generate_summary(self, code_files: Dict[str, str], parsed_data: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of the codebase"""
        try:
            # Analyze codebase structure
            analysis = self._analyze_codebase_structure(code_files, parsed_data)
            
            # Create summary prompt
            prompt = self._create_summary_prompt(analysis)
            
            # Generate summary - Updated for Ollama (single string input)
            full_prompt = f"""You are an expert code analyst. Provide clear, concise summaries of codebases.

                          {prompt}"""
            
            response = self.llm.invoke(full_prompt)

            if not response or not response.strip():
                logger.warning("Summary agent returned an empty response.")
                return "⚠️ Summary generation failed or returned empty content."

            return response
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"
        
    def _analyze_codebase_structure(self, code_files: Dict[str, str], parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure and content of the codebase"""
        logger.info(f"DEBUG: Analyzing {len(code_files)} code files, {len(parsed_data)} parsed")

        analysis = {
            "total_files": len(code_files),
            "languages": {},
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "file_types": {},
            "main_modules": [],
            "dependencies": set()
        }

        skipped_files = []

        for file_path, data in parsed_data.items():
            # Check if data is a valid type
            if not isinstance(data, (dict, ParseResult)):
                logger.warning(f"⚠️ Invalid type in parsed_data: {file_path} → {type(data)}")
            
            # Handle ParseResult objects
            if isinstance(data, ParseResult):
                if data.error:
                    skipped_files.append((file_path, f"Parse error: {data.error}"))
                    continue
                data = vars(data)

            if isinstance(data, dict):
                if data.get("error"):
                    skipped_files.append((file_path, f"Parse error: {data['error']}"))
                    continue
                data_dict = data
            else:
                skipped_files.append((file_path, f"Invalid data type: {type(data)}"))
                continue

            lang = data_dict.get("language", "unknown")
            analysis["languages"][lang] = analysis["languages"].get(lang, 0) + 1
            analysis["total_lines"] += data_dict.get("lines", 0)
            analysis["total_functions"] += len(data_dict.get("functions", []))
            analysis["total_classes"] += len(data_dict.get("classes", []))

            ext = file_path.split('.')[-1] if '.' in file_path else "no_ext"
            analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1

            if any(func.get("name") == "main" for func in data_dict.get("functions", [])):
                analysis["main_modules"].append(file_path)

            for imp in data_dict.get("imports", []):
                if imp.get("module"):
                    analysis["dependencies"].add(imp["module"])

        logger.info(f"✅ Parsed successfully: {analysis['total_files'] - len(skipped_files)} files")
        if skipped_files:
            logger.warning(f"⚠️ Skipped {len(skipped_files)} files due to errors or invalid format")
            for fp, reason in skipped_files[:5]:
                logger.warning(f" - {fp}: {reason}")

        analysis["dependencies"] = list(analysis["dependencies"])
        return analysis

    def _create_summary_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create a prompt for generating the codebase summary"""
        prompt = f"""
            Analyze this codebase and provide a comprehensive summary:

            **Codebase Statistics:**
            - Total files: {analysis['total_files']}
            - Total lines of code: {analysis['total_lines']}
            - Programming languages: {', '.join(analysis['languages'].keys())}
            - Functions: {analysis['total_functions']}
            - Classes: {analysis['total_classes']}

            **Language Distribution:**
            {self._format_dict(analysis['languages'])}

            **File Types:**
            {self._format_dict(analysis['file_types'])}

            **Main Modules:**
            {', '.join(analysis['main_modules']) if analysis['main_modules'] else 'None identified'}

            **Key Dependencies:**
            {', '.join(analysis['dependencies'][:10]) if analysis['dependencies'] else 'None identified'}

            Please provide:
            1. **Overview**: What this codebase appears to be (application type, purpose)
            2. **Architecture**: High-level structure and organization
            3. **Key Components**: Main modules and their purposes
            4. **Technology Stack**: Programming languages and frameworks used
            5. **Complexity Assessment**: Overall complexity and maintainability
            6. **Notable Patterns**: Any design patterns or architectural decisions observed

            Keep the summary concise but informative, suitable for developers who need to quickly understand this codebase.
            """
        return prompt
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for display in prompt"""
        if not d:
            return "None"
        return ', '.join([f"{k}: {v}" for k, v in d.items()])
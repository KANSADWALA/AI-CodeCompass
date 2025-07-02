import os
import requests
import zipfile
import tempfile
import shutil
from typing import Dict, List, Optional
from github import Github
import logging

logger = logging.getLogger(__name__)


class CodeFetcherAgent:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        print("GITHUB_TOKEN loaded:", bool(os.getenv("GITHUB_TOKEN")))
        self.github_client = Github(self.github_token) if self.github_token else None
        
        # Supported file extensions
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', 
            '.h', '.hpp', '.go', '.rs', '.php', '.rb', '.cs', '.swift',
            '.kt', '.scala', '.clj', '.hs', '.ml', '.r', '.m', '.vb',
            '.pl', '.sh', '.bat', '.ps1', '.yaml', '.yml', '.json',
            '.xml', '.html', '.css', '.scss', '.less', '.md', '.txt'
        }
    
    def fetch_from_url(self, repo_url: str) -> Dict[str, str]:
        """Fetch code from GitHub repository URL"""
        try:
            # Parse GitHub URL
            if "github.com" in repo_url:
                return self._fetch_from_github(repo_url)
            else:
                # Try to download as zip
                return self._fetch_from_zip_url(repo_url)
                
        except Exception as e:
            logger.error(f"Error fetching from URL {repo_url}: {e}")
            raise
    
    def fetch_from_path(self, repo_path: str) -> Dict[str, str]:
        """Fetch code from local directory"""
        try:
            code_files = {}
            
            for root, dirs, files in os.walk(repo_path):
                # Skip common directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                          d not in {'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist', 'target'}]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    # Check if file should be included
                    if self._should_include_file(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                code_files[relative_path] = content
                        except Exception as e:
                            logger.warning(f"Could not read file {file_path}: {e}")
            
            logger.info(f"Fetched {len(code_files)} files from {repo_path}")
            return code_files
            
        except Exception as e:
            logger.error(f"Error fetching from path {repo_path}: {e}")
            raise
    
    def _fetch_from_github(self, repo_url: str) -> Dict[str, str]:
        """Fetch code from GitHub repository"""
        try:
            # Extract owner and repo name from URL
            parts = repo_url.replace("https://github.com/", "").split("/")
            if len(parts) < 2:
                raise ValueError("Invalid GitHub URL format")
            
            owner, repo_name = parts[0], parts[1]
            
            if self.github_client:
                # Use GitHub API
                repo = self.github_client.get_repo(f"{owner}/{repo_name}")
                return self._fetch_github_contents(repo)
            else:
                # Download as zip
                zip_url = f"https://github.com/{owner}/{repo_name}/archive/main.zip"
                return self._fetch_from_zip_url(zip_url)
                
        except Exception as e:
            logger.error(f"Error fetching from GitHub {repo_url}: {e}")
            raise
    
    def _fetch_github_contents(self, repo) -> Dict[str, str]:
        """Fetch contents using GitHub API"""
        code_files = {}
        
        def fetch_recursive(contents):
            for content in contents:
                if content.type == "dir":
                    # Skip common directories
                    if content.name not in {'node_modules', '__pycache__', '.git', 'venv', 'env', 'build', 'dist', 'target'}:
                        try:
                            fetch_recursive(repo.get_contents(content.path))
                        except:
                            continue
                elif content.type == "file":
                    if self._should_include_file(content.name):
                        try:
                            file_content = content.decoded_content.decode('utf-8')
                            code_files[content.path] = file_content
                        except Exception as e:
                            logger.warning(f"Could not decode file {content.path}: {e}")
        
        try:
            contents = repo.get_contents("")
            fetch_recursive(contents)
        except Exception as e:
            logger.error(f"Error fetching GitHub contents: {e}")
            raise
        
        logger.info(f"Fetched {len(code_files)} files from GitHub")
        return code_files
    
    def _fetch_from_zip_url(self, zip_url: str) -> Dict[str, str]:
        """Download and extract zip file"""
        code_files = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download zip file
            response = requests.get(zip_url)
            response.raise_for_status()
            
            zip_path = os.path.join(temp_dir, "repo.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the extracted directory
            extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if extracted_dirs:
                repo_dir = os.path.join(temp_dir, extracted_dirs[0])
                code_files = self.fetch_from_path(repo_dir)
        
        return code_files
    
    def _should_include_file(self, file_path: str) -> bool:
        """Check if file should be included based on extension and size"""
        try:
            # Check extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in self.supported_extensions:
                return False
            
            # Check file size (skip very large files > 1MB)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 1024 * 1024:  # 1MB
                    return False
            
            return True
            
        except Exception:
            return False
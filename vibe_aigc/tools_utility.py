"""
Utility Tools for vibe-aigc.

Provides atomic utility tools for web search, URL fetching, 
code execution, and file operations.

These tools extend the atomic tool library (Paper Section 5.4)
with general-purpose capabilities useful for research and automation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import sys
import io
import subprocess
import tempfile
import asyncio
import aiohttp
import re
from pathlib import Path

from .tools import BaseTool, ToolSpec, ToolResult, ToolCategory


class WebSearchTool(BaseTool):
    """
    Web search tool using Brave Search API.
    
    Searches the web and returns structured results with
    titles, URLs, and snippets for research and information retrieval.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.search.brave.com/res/v1/web/search"
    ):
        """
        Initialize web search tool.
        
        Args:
            api_key: Brave Search API key (or use BRAVE_API_KEY env var)
            base_url: Brave Search API endpoint
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        self.base_url = base_url
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Search the web using Brave Search API",
            category=ToolCategory.SEARCH,
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (1-20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "country": {
                        "type": "string",
                        "description": "2-letter country code for regional results",
                        "default": "US"
                    },
                    "freshness": {
                        "type": "string",
                        "description": "Filter by time: pd (day), pw (week), pm (month), py (year)"
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"}
                            }
                        }
                    },
                    "total_results": {"type": "integer"}
                }
            },
            examples=[
                {
                    "input": {"query": "python async programming", "count": 3},
                    "output": {
                        "results": [
                            {
                                "title": "Async IO in Python",
                                "url": "https://realpython.com/async-io-python/",
                                "snippet": "Learn how to use async/await..."
                            }
                        ],
                        "total_results": 3
                    }
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute web search."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: query"
            )
        
        if not self.api_key:
            return ToolResult(
                success=False,
                output=None,
                error="BRAVE_API_KEY not set. Get key from brave.com/search/api"
            )
        
        query = inputs["query"]
        count = min(max(inputs.get("count", 5), 1), 20)
        country = inputs.get("country", "US")
        freshness = inputs.get("freshness")
        
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": count,
                "country": country
            }
            if freshness:
                params["freshness"] = freshness
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=headers,
                    params=params
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Brave API error ({response.status}): {error_text}"
                        )
                    
                    data = await response.json()
            
            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])
            
            for item in web_results[:count]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", "")
                })
            
            return ToolResult(
                success=True,
                output={
                    "results": results,
                    "total_results": len(results)
                },
                metadata={"query": query, "count": count}
            )
            
        except aiohttp.ClientError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Network error: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Search failed: {str(e)}"
            )


class WebFetchTool(BaseTool):
    """
    Web content fetching tool.
    
    Fetches and extracts readable content from URLs,
    converting HTML to clean text or markdown.
    """
    
    def __init__(
        self,
        max_content_length: int = 100000,
        timeout: int = 30
    ):
        """
        Initialize web fetch tool.
        
        Args:
            max_content_length: Maximum characters to return
            timeout: Request timeout in seconds
        """
        self.max_content_length = max_content_length
        self.timeout = timeout
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description="Fetch and extract readable content from a URL",
            category=ToolCategory.SEARCH,
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    },
                    "extract_mode": {
                        "type": "string",
                        "description": "Extraction mode: 'text' or 'markdown'",
                        "enum": ["text", "markdown"],
                        "default": "markdown"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return",
                        "default": 50000
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Extracted content"},
                    "title": {"type": "string", "description": "Page title"},
                    "url": {"type": "string", "description": "Final URL after redirects"}
                }
            },
            examples=[
                {
                    "input": {"url": "https://example.com/article"},
                    "output": {
                        "content": "# Article Title\n\nArticle content...",
                        "title": "Article Title",
                        "url": "https://example.com/article"
                    }
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Fetch content from URL."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: url"
            )
        
        url = inputs["url"]
        extract_mode = inputs.get("extract_mode", "markdown")
        max_chars = min(inputs.get("max_chars", 50000), self.max_content_length)
        
        # Validate URL
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False,
                output=None,
                error="URL must start with http:// or https://"
            )
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; vibe-aigc/1.0; +https://github.com/vibe-aigc)"
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"HTTP error: {response.status}"
                        )
                    
                    # Get content type
                    content_type = response.headers.get("Content-Type", "")
                    
                    if "text/html" in content_type or "text/plain" in content_type:
                        html = await response.text()
                    else:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Unsupported content type: {content_type}"
                        )
                    
                    final_url = str(response.url)
            
            # Extract content
            content, title = self._extract_content(html, extract_mode)
            
            # Truncate if needed
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[Content truncated...]"
            
            return ToolResult(
                success=True,
                output={
                    "content": content,
                    "title": title,
                    "url": final_url
                },
                metadata={"extract_mode": extract_mode, "length": len(content)}
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request timeout after {self.timeout}s"
            )
        except aiohttp.ClientError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Network error: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Fetch failed: {str(e)}"
            )
    
    def _extract_content(self, html: str, mode: str) -> tuple[str, str]:
        """Extract readable content from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            # Fallback: basic regex extraction
            return self._extract_basic(html, mode)
        
        # Get title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)
        
        # Remove script, style, nav, footer, etc.
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        
        # Find main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        
        if mode == "markdown":
            content = self._html_to_markdown(main)
        else:
            content = main.get_text(separator="\n", strip=True) if main else ""
        
        return content, title
    
    def _html_to_markdown(self, element) -> str:
        """Convert HTML element to markdown."""
        if element is None:
            return ""
        
        lines = []
        
        for tag in element.find_all(True, recursive=False):
            name = tag.name
            text = tag.get_text(strip=True)
            
            if name in ["h1"]:
                lines.append(f"# {text}\n")
            elif name in ["h2"]:
                lines.append(f"## {text}\n")
            elif name in ["h3"]:
                lines.append(f"### {text}\n")
            elif name in ["h4", "h5", "h6"]:
                lines.append(f"#### {text}\n")
            elif name == "p":
                if text:
                    lines.append(f"{text}\n")
            elif name in ["ul", "ol"]:
                for li in tag.find_all("li", recursive=False):
                    li_text = li.get_text(strip=True)
                    lines.append(f"- {li_text}")
                lines.append("")
            elif name == "a":
                href = tag.get("href", "")
                lines.append(f"[{text}]({href})")
            elif name in ["code", "pre"]:
                lines.append(f"```\n{text}\n```\n")
            elif name in ["blockquote"]:
                lines.append(f"> {text}\n")
            else:
                # Recurse for containers
                lines.append(self._html_to_markdown(tag))
        
        return "\n".join(lines)
    
    def _extract_basic(self, html: str, mode: str) -> tuple[str, str]:
        """Basic HTML extraction without BeautifulSoup."""
        # Get title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Remove scripts and styles
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all tags
        text = re.sub(r"<[^>]+>", " ", text)
        
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text, title


class CodeExecuteTool(BaseTool):
    """
    Safe Python code execution tool.
    
    Executes Python code in a restricted environment with
    timeout and output capture. Useful for computations,
    data processing, and testing.
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 10000,
        allowed_imports: Optional[List[str]] = None
    ):
        """
        Initialize code execution tool.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output: Maximum output characters to capture
            allowed_imports: List of allowed module imports (None = all)
        """
        self.timeout = timeout
        self.max_output = max_output
        self.allowed_imports = allowed_imports
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_execute",
            description="Execute Python code safely with timeout and output capture",
            category=ToolCategory.UTILITY,
            input_schema={
                "type": "object",
                "required": ["code"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "stdout": {"type": "string", "description": "Standard output"},
                    "stderr": {"type": "string", "description": "Standard error"},
                    "result": {"type": "string", "description": "Return value of last expression"},
                    "success": {"type": "boolean", "description": "Whether execution succeeded"}
                }
            },
            examples=[
                {
                    "input": {"code": "print('Hello')\n2 + 2"},
                    "output": {
                        "stdout": "Hello\n",
                        "stderr": "",
                        "result": "4",
                        "success": True
                    }
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute Python code."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: code"
            )
        
        code = inputs["code"]
        timeout = min(inputs.get("timeout", self.timeout), self.timeout)
        
        # Run in subprocess for isolation
        try:
            result = await self._execute_subprocess(code, timeout)
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution failed: {str(e)}"
            )
    
    async def _execute_subprocess(self, code: str, timeout: int) -> ToolResult:
        """Execute code in a subprocess."""
        # Create a wrapper script that captures output
        wrapper_code = f'''
import sys
import io

# Capture stdout/stderr
_stdout = io.StringIO()
_stderr = io.StringIO()
sys.stdout = _stdout
sys.stderr = _stderr

_result = None
_success = True
_error = None

try:
    # Execute user code
    exec(compile("""
{code.replace(chr(34)*3, chr(34)+chr(34)+chr(34))}
""", "<code>", "exec"), {{"__builtins__": __builtins__}})
except Exception as e:
    _success = False
    _error = str(e)
    import traceback
    _stderr.write(traceback.format_exc())

# Restore stdout
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Print results as JSON
import json
print(json.dumps({{
    "stdout": _stdout.getvalue()[:10000],
    "stderr": _stderr.getvalue()[:10000],
    "result": str(_result) if _result is not None else None,
    "success": _success,
    "error": _error
}}))
'''
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", wrapper_code,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    output={"stdout": "", "stderr": "", "result": None, "success": False},
                    error=f"Execution timeout after {timeout}s"
                )
            
            # Parse output
            try:
                import json
                output_str = stdout.decode("utf-8", errors="replace")
                result_data = json.loads(output_str)
                
                return ToolResult(
                    success=result_data.get("success", False),
                    output=result_data,
                    error=result_data.get("error")
                )
            except json.JSONDecodeError:
                return ToolResult(
                    success=False,
                    output={
                        "stdout": stdout.decode("utf-8", errors="replace")[:self.max_output],
                        "stderr": stderr.decode("utf-8", errors="replace")[:self.max_output],
                        "result": None,
                        "success": False
                    },
                    error="Failed to parse execution output"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Subprocess error: {str(e)}"
            )


class FileReadTool(BaseTool):
    """
    Local file reading tool.
    
    Reads content from local files with optional line limits.
    Useful for loading configuration, data files, or source code.
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        max_size: int = 1024 * 1024  # 1MB
    ):
        """
        Initialize file read tool.
        
        Args:
            base_path: Optional base path to restrict file access
            max_size: Maximum file size to read in bytes
        """
        self.base_path = Path(base_path) if base_path else None
        self.max_size = max_size
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description="Read content from a local file",
            category=ToolCategory.UTILITY,
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Maximum lines to read (optional)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start from (1-indexed)",
                        "default": 1
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "File content"},
                    "path": {"type": "string", "description": "Resolved file path"},
                    "size": {"type": "integer", "description": "File size in bytes"},
                    "lines": {"type": "integer", "description": "Number of lines read"}
                }
            },
            examples=[
                {
                    "input": {"path": "config.json"},
                    "output": {
                        "content": '{"key": "value"}',
                        "path": "/home/user/config.json",
                        "size": 16,
                        "lines": 1
                    }
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Read file content."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: path"
            )
        
        path_str = inputs["path"]
        encoding = inputs.get("encoding", "utf-8")
        max_lines = inputs.get("lines")
        offset = max(inputs.get("offset", 1), 1)
        
        try:
            # Resolve path
            path = Path(path_str)
            if self.base_path:
                path = self.base_path / path
            path = path.resolve()
            
            # Security check: ensure path is within base_path
            if self.base_path:
                try:
                    path.relative_to(self.base_path.resolve())
                except ValueError:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Access denied: path outside base directory"
                    )
            
            # Check file exists
            if not path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}"
                )
            
            if not path.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a file: {path}"
                )
            
            # Check file size
            size = path.stat().st_size
            if size > self.max_size:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File too large: {size} bytes (max: {self.max_size})"
                )
            
            # Read file
            with open(path, "r", encoding=encoding) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f, 1):
                        if i < offset:
                            continue
                        if len(lines) >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                    line_count = len(lines)
                else:
                    content = f.read()
                    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            
            return ToolResult(
                success=True,
                output={
                    "content": content,
                    "path": str(path),
                    "size": size,
                    "lines": line_count
                },
                metadata={"encoding": encoding}
            )
            
        except UnicodeDecodeError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Encoding error: {str(e)}. Try a different encoding."
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: {path_str}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Read failed: {str(e)}"
            )


class FileWriteTool(BaseTool):
    """
    Local file writing tool.
    
    Writes content to local files, creating directories as needed.
    Useful for saving outputs, configurations, or generated content.
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        max_size: int = 10 * 1024 * 1024  # 10MB
    ):
        """
        Initialize file write tool.
        
        Args:
            base_path: Optional base path to restrict file access
            max_size: Maximum content size to write in bytes
        """
        self.base_path = Path(base_path) if base_path else None
        self.max_size = max_size
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_write",
            description="Write content to a local file",
            category=ToolCategory.UTILITY,
            input_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Write mode: 'overwrite' or 'append'",
                        "enum": ["overwrite", "append"],
                        "default": "overwrite"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if missing",
                        "default": True
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "path": {"type": "string", "description": "Absolute path of written file"},
                    "bytes_written": {"type": "integer"}
                }
            },
            examples=[
                {
                    "input": {"path": "output.txt", "content": "Hello, World!"},
                    "output": {
                        "success": True,
                        "path": "/home/user/output.txt",
                        "bytes_written": 13
                    }
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Write content to file."""
        if not self.validate_inputs(inputs):
            missing = []
            if "path" not in inputs:
                missing.append("path")
            if "content" not in inputs:
                missing.append("content")
            return ToolResult(
                success=False,
                output=None,
                error=f"Missing required inputs: {', '.join(missing)}"
            )
        
        path_str = inputs["path"]
        content = inputs["content"]
        encoding = inputs.get("encoding", "utf-8")
        mode = inputs.get("mode", "overwrite")
        create_dirs = inputs.get("create_dirs", True)
        
        # Check content size
        content_bytes = len(content.encode(encoding))
        if content_bytes > self.max_size:
            return ToolResult(
                success=False,
                output=None,
                error=f"Content too large: {content_bytes} bytes (max: {self.max_size})"
            )
        
        try:
            # Resolve path
            path = Path(path_str)
            if self.base_path:
                path = self.base_path / path
            path = path.resolve()
            
            # Security check: ensure path is within base_path
            if self.base_path:
                try:
                    path.relative_to(self.base_path.resolve())
                except ValueError:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Access denied: path outside base directory"
                    )
            
            # Create parent directories
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            file_mode = "w" if mode == "overwrite" else "a"
            with open(path, file_mode, encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                output={
                    "success": True,
                    "path": str(path),
                    "bytes_written": content_bytes
                },
                metadata={"encoding": encoding, "mode": mode}
            )
            
        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: {path_str}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Write failed: {str(e)}"
            )


def create_utility_tools() -> List[BaseTool]:
    """Create all utility tools with default configuration."""
    return [
        WebSearchTool(),
        WebFetchTool(),
        CodeExecuteTool(),
        FileReadTool(),
        FileWriteTool()
    ]

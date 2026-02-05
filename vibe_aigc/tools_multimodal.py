"""
Multi-Modal Content Generation Tools.

Extends the base tool library with:
- Image generation (DALL-E, Replicate/Flux)
- Video generation (Replicate, Runway)
- Audio generation (ElevenLabs, music models)
- Web search (Brave, Google)
- Web scraping
"""

import os
import asyncio
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .tools import BaseTool, ToolResult, ToolSpec, ToolCategory


class ImageGenerationTool(BaseTool):
    """
    Image generation using DALL-E 3 or Replicate models.
    
    Supports:
    - OpenAI DALL-E 3
    - Replicate Flux
    - Replicate SDXL
    """
    
    def __init__(
        self,
        provider: str = "openai",  # openai, replicate
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model or self._default_model()
        self.api_key = api_key or os.getenv(self._api_key_env())
    
    def _default_model(self) -> str:
        defaults = {
            "openai": "dall-e-3",
            "replicate": "black-forest-labs/flux-1.1-pro"
        }
        return defaults.get(self.provider, "dall-e-3")
    
    def _api_key_env(self) -> str:
        envs = {
            "openai": "OPENAI_API_KEY",
            "replicate": "REPLICATE_API_TOKEN"
        }
        return envs.get(self.provider, "OPENAI_API_KEY")
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="image_generate",
            description="Generate images from text descriptions",
            category=ToolCategory.IMAGE,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Image description"},
                    "size": {"type": "string", "description": "Image size (e.g., 1024x1024)"},
                    "style": {"type": "string", "description": "Style (vivid, natural)"},
                    "quality": {"type": "string", "description": "Quality (standard, hd)"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "base64": {"type": "string"},
                    "revised_prompt": {"type": "string"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate an image."""
        if not self.validate_inputs(inputs):
            return ToolResult(success=False, output=None, error="Missing required input: prompt")
        
        prompt = inputs["prompt"]
        size = inputs.get("size", "1024x1024")
        style = inputs.get("style", "vivid")
        quality = inputs.get("quality", "standard")
        
        try:
            if self.provider == "openai":
                return await self._execute_dalle(prompt, size, style, quality)
            elif self.provider == "replicate":
                return await self._execute_replicate(prompt, size)
            else:
                return ToolResult(success=False, output=None, error=f"Unknown provider: {self.provider}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _execute_dalle(self, prompt: str, size: str, style: str, quality: str) -> ToolResult:
        """Generate image using DALL-E 3."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return ToolResult(success=False, output=None, error="openai package not installed")
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        response = await client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            style=style,
            quality=quality,
            n=1
        )
        
        image_data = response.data[0]
        
        return ToolResult(
            success=True,
            output={
                "url": image_data.url,
                "revised_prompt": image_data.revised_prompt
            },
            metadata={"model": self.model, "provider": "openai"}
        )
    
    async def _execute_replicate(self, prompt: str, size: str) -> ToolResult:
        """Generate image using Replicate (Flux)."""
        try:
            import replicate
        except ImportError:
            return ToolResult(success=False, output=None, error="replicate package not installed. Run: pip install replicate")
        
        # Parse size
        try:
            width, height = map(int, size.split("x"))
        except:
            width, height = 1024, 1024
        
        output = await asyncio.to_thread(
            replicate.run,
            self.model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "output_format": "png"
            }
        )
        
        # Replicate returns a FileOutput or URL
        url = str(output) if not isinstance(output, list) else str(output[0])
        
        return ToolResult(
            success=True,
            output={"url": url},
            metadata={"model": self.model, "provider": "replicate"}
        )


class VideoGenerationTool(BaseTool):
    """
    Video generation using Replicate or Runway models.
    
    Supports:
    - Replicate video models (minimax, runway-gen3)
    - Image-to-video
    - Text-to-video
    """
    
    def __init__(
        self,
        provider: str = "replicate",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model or "minimax/video-01"
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_generate",
            description="Generate videos from text or images",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string"},
                    "image_url": {"type": "string", "description": "Source image for I2V"},
                    "duration": {"type": "integer", "description": "Video duration in seconds"},
                    "fps": {"type": "integer", "description": "Frames per second"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "duration": {"type": "number"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate a video."""
        if not self.validate_inputs(inputs):
            return ToolResult(success=False, output=None, error="Missing required input: prompt")
        
        try:
            import replicate
        except ImportError:
            return ToolResult(success=False, output=None, error="replicate package not installed")
        
        prompt = inputs["prompt"]
        image_url = inputs.get("image_url")
        
        try:
            input_params = {"prompt": prompt}
            if image_url:
                input_params["first_frame_image"] = image_url
            
            output = await asyncio.to_thread(
                replicate.run,
                self.model,
                input=input_params
            )
            
            url = str(output) if not isinstance(output, list) else str(output[0])
            
            return ToolResult(
                success=True,
                output={"url": url},
                metadata={"model": self.model, "provider": self.provider}
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class AudioGenerationTool(BaseTool):
    """
    Audio generation for music and sound effects.
    
    Supports:
    - Replicate music models (MusicGen, Riffusion)
    - Sound effect generation
    """
    
    def __init__(
        self,
        model: str = "meta/musicgen:stereo-melody-large",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="audio_generate",
            description="Generate music or sound effects from text",
            category=ToolCategory.AUDIO,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string"},
                    "duration": {"type": "integer", "description": "Duration in seconds"},
                    "temperature": {"type": "number"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate audio."""
        if not self.validate_inputs(inputs):
            return ToolResult(success=False, output=None, error="Missing required input: prompt")
        
        try:
            import replicate
        except ImportError:
            return ToolResult(success=False, output=None, error="replicate package not installed")
        
        try:
            output = await asyncio.to_thread(
                replicate.run,
                self.model,
                input={
                    "prompt": inputs["prompt"],
                    "duration": inputs.get("duration", 10),
                    "model_version": "stereo-melody-large"
                }
            )
            
            url = str(output) if not isinstance(output, list) else str(output[0])
            
            return ToolResult(
                success=True,
                output={"url": url},
                metadata={"model": self.model}
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class TTSTool(BaseTool):
    """
    Text-to-Speech using ElevenLabs or OpenAI.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        voice: str = "alloy",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.voice = voice
        self.api_key = api_key or os.getenv(
            "ELEVENLABS_API_KEY" if provider == "elevenlabs" else "OPENAI_API_KEY"
        )
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="tts",
            description="Convert text to speech",
            category=ToolCategory.AUDIO,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "voice": {"type": "string"},
                    "speed": {"type": "number"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "audio_base64": {"type": "string"},
                    "url": {"type": "string"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Generate speech from text."""
        if not self.validate_inputs(inputs):
            return ToolResult(success=False, output=None, error="Missing required input: text")
        
        text = inputs["text"]
        voice = inputs.get("voice", self.voice)
        
        try:
            if self.provider == "openai":
                return await self._execute_openai_tts(text, voice)
            elif self.provider == "elevenlabs":
                return await self._execute_elevenlabs(text, voice)
            else:
                return ToolResult(success=False, output=None, error=f"Unknown provider: {self.provider}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def _execute_openai_tts(self, text: str, voice: str) -> ToolResult:
        """Generate speech using OpenAI TTS."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return ToolResult(success=False, output=None, error="openai package not installed")
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        response = await client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        return ToolResult(
            success=True,
            output={"audio_base64": audio_base64},
            metadata={"voice": voice, "provider": "openai"}
        )
    
    async def _execute_elevenlabs(self, text: str, voice: str) -> ToolResult:
        """Generate speech using ElevenLabs."""
        try:
            from elevenlabs import AsyncElevenLabs
        except ImportError:
            return ToolResult(success=False, output=None, error="elevenlabs package not installed")
        
        client = AsyncElevenLabs(api_key=self.api_key)
        
        audio = await client.generate(
            text=text,
            voice=voice,
            model="eleven_turbo_v2_5"
        )
        
        # Collect audio bytes
        audio_bytes = b"".join([chunk async for chunk in audio])
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        return ToolResult(
            success=True,
            output={"audio_base64": audio_base64},
            metadata={"voice": voice, "provider": "elevenlabs"}
        )


class SearchTool(BaseTool):
    """
    Web search using Brave Search API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search",
            description="Search the web for information",
            category=ToolCategory.SEARCH,
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer", "description": "Number of results"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Search the web."""
        if not self.validate_inputs(inputs):
            return ToolResult(success=False, output=None, error="Missing required input: query")
        
        if not self.api_key:
            return ToolResult(success=False, output=None, error="BRAVE_API_KEY not set")
        
        query = inputs["query"]
        count = inputs.get("count", 5)
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": count},
                    headers={"X-Subscription-Token": self.api_key}
                )
                response.raise_for_status()
                data = response.json()
            
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description")
                })
            
            return ToolResult(
                success=True,
                output={"results": results, "query": query},
                metadata={"count": len(results)}
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ScrapeTool(BaseTool):
    """
    Web page content extraction.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="scrape",
            description="Extract content from a web page",
            category=ToolCategory.SEARCH,
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string"},
                    "selector": {"type": "string", "description": "CSS selector"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "title": {"type": "string"}
                }
            }
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Scrape a web page."""
        if not self.validate_inputs(inputs):
            return ToolResult(success=False, output=None, error="Missing required input: url")
        
        url = inputs["url"]
        
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            return ToolResult(
                success=False, output=None,
                error="httpx and beautifulsoup4 required. Run: pip install httpx beautifulsoup4"
            )
        
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=30)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else ""
            
            # Limit text length
            if len(text) > 10000:
                text = text[:10000] + "..."
            
            return ToolResult(
                success=True,
                output={"text": text, "title": title, "url": url},
                metadata={"length": len(text)}
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


def register_multimodal_tools(registry: 'ToolRegistry') -> None:
    """Register all multi-modal tools in a registry."""
    from .tools import ToolRegistry
    
    registry.register(ImageGenerationTool(provider="openai"))
    registry.register(VideoGenerationTool())
    registry.register(AudioGenerationTool())
    registry.register(TTSTool(provider="openai"))
    registry.register(SearchTool())
    registry.register(ScrapeTool())


def create_full_registry() -> 'ToolRegistry':
    """Create a registry with all tools (text + multimodal)."""
    from .tools import create_default_registry
    
    registry = create_default_registry()
    register_multimodal_tools(registry)
    return registry

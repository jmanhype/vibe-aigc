"""Audio generation for music videos.

Supports:
- Music generation (Riffusion, MusicGen)
- Voice/TTS (ElevenLabs, local TTS)
- Sound effects
"""

import asyncio
import aiohttp
import base64
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .tools import BaseTool, ToolResult, ToolSpec, ToolCategory


@dataclass
class AudioConfig:
    """Configuration for audio generation."""
    provider: str = "riffusion"  # riffusion, musicgen, elevenlabs
    api_key: Optional[str] = None
    output_dir: str = "./audio_output"


class RiffusionBackend:
    """Music generation using Riffusion (via Replicate)."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.base_url = "https://api.replicate.com/v1"
    
    async def generate_music(
        self,
        prompt: str,
        duration: float = 8.0,  # seconds
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate music from a text prompt.
        
        Args:
            prompt: Description of the music (e.g., "upbeat electronic cyberpunk")
            duration: Length in seconds
            seed: Random seed for reproducibility
            
        Returns:
            Dict with audio URL or error
        """
        if not self.api_token:
            return {"error": "No Replicate API token. Set REPLICATE_API_TOKEN."}
        
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "version": "8cf61ea6c56afd61d8f5b9ffd14d7c216c0a93844ce2d82ac1c9ecc9c7f24e05",
            "input": {
                "prompt_a": prompt,
                "denoising": 0.75,
                "prompt_b": prompt,  # Same prompt for consistency
                "alpha": 0.5,
                "num_inference_steps": 50,
                "seed_image_id": "vibes"
            }
        }
        
        if seed is not None:
            payload["input"]["seed"] = seed
        
        try:
            async with aiohttp.ClientSession() as session:
                # Start prediction
                async with session.post(
                    f"{self.base_url}/predictions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status != 201:
                        error = await resp.text()
                        return {"error": f"Failed to start: {error}"}
                    result = await resp.json()
                    prediction_id = result.get("id")
                
                # Poll for completion
                for _ in range(60):  # Max 60 seconds
                    async with session.get(
                        f"{self.base_url}/predictions/{prediction_id}",
                        headers=headers
                    ) as resp:
                        result = await resp.json()
                        status = result.get("status")
                        
                        if status == "succeeded":
                            output = result.get("output", {})
                            return {
                                "audio_url": output.get("audio"),
                                "spectrogram_url": output.get("spectrogram"),
                                "prompt": prompt
                            }
                        elif status == "failed":
                            return {"error": result.get("error", "Generation failed")}
                    
                    await asyncio.sleep(1)
                
                return {"error": "Timeout waiting for generation"}
                
        except Exception as e:
            return {"error": str(e)}


class MusicGenBackend:
    """Music generation using Meta's MusicGen (via Replicate)."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.base_url = "https://api.replicate.com/v1"
    
    async def generate_music(
        self,
        prompt: str,
        duration: int = 8,
        model_version: str = "melody",  # small, medium, melody, large
        continuation: bool = False,
        input_audio: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate music using MusicGen.
        
        Args:
            prompt: Text description of desired music
            duration: Length in seconds (max 30)
            model_version: Model size/type
            continuation: Whether to continue from input_audio
            input_audio: URL of audio to continue from
            
        Returns:
            Dict with audio URL or error
        """
        if not self.api_token:
            return {"error": "No Replicate API token"}
        
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # MusicGen model on Replicate
        payload = {
            "version": "b05b1dff1d8c6dc63d14b0cdb42135378dcb87f6373b0d3d341ede46e59e2b38",
            "input": {
                "prompt": prompt,
                "duration": min(duration, 30),
                "model_version": model_version,
                "output_format": "mp3",
                "normalization_strategy": "peak"
            }
        }
        
        if continuation and input_audio:
            payload["input"]["continuation"] = True
            payload["input"]["input_audio"] = input_audio
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/predictions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status != 201:
                        return {"error": await resp.text()}
                    result = await resp.json()
                    prediction_id = result.get("id")
                
                # Poll for completion
                for _ in range(120):  # MusicGen can take longer
                    async with session.get(
                        f"{self.base_url}/predictions/{prediction_id}",
                        headers=headers
                    ) as resp:
                        result = await resp.json()
                        status = result.get("status")
                        
                        if status == "succeeded":
                            return {
                                "audio_url": result.get("output"),
                                "prompt": prompt,
                                "duration": duration
                            }
                        elif status == "failed":
                            return {"error": result.get("error")}
                    
                    await asyncio.sleep(1)
                
                return {"error": "Timeout"}
                
        except Exception as e:
            return {"error": str(e)}


class ElevenLabsBackend:
    """Voice and speech synthesis using ElevenLabs."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
    
    async def text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
        model_id: str = "eleven_monolingual_v1",
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ) -> Dict[str, Any]:
        """Generate speech from text.
        
        Args:
            text: Text to speak
            voice_id: ElevenLabs voice ID
            model_id: Model to use
            stability: Voice stability (0-1)
            similarity_boost: Voice similarity (0-1)
            
        Returns:
            Dict with audio data or error
        """
        if not self.api_key:
            return {"error": "No ElevenLabs API key"}
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/text-to-speech/{voice_id}",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        return {"error": await resp.text()}
                    
                    audio_data = await resp.read()
                    return {
                        "audio_data": base64.b64encode(audio_data).decode(),
                        "format": "mp3",
                        "text": text
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def list_voices(self) -> List[Dict[str, str]]:
        """List available voices."""
        if not self.api_key:
            return []
        
        headers = {"xi-api-key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/voices",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [
                            {"id": v["voice_id"], "name": v["name"]}
                            for v in data.get("voices", [])
                        ]
        except:
            pass
        return []


class MusicGenerationTool(BaseTool):
    """Tool for generating music."""
    
    def __init__(self, api_token: Optional[str] = None, backend: str = "musicgen"):
        self.api_token = api_token
        self.backend_name = backend
        
        if backend == "riffusion":
            self.backend = RiffusionBackend(api_token)
        else:
            self.backend = MusicGenBackend(api_token)
        
        self._spec = ToolSpec(
            name="music_generation",
            description="Generate music from text description",
            category=ToolCategory.AUDIO,
            input_schema={
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string", "description": "Music description"},
                    "duration": {"type": "integer", "default": 8},
                    "seed": {"type": "integer"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "audio_url": {"type": "string"},
                    "prompt": {"type": "string"}
                }
            }
        )
    
    @property
    def spec(self) -> ToolSpec:
        return self._spec
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        prompt = inputs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, output=None, error="No prompt")
        
        result = await self.backend.generate_music(
            prompt=prompt,
            duration=inputs.get("duration", 8),
            seed=inputs.get("seed")
        )
        
        if "error" in result:
            return ToolResult(success=False, output=None, error=result["error"])
        
        return ToolResult(
            success=True,
            output=result,
            metadata={"backend": self.backend_name}
        )


class TTSTool(BaseTool):
    """Tool for text-to-speech."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.backend = ElevenLabsBackend(api_key)
        self._spec = ToolSpec(
            name="text_to_speech",
            description="Convert text to speech audio",
            category=ToolCategory.AUDIO,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "voice_id": {"type": "string"},
                    "stability": {"type": "number", "default": 0.5},
                    "similarity_boost": {"type": "number", "default": 0.75}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "audio_data": {"type": "string", "description": "Base64 encoded audio"},
                    "format": {"type": "string"}
                }
            }
        )
    
    @property
    def spec(self) -> ToolSpec:
        return self._spec
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        text = inputs.get("text", "")
        if not text:
            return ToolResult(success=False, output=None, error="No text")
        
        result = await self.backend.text_to_speech(
            text=text,
            voice_id=inputs.get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
            stability=inputs.get("stability", 0.5),
            similarity_boost=inputs.get("similarity_boost", 0.75)
        )
        
        if "error" in result:
            return ToolResult(success=False, output=None, error=result["error"])
        
        return ToolResult(success=True, output=result)

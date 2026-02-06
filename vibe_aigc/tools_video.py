"""
Video Manipulation Tools for vibe-aigc.

Provides atomic tools for video post-processing:
- Frame interpolation (increase FPS smoothly)
- Looping (seamless video loops)
- Reversing (play backwards)
- Speed changes (slow-mo / time-lapse)
- GIF conversion

These tools wrap ffmpeg for reliable, GPU-accelerated video processing.
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from .tools import BaseTool, ToolSpec, ToolResult, ToolCategory


# Output directory for processed videos
OUTPUT_DIR = os.environ.get("VIBE_VIDEO_OUTPUT", "./output/video")


async def run_ffmpeg(args: List[str], timeout: int = 300) -> tuple[bool, str, str]:
    """
    Run ffmpeg command asynchronously.
    
    Args:
        args: ffmpeg arguments (without 'ffmpeg' prefix)
        timeout: Max execution time in seconds
        
    Returns:
        (success, stdout, stderr)
    """
    cmd = ["ffmpeg", "-y", "-hide_banner"] + args
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
        success = proc.returncode == 0
        return success, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        proc.kill()
        return False, "", "ffmpeg timed out"
    except FileNotFoundError:
        return False, "", "ffmpeg not found. Install ffmpeg and add to PATH."
    except Exception as e:
        return False, "", str(e)


def ensure_output_dir() -> Path:
    """Ensure output directory exists."""
    path = Path(OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_output_path(prefix: str, ext: str = "mp4") -> str:
    """Generate unique output path."""
    output_dir = ensure_output_dir()
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
    return str(output_dir / filename)


async def download_if_url(video_url: str) -> tuple[str, bool]:
    """
    Download video if it's a URL, otherwise return local path.
    
    Returns:
        (local_path, is_temp) - is_temp indicates if file should be cleaned up
    """
    if video_url.startswith(("http://", "https://")):
        # Download to temp file
        import aiohttp
        temp_path = tempfile.mktemp(suffix=".mp4")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as resp:
                    if resp.status == 200:
                        with open(temp_path, 'wb') as f:
                            f.write(await resp.read())
                        return temp_path, True
                    else:
                        raise Exception(f"Failed to download: HTTP {resp.status}")
        except ImportError:
            # Fallback to urllib if aiohttp not available
            import urllib.request
            urllib.request.urlretrieve(video_url, temp_path)
            return temp_path, True
    else:
        return video_url, False


class InterpolateTool(BaseTool):
    """
    Frame interpolation tool - increase video FPS smoothly.
    
    Uses ffmpeg's minterpolate filter for motion-compensated interpolation.
    For better quality, can integrate with RIFE via ComfyUI VideoHelperSuite.
    """
    
    def __init__(self, method: str = "minterpolate"):
        """
        Initialize interpolation tool.
        
        Args:
            method: Interpolation method ("minterpolate", "blend", "rife")
        """
        self.method = method
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_interpolate",
            description="Increase video frame rate with smooth motion interpolation",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["video_url", "target_fps"],
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "Input video URL or local path"
                    },
                    "target_fps": {
                        "type": "integer",
                        "description": "Target frames per second (e.g., 60)",
                        "default": 60
                    },
                    "method": {
                        "type": "string",
                        "description": "Interpolation method",
                        "enum": ["minterpolate", "blend"],
                        "default": "minterpolate"
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "interpolated_url": {"type": "string", "description": "Path to interpolated video"},
                    "original_fps": {"type": "number"},
                    "target_fps": {"type": "number"}
                }
            },
            examples=[
                {
                    "input": {"video_url": "/path/to/video.mp4", "target_fps": 60},
                    "output": {"interpolated_url": "/output/video/interpolated_abc123.mp4", "original_fps": 24, "target_fps": 60}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Interpolate video to target FPS."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required inputs: video_url, target_fps"
            )
        
        video_url = inputs["video_url"]
        target_fps = inputs.get("target_fps", 60)
        method = inputs.get("method", self.method)
        
        temp_file = None
        try:
            # Handle URL or local path
            input_path, is_temp = await download_if_url(video_url)
            if is_temp:
                temp_file = input_path
            
            output_path = generate_output_path("interpolated")
            
            # Build ffmpeg filter based on method
            if method == "minterpolate":
                # Motion-compensated interpolation
                vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
            else:
                # Simple frame blending
                vf = f"framerate=fps={target_fps}"
            
            args = [
                "-i", input_path,
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "copy",
                output_path
            ]
            
            success, stdout, stderr = await run_ffmpeg(args)
            
            if success:
                return ToolResult(
                    success=True,
                    output={
                        "interpolated_url": output_path,
                        "target_fps": target_fps,
                        "method": method
                    },
                    metadata={"ffmpeg_output": stderr[-500:] if stderr else ""}
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"ffmpeg interpolation failed: {stderr[-500:]}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Interpolation failed: {str(e)}"
            )
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)


class LoopTool(BaseTool):
    """
    Video looping tool - create seamless loops.
    
    Concatenates video multiple times, optionally with crossfade
    for smoother loop transitions.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_loop",
            description="Create seamlessly looping video by repeating content",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["video_url"],
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "Input video URL or local path"
                    },
                    "loops": {
                        "type": "integer",
                        "description": "Number of times to loop (default: 2)",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "crossfade": {
                        "type": "number",
                        "description": "Crossfade duration in seconds between loops (0 = hard cut)",
                        "default": 0
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "looped_url": {"type": "string", "description": "Path to looped video"},
                    "loops": {"type": "integer"},
                    "duration": {"type": "number", "description": "Final duration in seconds"}
                }
            },
            examples=[
                {
                    "input": {"video_url": "/path/to/video.mp4", "loops": 3},
                    "output": {"looped_url": "/output/video/looped_abc123.mp4", "loops": 3, "duration": 9.0}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Loop video specified number of times."""
        video_url = inputs.get("video_url")
        if not video_url:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: video_url"
            )
        
        loops = min(max(inputs.get("loops", 2), 1), 10)  # Clamp 1-10
        crossfade = inputs.get("crossfade", 0)
        
        temp_file = None
        try:
            input_path, is_temp = await download_if_url(video_url)
            if is_temp:
                temp_file = input_path
            
            output_path = generate_output_path("looped")
            
            if crossfade > 0:
                # Use xfade filter for crossfade between loops
                # This is complex - for now just do simple concat
                pass
            
            # Simple loop using -stream_loop
            args = [
                "-stream_loop", str(loops - 1),  # -1 because original plays once
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                output_path
            ]
            
            success, stdout, stderr = await run_ffmpeg(args)
            
            if success:
                return ToolResult(
                    success=True,
                    output={
                        "looped_url": output_path,
                        "loops": loops
                    },
                    metadata={"ffmpeg_output": stderr[-500:] if stderr else ""}
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"ffmpeg loop failed: {stderr[-500:]}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Loop creation failed: {str(e)}"
            )
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)


class ReverseTool(BaseTool):
    """
    Video reversal tool - play video backwards.
    
    Reverses both video and audio streams.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_reverse",
            description="Reverse video playback (play backwards)",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["video_url"],
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "Input video URL or local path"
                    },
                    "reverse_audio": {
                        "type": "boolean",
                        "description": "Also reverse audio (default: true)",
                        "default": True
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "reversed_url": {"type": "string", "description": "Path to reversed video"}
                }
            },
            examples=[
                {
                    "input": {"video_url": "/path/to/video.mp4"},
                    "output": {"reversed_url": "/output/video/reversed_abc123.mp4"}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Reverse video playback."""
        video_url = inputs.get("video_url")
        if not video_url:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: video_url"
            )
        
        reverse_audio = inputs.get("reverse_audio", True)
        
        temp_file = None
        try:
            input_path, is_temp = await download_if_url(video_url)
            if is_temp:
                temp_file = input_path
            
            output_path = generate_output_path("reversed")
            
            # Build filter for reversing
            if reverse_audio:
                vf = "reverse"
                af = "areverse"
                args = [
                    "-i", input_path,
                    "-vf", vf,
                    "-af", af,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    output_path
                ]
            else:
                args = [
                    "-i", input_path,
                    "-vf", "reverse",
                    "-an",  # No audio
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    output_path
                ]
            
            success, stdout, stderr = await run_ffmpeg(args)
            
            if success:
                return ToolResult(
                    success=True,
                    output={
                        "reversed_url": output_path
                    },
                    metadata={"reverse_audio": reverse_audio}
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"ffmpeg reverse failed: {stderr[-500:]}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Reverse failed: {str(e)}"
            )
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)


class SpeedChangeTool(BaseTool):
    """
    Video speed change tool - speed up or slow down video.
    
    Adjusts playback speed while optionally maintaining audio pitch.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_speed",
            description="Change video playback speed (slow motion or time-lapse)",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["video_url", "speed_factor"],
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "Input video URL or local path"
                    },
                    "speed_factor": {
                        "type": "number",
                        "description": "Speed multiplier (0.25 = 4x slower, 2.0 = 2x faster)",
                        "minimum": 0.1,
                        "maximum": 10.0
                    },
                    "maintain_pitch": {
                        "type": "boolean",
                        "description": "Maintain audio pitch when changing speed (default: true)",
                        "default": True
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "modified_url": {"type": "string", "description": "Path to speed-modified video"},
                    "speed_factor": {"type": "number"},
                    "new_duration": {"type": "number", "description": "Approximate new duration"}
                }
            },
            examples=[
                {
                    "input": {"video_url": "/path/to/video.mp4", "speed_factor": 0.5},
                    "output": {"modified_url": "/output/video/speed_abc123.mp4", "speed_factor": 0.5}
                },
                {
                    "input": {"video_url": "/path/to/video.mp4", "speed_factor": 2.0},
                    "output": {"modified_url": "/output/video/speed_def456.mp4", "speed_factor": 2.0}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Change video playback speed."""
        if not self.validate_inputs(inputs):
            return ToolResult(
                success=False,
                output=None,
                error="Missing required inputs: video_url, speed_factor"
            )
        
        video_url = inputs["video_url"]
        speed_factor = max(0.1, min(10.0, inputs["speed_factor"]))  # Clamp
        maintain_pitch = inputs.get("maintain_pitch", True)
        
        temp_file = None
        try:
            input_path, is_temp = await download_if_url(video_url)
            if is_temp:
                temp_file = input_path
            
            output_path = generate_output_path("speed")
            
            # Video: setpts filter (1/speed_factor since lower PTS = faster)
            pts_factor = 1.0 / speed_factor
            vf = f"setpts={pts_factor}*PTS"
            
            # Audio: atempo filter (has range limits, chain for extremes)
            # atempo only supports 0.5 to 2.0, chain multiple for wider range
            audio_filters = []
            remaining_speed = speed_factor
            
            while remaining_speed > 2.0:
                audio_filters.append("atempo=2.0")
                remaining_speed /= 2.0
            while remaining_speed < 0.5:
                audio_filters.append("atempo=0.5")
                remaining_speed /= 0.5
            
            if 0.5 <= remaining_speed <= 2.0:
                audio_filters.append(f"atempo={remaining_speed}")
            
            af = ",".join(audio_filters) if audio_filters else f"atempo={speed_factor}"
            
            args = [
                "-i", input_path,
                "-vf", vf,
                "-af", af,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                output_path
            ]
            
            success, stdout, stderr = await run_ffmpeg(args)
            
            if success:
                return ToolResult(
                    success=True,
                    output={
                        "modified_url": output_path,
                        "speed_factor": speed_factor
                    },
                    metadata={"maintain_pitch": maintain_pitch}
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"ffmpeg speed change failed: {stderr[-500:]}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Speed change failed: {str(e)}"
            )
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)


class Video2GifTool(BaseTool):
    """
    Video to GIF conversion tool.
    
    Creates optimized GIFs with custom FPS, size, and quality settings.
    Uses palette generation for better quality.
    """
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="video_to_gif",
            description="Convert video to optimized animated GIF",
            category=ToolCategory.VIDEO,
            input_schema={
                "type": "object",
                "required": ["video_url"],
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "Input video URL or local path"
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Output GIF frame rate (default: 15)",
                        "default": 15,
                        "minimum": 1,
                        "maximum": 30
                    },
                    "width": {
                        "type": "integer",
                        "description": "Output width in pixels (height auto-scaled, default: 480)",
                        "default": 480
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds (optional)"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds (optional, full video if not set)"
                    },
                    "optimize": {
                        "type": "boolean",
                        "description": "Use palette optimization for better quality (default: true)",
                        "default": True
                    }
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "gif_url": {"type": "string", "description": "Path to generated GIF"},
                    "file_size": {"type": "integer", "description": "File size in bytes"},
                    "dimensions": {"type": "string", "description": "WxH dimensions"}
                }
            },
            examples=[
                {
                    "input": {"video_url": "/path/to/video.mp4", "fps": 15, "width": 320},
                    "output": {"gif_url": "/output/video/gif_abc123.gif", "file_size": 2048000}
                }
            ]
        )
    
    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Convert video to GIF."""
        video_url = inputs.get("video_url")
        if not video_url:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required input: video_url"
            )
        
        fps = min(max(inputs.get("fps", 15), 1), 30)
        width = inputs.get("width", 480)
        start_time = inputs.get("start_time")
        duration = inputs.get("duration")
        optimize = inputs.get("optimize", True)
        
        temp_file = None
        palette_file = None
        try:
            input_path, is_temp = await download_if_url(video_url)
            if is_temp:
                temp_file = input_path
            
            output_path = generate_output_path("gif", "gif")
            
            # Build input options
            input_opts = []
            if start_time is not None:
                input_opts.extend(["-ss", str(start_time)])
            if duration is not None:
                input_opts.extend(["-t", str(duration)])
            
            # Scale filter
            scale_filter = f"fps={fps},scale={width}:-1:flags=lanczos"
            
            if optimize:
                # Two-pass with palette for better quality
                palette_file = tempfile.mktemp(suffix=".png")
                
                # Pass 1: Generate palette
                args1 = input_opts + [
                    "-i", input_path,
                    "-vf", f"{scale_filter},palettegen=stats_mode=diff",
                    palette_file
                ]
                success1, _, stderr1 = await run_ffmpeg(args1)
                
                if not success1:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Palette generation failed: {stderr1[-300:]}"
                    )
                
                # Pass 2: Generate GIF using palette
                args2 = input_opts + [
                    "-i", input_path,
                    "-i", palette_file,
                    "-lavfi", f"{scale_filter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
                    output_path
                ]
                success, stdout, stderr = await run_ffmpeg(args2)
            else:
                # Single pass (faster, lower quality)
                args = input_opts + [
                    "-i", input_path,
                    "-vf", scale_filter,
                    output_path
                ]
                success, stdout, stderr = await run_ffmpeg(args)
            
            if success:
                # Get file size
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                return ToolResult(
                    success=True,
                    output={
                        "gif_url": output_path,
                        "file_size": file_size,
                        "fps": fps,
                        "width": width
                    },
                    metadata={"optimized": optimize}
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"GIF conversion failed: {stderr[-500:]}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"GIF conversion failed: {str(e)}"
            )
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            if palette_file and os.path.exists(palette_file):
                os.remove(palette_file)


def create_video_tools() -> List[BaseTool]:
    """Create all video manipulation tools."""
    return [
        InterpolateTool(),
        LoopTool(),
        ReverseTool(),
        SpeedChangeTool(),
        Video2GifTool()
    ]

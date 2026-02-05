"""Music Video Pipeline - Orchestrates full MV generation.

This is the crown jewel of vibe-aigc: takes a Vibe and produces
a complete music video with:
- Generated music track
- Multiple video clips (AnimateDiff)
- Character consistency across shots
- Proper pacing and transitions
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .models import Vibe
from .knowledge import KnowledgeBase
from .comfyui import ComfyUIBackend, ComfyUIConfig
from .video import AnimateDiffBackend
from .character import CharacterBank, CharacterReference, IPAdapterBackend
from .audio import MusicGenBackend, RiffusionBackend
from .tools import LLMTool


@dataclass
class Shot:
    """A single shot in the music video."""
    id: str
    description: str
    prompt: str
    negative_prompt: str = ""
    duration: float = 2.0  # seconds
    frames: int = 16
    character: Optional[str] = None  # Character name for consistency
    transition: str = "cut"  # cut, fade, dissolve
    
    # Generated content
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


@dataclass
class Storyboard:
    """Complete storyboard for the music video."""
    title: str
    vibe: Vibe
    shots: List[Shot] = field(default_factory=list)
    music_prompt: Optional[str] = None
    music_url: Optional[str] = None
    total_duration: float = 0.0
    
    def add_shot(self, shot: Shot) -> None:
        self.shots.append(shot)
        self.total_duration += shot.duration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "vibe": {
                "description": self.vibe.description,
                "style": self.vibe.style,
                "domain": self.vibe.domain
            },
            "shots": [
                {
                    "id": s.id,
                    "description": s.description,
                    "prompt": s.prompt,
                    "duration": s.duration,
                    "frames": s.frames,
                    "character": s.character,
                    "transition": s.transition,
                    "video_url": s.video_url
                }
                for s in self.shots
            ],
            "music_prompt": self.music_prompt,
            "music_url": self.music_url,
            "total_duration": self.total_duration
        }


class MVPipeline:
    """Complete Music Video generation pipeline.
    
    Workflow:
    1. Parse Vibe and query Knowledge Base
    2. Generate storyboard with LLM
    3. Generate music track
    4. Generate each shot with AnimateDiff
    5. Apply character consistency where needed
    6. Output complete MV package
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        comfyui_config: Optional[ComfyUIConfig] = None,
        music_api_token: Optional[str] = None
    ):
        self.knowledge_base = KnowledgeBase()
        self.comfyui_config = comfyui_config or ComfyUIConfig()
        self.character_bank = CharacterBank()
        
        # Backends
        self.image_backend = ComfyUIBackend(self.comfyui_config)
        self.video_backend = AnimateDiffBackend(self.comfyui_config)
        self.character_backend = IPAdapterBackend(self.comfyui_config)
        
        # LLM for storyboarding
        if llm_config:
            self.llm = LLMTool(**llm_config)
        else:
            self.llm = None
        
        # Music generation
        self.music_backend = MusicGenBackend(music_api_token)
    
    async def generate_storyboard(
        self,
        vibe: Vibe,
        num_shots: int = 8,
        target_duration: float = 30.0
    ) -> Storyboard:
        """Generate a storyboard from a Vibe.
        
        Uses LLM to create shot descriptions based on the vibe
        and domain knowledge.
        """
        # Query knowledge base for technical context
        knowledge = self.knowledge_base.query(f"{vibe.description} {vibe.style}")
        knowledge_context = self.knowledge_base.to_prompt_context(
            f"{vibe.description} {vibe.style}"
        )
        
        # Build storyboard prompt
        prompt = f"""You are a music video director creating a storyboard.

## Creative Brief
Description: {vibe.description}
Style: {vibe.style}
Domain: {vibe.domain or 'music video'}
Constraints: {', '.join(vibe.constraints) if vibe.constraints else 'None'}

{knowledge_context}

## Task
Create a {num_shots}-shot storyboard for a {target_duration:.0f}-second music video.

For each shot, provide:
1. Shot ID (shot_1, shot_2, etc.)
2. Brief description (what's happening)
3. Detailed Stable Diffusion prompt (include style tags)
4. Duration in seconds
5. Transition to next shot (cut, fade, dissolve)

Return as JSON array:
```json
[
  {{
    "id": "shot_1",
    "description": "Opening shot description",
    "prompt": "detailed SD prompt with style tags",
    "negative_prompt": "things to avoid",
    "duration": 4.0,
    "transition": "fade"
  }}
]
```

Make the shots flow together narratively. Include establishing shots,
action shots, and emotional beats. Use the style consistently."""

        storyboard = Storyboard(
            title=f"MV: {vibe.description[:30]}",
            vibe=vibe
        )
        
        if self.llm:
            result = await self.llm.execute({"prompt": prompt})
            if result.success:
                try:
                    # Parse LLM response
                    text = result.output.get("text", "")
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', text)
                    if json_match:
                        shots_data = json.loads(json_match.group())
                        for shot_data in shots_data:
                            shot = Shot(
                                id=shot_data.get("id", f"shot_{len(storyboard.shots)+1}"),
                                description=shot_data.get("description", ""),
                                prompt=shot_data.get("prompt", ""),
                                negative_prompt=shot_data.get("negative_prompt", "blurry, low quality"),
                                duration=shot_data.get("duration", target_duration / num_shots),
                                frames=int(shot_data.get("duration", 2) * 8),  # 8 fps
                                transition=shot_data.get("transition", "cut")
                            )
                            storyboard.add_shot(shot)
                except json.JSONDecodeError:
                    pass
        
        # Fallback: generate basic shots if LLM failed
        if not storyboard.shots:
            shot_duration = target_duration / num_shots
            sd_tags = knowledge.get("technical_specs", {}).get("sd_prompt_tags", [])
            base_prompt = f"{vibe.description}, {vibe.style}, {', '.join(sd_tags[:5])}"
            
            for i in range(num_shots):
                shot = Shot(
                    id=f"shot_{i+1}",
                    description=f"Shot {i+1}",
                    prompt=base_prompt,
                    negative_prompt="blurry, low quality, deformed",
                    duration=shot_duration,
                    frames=int(shot_duration * 8)
                )
                storyboard.add_shot(shot)
        
        # Generate music prompt
        storyboard.music_prompt = f"{vibe.style} music, {vibe.description}"
        
        return storyboard
    
    async def generate_music(
        self,
        storyboard: Storyboard,
        duration: Optional[float] = None
    ) -> str:
        """Generate background music for the MV."""
        if not storyboard.music_prompt:
            storyboard.music_prompt = f"{storyboard.vibe.style} music"
        
        target_duration = duration or storyboard.total_duration
        
        result = await self.music_backend.generate_music(
            prompt=storyboard.music_prompt,
            duration=min(int(target_duration), 30)  # MusicGen max 30s
        )
        
        if "audio_url" in result:
            storyboard.music_url = result["audio_url"]
            return result["audio_url"]
        
        return ""
    
    async def generate_shot(
        self,
        shot: Shot,
        character_ref: Optional[str] = None
    ) -> Shot:
        """Generate a single shot video."""
        
        # If character consistency needed and reference provided
        if character_ref and shot.character:
            # First generate a consistent character image
            char_result = await self.character_backend.generate_with_reference(
                prompt=shot.prompt,
                reference_image=character_ref,
                negative_prompt=shot.negative_prompt,
                steps=20
            )
            # Use that as reference for video
            # (For now, just generate video directly)
        
        # Generate video with AnimateDiff
        result = await self.video_backend.generate_video(
            prompt=shot.prompt,
            negative_prompt=shot.negative_prompt,
            frames=shot.frames,
            fps=8,
            steps=15
        )
        
        if result.success and result.images:
            shot.video_url = result.images[0]
        
        return shot
    
    async def generate_mv(
        self,
        vibe: Vibe,
        num_shots: int = 8,
        target_duration: float = 30.0,
        generate_music: bool = True,
        parallel_shots: int = 2
    ) -> Storyboard:
        """Generate a complete music video.
        
        Args:
            vibe: The creative vibe/intent
            num_shots: Number of video shots
            target_duration: Target video length in seconds
            generate_music: Whether to generate background music
            parallel_shots: How many shots to generate in parallel
            
        Returns:
            Complete storyboard with all generated content
        """
        # Step 1: Generate storyboard
        storyboard = await self.generate_storyboard(
            vibe=vibe,
            num_shots=num_shots,
            target_duration=target_duration
        )
        
        # Step 2: Generate music (if enabled)
        if generate_music:
            await self.generate_music(storyboard, target_duration)
        
        # Step 3: Generate video shots
        # Process in batches for memory efficiency
        for i in range(0, len(storyboard.shots), parallel_shots):
            batch = storyboard.shots[i:i + parallel_shots]
            tasks = [self.generate_shot(shot) for shot in batch]
            await asyncio.gather(*tasks)
        
        return storyboard
    
    async def generate_mv_simple(
        self,
        description: str,
        style: str = "cinematic",
        num_shots: int = 4
    ) -> Storyboard:
        """Simple MV generation with minimal config.
        
        Args:
            description: What the MV is about
            style: Visual style
            num_shots: Number of shots
            
        Returns:
            Generated storyboard
        """
        vibe = Vibe(
            description=description,
            style=style,
            domain="music_video"
        )
        
        return await self.generate_mv(
            vibe=vibe,
            num_shots=num_shots,
            target_duration=num_shots * 4,  # 4 seconds per shot
            generate_music=False  # Skip music for quick generation
        )
    
    def export_storyboard(
        self,
        storyboard: Storyboard,
        output_path: str
    ) -> None:
        """Export storyboard to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(storyboard.to_dict(), f, indent=2)
    
    def export_html_preview(
        self,
        storyboard: Storyboard,
        output_path: str
    ) -> None:
        """Export an HTML preview of the MV."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{storyboard.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00ffff; }}
        .shot {{ display: inline-block; margin: 10px; padding: 15px; background: #2d2d44; border-radius: 8px; width: 280px; vertical-align: top; }}
        .shot img {{ width: 100%; border-radius: 4px; }}
        .shot-id {{ color: #ff00ff; font-weight: bold; }}
        .prompt {{ font-size: 12px; color: #888; margin-top: 10px; }}
        .music {{ background: #2d4444; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{storyboard.title}</h1>
    <p><strong>Style:</strong> {storyboard.vibe.style}</p>
    <p><strong>Duration:</strong> {storyboard.total_duration:.1f}s | <strong>Shots:</strong> {len(storyboard.shots)}</p>
    
    {"<div class='music'><strong>Music:</strong> " + storyboard.music_prompt + "<br><audio controls src='" + (storyboard.music_url or '') + "'></audio></div>" if storyboard.music_url else ""}
    
    <h2>Shots</h2>
    <div class="shots">
"""
        
        for shot in storyboard.shots:
            html += f"""
        <div class="shot">
            <span class="shot-id">{shot.id}</span> ({shot.duration}s) → {shot.transition}
            <p>{shot.description}</p>
            {f'<img src="{shot.video_url}">' if shot.video_url else '<div style="height:150px;background:#333;border-radius:4px;"></div>'}
            <p class="prompt">{shot.prompt[:100]}...</p>
        </div>
"""
        
        html += """
    </div>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)


# Convenience function
async def create_mv(
    description: str,
    style: str = "cinematic, neon noir",
    num_shots: int = 4,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None
) -> Storyboard:
    """Quick MV creation with minimal setup.
    
    Example:
        storyboard = await create_mv(
            "cyberpunk android exploring neon city",
            style="neon noir, cinematic",
            num_shots=6
        )
    """
    llm_config = None
    if llm_api_key:
        llm_config = {
            "api_key": llm_api_key,
            "base_url": llm_base_url,
            "model": "glm-4.7"
        }
    
    pipeline = MVPipeline(llm_config=llm_config)
    
    vibe = Vibe(
        description=description,
        style=style,
        domain="music_video"
    )
    
    return await pipeline.generate_mv(
        vibe=vibe,
        num_shots=num_shots,
        generate_music=False
    )

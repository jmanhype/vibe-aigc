"""
VLM Feedback System - Visual analysis for adaptive replanning.

Integrates Gemini VLM with the MetaPlanner to provide:
- Visual quality assessment of generated content
- Specific improvement suggestions
- Automatic prompt refinement
- Iterative quality improvement

This is the "eyes" of the AIGC system - it can SEE what it generates.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class FeedbackResult:
    """Result of VLM analysis."""
    quality_score: float  # 1-10
    media_type: MediaType
    description: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    prompt_improvements: List[str] = field(default_factory=list)
    parameter_changes: Dict[str, Any] = field(default_factory=dict)
    should_retry: bool = False
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "media_type": self.media_type.value,
            "description": self.description,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "prompt_improvements": self.prompt_improvements,
            "parameter_changes": self.parameter_changes,
            "should_retry": self.should_retry
        }


class VLMFeedback:
    """
    Visual Language Model feedback system.
    
    Uses Gemini to analyze generated content and provide
    actionable feedback for the MetaPlanner.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        quality_threshold: float = 7.0
    ):
        self.api_key = api_key or self._get_api_key()
        self.model_name = model
        self.quality_threshold = quality_threshold
        self.vlm = None
        
        if HAS_GENAI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.vlm = genai.GenerativeModel(self.model_name)
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or ComfyUI config."""
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
        
        # Try ComfyUI Gemini config
        config_path = Path("C:/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Gemini/config.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f).get("GEMINI_API_KEY")
            except:
                pass
        return None
    
    @property
    def available(self) -> bool:
        """Check if VLM is available."""
        return self.vlm is not None
    
    def analyze_image(self, image_path: Path, context: str = "") -> FeedbackResult:
        """Analyze an image and return feedback."""
        if not self.available or not HAS_PIL:
            return FeedbackResult(
                quality_score=5.0,
                media_type=MediaType.IMAGE,
                description="VLM not available",
                should_retry=False
            )
        
        img = Image.open(image_path)
        
        prompt = f"""You are an expert AI art director analyzing AI-generated images for quality.

Original prompt: {context}

IMPORTANT: You MUST provide specific, actionable prompt improvements.

Analyze this image and respond ONLY with valid JSON (no markdown):
{{
    "quality_score": <1-10 based on: composition, detail, prompt adherence, aesthetic quality>,
    "description": "<brief description of what you see>",
    "strengths": ["<specific strength 1>", "<specific strength 2>"],
    "weaknesses": ["<specific weakness 1>", "<specific weakness 2>"],
    "prompt_improvements": [
        "<SPECIFIC phrase to ADD to prompt to fix weakness 1>",
        "<SPECIFIC phrase to ADD to prompt to fix weakness 2>",
        "<SPECIFIC quality modifier to add>"
    ],
    "parameter_changes": {{
        "cfg": <suggest higher/lower cfg if needed, or null>,
        "steps": <suggest more/fewer steps if needed, or null>
    }}
}}

REQUIRED: prompt_improvements must have at least 2 specific suggestions like:
- "add sharp focus" if blurry
- "add dramatic shadows" if flat lighting
- "add intricate details" if lacking detail
- "add correct anatomy" if distorted

Score guide: 1-3 poor, 4-5 mediocre, 6-7 good, 8-9 excellent, 10 perfect."""

        try:
            response = self.vlm.generate_content([prompt, img])
            return self._parse_response(response.text, MediaType.IMAGE)
        except Exception as e:
            return FeedbackResult(
                quality_score=5.0,
                media_type=MediaType.IMAGE,
                description=f"Analysis failed: {e}",
                should_retry=False
            )
    
    def analyze_video(self, video_path: Path, context: str = "") -> FeedbackResult:
        """Analyze a video and return feedback."""
        if not self.available:
            return FeedbackResult(
                quality_score=5.0,
                media_type=MediaType.VIDEO,
                description="VLM not available",
                should_retry=False
            )
        
        try:
            # Upload video to Gemini
            video_file = genai.upload_file(str(video_path))
            
            # Wait for processing
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                return FeedbackResult(
                    quality_score=5.0,
                    media_type=MediaType.VIDEO,
                    description="Video processing failed",
                    should_retry=True
                )
            
            prompt = f"""You are an AI video director analyzing AI-generated video.

Context: {context}

Analyze this video and respond in JSON format:
{{
    "quality_score": <1-10>,
    "description": "<what happens in the video>",
    "motion_quality": "<smooth/jerky/static>",
    "temporal_consistency": "<consistent/flickering/morphing>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "prompt_improvements": ["<specific prompt addition>", ...],
    "parameter_changes": {{
        "steps": <suggested steps or null>,
        "cfg": <suggested cfg or null>,
        "motion_scale": <suggested motion scale or null>,
        "frame_count": <suggested frames or null>
    }}
}}

Focus on motion quality and temporal consistency."""

            response = self.vlm.generate_content([prompt, video_file])
            
            # Clean up
            try:
                genai.delete_file(video_file.name)
            except:
                pass
            
            return self._parse_response(response.text, MediaType.VIDEO)
            
        except Exception as e:
            return FeedbackResult(
                quality_score=5.0,
                media_type=MediaType.VIDEO,
                description=f"Analysis failed: {e}",
                should_retry=True
            )
    
    def analyze_media(self, media_path: Path, context: str = "") -> FeedbackResult:
        """Analyze any media file (auto-detect type)."""
        suffix = media_path.suffix.lower()
        
        if suffix in ['.mp4', '.webm', '.mov', '.avi']:
            return self.analyze_video(media_path, context)
        elif suffix in ['.webp', '.gif']:
            # Check if animated
            if HAS_PIL:
                try:
                    img = Image.open(media_path)
                    if getattr(img, 'n_frames', 1) > 1:
                        return self.analyze_video(media_path, context)
                except:
                    pass
            return self.analyze_image(media_path, context)
        else:
            return self.analyze_image(media_path, context)
    
    def _parse_response(self, text: str, media_type: MediaType) -> FeedbackResult:
        """Parse VLM response into FeedbackResult."""
        try:
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text.strip())
            
            quality = float(data.get("quality_score", 5.0))
            
            return FeedbackResult(
                quality_score=quality,
                media_type=media_type,
                description=data.get("description", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                prompt_improvements=data.get("prompt_improvements", []),
                parameter_changes=data.get("parameter_changes", {}),
                should_retry=quality < self.quality_threshold,
                raw_response=text
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return FeedbackResult(
                quality_score=5.0,
                media_type=media_type,
                description=text[:500] if text else "Parse failed",
                should_retry=True,
                raw_response=text
            )
    
    def suggest_improvements(self, feedback: FeedbackResult, current_prompt: str) -> str:
        """Generate an improved prompt based on feedback."""
        if not feedback.prompt_improvements:
            return current_prompt
        
        # Add top 3 improvements
        additions = ", ".join(feedback.prompt_improvements[:3])
        return f"{current_prompt}, {additions}"
    
    def should_retry(self, feedback: FeedbackResult) -> bool:
        """Determine if generation should be retried."""
        return feedback.quality_score < self.quality_threshold


def create_vlm_feedback(api_key: Optional[str] = None) -> VLMFeedback:
    """Factory function to create VLM feedback system."""
    return VLMFeedback(api_key=api_key)

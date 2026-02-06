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
from typing import Dict, List, Optional, Any, Set
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
    
    # Weakness patterns → prompt refinements mapping
    # Key: pattern to match in weakness text (lowercase)
    # Value: list of prompt additions to fix the issue
    WEAKNESS_REFINEMENTS: Dict[str, List[str]] = {
        # Lighting issues
        "lighting": ["dramatic lighting", "volumetric rays"],
        "flat light": ["dramatic lighting", "volumetric", "rim lighting"],
        "harsh light": ["soft diffused lighting", "golden hour"],
        "dark": ["well-lit", "bright ambient light"],
        "overexposed": ["balanced exposure", "natural lighting"],
        "underexposed": ["bright lighting", "fill light"],
        "shadow": ["deep shadows", "chiaroscuro", "dramatic shadows"],
        
        # Composition issues
        "centered": ["rule of thirds", "dynamic composition", "off-center subject"],
        "composition": ["balanced composition", "visual flow", "golden ratio"],
        "boring": ["dynamic angle", "interesting perspective"],
        "static": ["dynamic pose", "sense of motion", "action shot"],
        "framing": ["well-framed", "cinematic framing"],
        "cropped": ["full frame", "complete composition"],
        
        # Color issues
        "muddy color": ["vibrant colors", "high saturation", "color pop"],
        "dull color": ["rich saturated colors", "vivid tones"],
        "color": ["harmonious color palette", "color grading"],
        "saturation": ["balanced saturation", "rich colors"],
        "washed out": ["deep contrast", "saturated colors"],
        "oversaturated": ["natural color balance", "subtle tones"],
        
        # Detail/Sharpness issues
        "blurry": ["sharp focus", "crisp details", "8k uhd"],
        "blur": ["tack sharp", "high detail", "crystal clear"],
        "soft": ["sharp details", "crisp edges"],
        "lack detail": ["intricate details", "fine textures", "highly detailed"],
        "low detail": ["ultra detailed", "8k resolution", "intricate"],
        "noise": ["clean image", "noise-free", "pristine quality"],
        "grainy": ["smooth gradients", "clean render"],
        "artifact": ["clean render", "artifact-free", "pristine"],
        
        # Quality issues
        "quality": ["masterpiece", "best quality", "professional"],
        "amateur": ["professional quality", "expert craftsmanship"],
        "generic": ["unique style", "distinctive aesthetic"],
        
        # Anatomy/Form issues
        "anatomy": ["correct anatomy", "proper proportions"],
        "proportion": ["anatomically correct", "proper proportions"],
        "distort": ["undistorted", "proper form", "correct proportions"],
        "hand": ["detailed hands", "correct hand anatomy"],
        "finger": ["five fingers", "proper hand structure"],
        "face": ["detailed face", "expressive features"],
        "eye": ["detailed eyes", "proper eye anatomy"],
        
        # Style issues
        "style": ["consistent style", "cohesive aesthetic"],
        "inconsistent": ["unified style", "coherent design"],
        "realistic": ["photorealistic", "hyperrealistic", "lifelike"],
        
        # Texture issues
        "texture": ["rich textures", "tactile detail", "material definition"],
        "smooth": ["textured surfaces", "natural imperfections"],
        "plastic": ["natural materials", "organic textures"],
        
        # Depth/Dimension issues
        "flat": ["depth of field", "3D depth", "dimensional"],
        "depth": ["strong depth", "layered composition", "foreground-background separation"],
        "2d": ["volumetric", "three-dimensional", "sculptural"],
        
        # Motion (for video)
        "jerky": ["smooth motion", "fluid movement"],
        "flicker": ["temporal consistency", "stable frames"],
        "morph": ["consistent forms", "stable identity"],
    }
    
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
    
    def refine_prompt(self, original: str, feedback: FeedbackResult) -> str:
        """
        Intelligently refine prompt based on detected weaknesses.
        
        This is the SMART refinement method that:
        1. Parses weaknesses for actionable patterns
        2. Maps patterns to specific prompt additions
        3. Deduplicates and orders refinements
        4. Combines with VLM's direct suggestions
        
        Args:
            original: The original prompt
            feedback: VLM feedback with weaknesses
            
        Returns:
            Refined prompt with targeted improvements
        """
        refinements_set: set = set()
        original_lower = original.lower()
        
        # 1. Parse weaknesses and match to refinement patterns
        for weakness in feedback.weaknesses:
            weakness_lower = weakness.lower()
            
            for pattern, additions in self.WEAKNESS_REFINEMENTS.items():
                if pattern in weakness_lower:
                    for addition in additions:
                        # Don't add if already in original prompt
                        if addition.lower() not in original_lower:
                            refinements_set.add(addition)
        
        # 2. Add VLM's direct prompt_improvements (top 3, filtered)
        for improvement in feedback.prompt_improvements[:3]:
            improvement_clean = improvement.strip().lower()
            # Skip if it's just generic advice or already covered
            if len(improvement_clean) > 3 and improvement_clean not in original_lower:
                # Check if it's not already captured by our refinements
                already_covered = any(
                    ref.lower() in improvement_clean or improvement_clean in ref.lower()
                    for ref in refinements_set
                )
                if not already_covered:
                    refinements_set.add(improvement.strip())
        
        # 3. Limit total refinements to avoid prompt bloat
        refinements = list(refinements_set)[:6]
        
        if not refinements:
            return original
        
        # 4. Construct refined prompt
        refinement_str = ", ".join(refinements)
        
        # Check if original already ends with quality terms
        quality_endings = ["quality", "detailed", "resolution", "masterpiece"]
        original_stripped = original.rstrip(" ,.")
        
        # Smart insertion: put refinements before closing quality terms if present
        for ending in quality_endings:
            if original_stripped.lower().endswith(ending):
                # Find last comma and insert before quality terms
                last_comma = original.rfind(",")
                if last_comma > len(original) // 2:  # Only if comma is in latter half
                    return f"{original[:last_comma]}, {refinement_str}{original[last_comma:]}"
        
        return f"{original}, {refinement_str}"
    
    def suggest_improvements(self, feedback: FeedbackResult, current_prompt: str) -> str:
        """Generate an improved prompt based on feedback.
        
        This is the main entry point used by VibeBackend.
        Uses refine_prompt() for smart refinement.
        """
        # Use the smart refinement method
        return self.refine_prompt(current_prompt, feedback)
    
    def should_retry(self, feedback: FeedbackResult) -> bool:
        """Determine if generation should be retried."""
        return feedback.quality_score < self.quality_threshold
    
    def get_refinement_summary(self, original: str, feedback: FeedbackResult) -> Dict[str, Any]:
        """Get detailed breakdown of what refinements were applied and why.
        
        Useful for debugging and understanding the refinement process.
        """
        applied_refinements = []
        original_lower = original.lower()
        
        for weakness in feedback.weaknesses:
            weakness_lower = weakness.lower()
            matched_patterns = []
            
            for pattern, additions in self.WEAKNESS_REFINEMENTS.items():
                if pattern in weakness_lower:
                    for addition in additions:
                        if addition.lower() not in original_lower:
                            matched_patterns.append({
                                "pattern": pattern,
                                "addition": addition
                            })
            
            if matched_patterns:
                applied_refinements.append({
                    "weakness": weakness,
                    "refinements": matched_patterns
                })
        
        refined = self.refine_prompt(original, feedback)
        
        return {
            "original_prompt": original,
            "refined_prompt": refined,
            "quality_score": feedback.quality_score,
            "weaknesses_parsed": len(feedback.weaknesses),
            "refinements_applied": applied_refinements,
            "vlm_suggestions_used": feedback.prompt_improvements[:3]
        }


def create_vlm_feedback(api_key: Optional[str] = None) -> VLMFeedback:
    """Factory function to create VLM feedback system."""
    return VLMFeedback(api_key=api_key)

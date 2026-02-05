"""
Asset Bank for Consistency Management.

Based on Paper Section 4 (AutoMV):
"Director Agent to manage a shared Character Bank"

Provides:
- Character profiles with visual consistency
- Style guides (colors, fonts, mood)
- Generated artifact storage
- Reference tracking
"""

import os
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil


@dataclass
class Character:
    """Character profile for consistency across content."""
    
    id: str
    name: str
    description: str
    visual_description: str = ""
    personality: str = ""
    voice_description: str = ""
    
    # Visual references
    reference_images: List[str] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Generate prompt context for this character."""
        lines = [
            f"Character: {self.name}",
            f"Description: {self.description}"
        ]
        if self.visual_description:
            lines.append(f"Visual: {self.visual_description}")
        if self.personality:
            lines.append(f"Personality: {self.personality}")
        if self.color_palette:
            lines.append(f"Colors: {', '.join(self.color_palette)}")
        return "\n".join(lines)


@dataclass
class StyleGuide:
    """Style guide for visual/tonal consistency."""
    
    id: str
    name: str
    description: str
    
    # Visual style
    color_palette: List[str] = field(default_factory=list)
    typography: Dict[str, str] = field(default_factory=dict)
    mood: str = ""
    aesthetic: str = ""
    
    # References
    reference_images: List[str] = field(default_factory=list)
    mood_board: List[str] = field(default_factory=list)
    
    # Technical
    aspect_ratio: str = ""
    resolution: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Generate prompt context for this style."""
        lines = [
            f"Style: {self.name}",
            f"Description: {self.description}"
        ]
        if self.mood:
            lines.append(f"Mood: {self.mood}")
        if self.aesthetic:
            lines.append(f"Aesthetic: {self.aesthetic}")
        if self.color_palette:
            lines.append(f"Colors: {', '.join(self.color_palette)}")
        if self.aspect_ratio:
            lines.append(f"Aspect Ratio: {self.aspect_ratio}")
        return "\n".join(lines)


@dataclass
class Artifact:
    """Generated artifact (image, audio, video, text)."""
    
    id: str
    type: str  # image, audio, video, text
    name: str
    description: str
    
    # Location
    url: Optional[str] = None
    local_path: Optional[str] = None
    base64_data: Optional[str] = None
    
    # Relations
    source_node_id: Optional[str] = None
    related_character_id: Optional[str] = None
    style_guide_id: Optional[str] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    prompt_used: Optional[str] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssetBank:
    """
    Shared asset storage for workflow consistency.
    
    Manages:
    - Characters (for consistent portrayal)
    - Style guides (for visual consistency)
    - Generated artifacts (images, audio, etc.)
    """
    
    def __init__(self, storage_dir: str = ".vibe_assets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._characters: Dict[str, Character] = {}
        self._style_guides: Dict[str, StyleGuide] = {}
        self._artifacts: Dict[str, Artifact] = {}
        
        # Load existing assets
        self._load_from_disk()
    
    # ==================== Characters ====================
    
    def add_character(self, character: Character) -> str:
        """Add a character to the bank."""
        self._characters[character.id] = character
        self._save_to_disk()
        return character.id
    
    def get_character(self, character_id: str) -> Optional[Character]:
        """Get a character by ID."""
        return self._characters.get(character_id)
    
    def find_character(self, name: str) -> Optional[Character]:
        """Find a character by name."""
        for char in self._characters.values():
            if char.name.lower() == name.lower():
                return char
        return None
    
    def list_characters(self) -> List[Character]:
        """List all characters."""
        return list(self._characters.values())
    
    def update_character(self, character_id: str, updates: Dict[str, Any]) -> bool:
        """Update a character's attributes."""
        if character_id not in self._characters:
            return False
        
        char = self._characters[character_id]
        for key, value in updates.items():
            if hasattr(char, key):
                setattr(char, key, value)
        
        self._save_to_disk()
        return True
    
    def add_character_reference(self, character_id: str, image_url: str) -> bool:
        """Add a reference image to a character."""
        if character_id not in self._characters:
            return False
        
        self._characters[character_id].reference_images.append(image_url)
        self._save_to_disk()
        return True
    
    # ==================== Style Guides ====================
    
    def add_style_guide(self, style: StyleGuide) -> str:
        """Add a style guide."""
        self._style_guides[style.id] = style
        self._save_to_disk()
        return style.id
    
    def get_style_guide(self, style_id: str) -> Optional[StyleGuide]:
        """Get a style guide by ID."""
        return self._style_guides.get(style_id)
    
    def list_style_guides(self) -> List[StyleGuide]:
        """List all style guides."""
        return list(self._style_guides.values())
    
    # ==================== Artifacts ====================
    
    def store_artifact(self, artifact: Artifact) -> str:
        """Store a generated artifact."""
        self._artifacts[artifact.id] = artifact
        self._save_to_disk()
        return artifact.id
    
    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID."""
        return self._artifacts.get(artifact_id)
    
    def find_artifacts(
        self,
        type: Optional[str] = None,
        character_id: Optional[str] = None,
        node_id: Optional[str] = None
    ) -> List[Artifact]:
        """Find artifacts matching criteria."""
        results = []
        for artifact in self._artifacts.values():
            if type and artifact.type != type:
                continue
            if character_id and artifact.related_character_id != character_id:
                continue
            if node_id and artifact.source_node_id != node_id:
                continue
            results.append(artifact)
        return results
    
    def get_character_images(self, character_id: str) -> List[Artifact]:
        """Get all images for a character."""
        return self.find_artifacts(type="image", character_id=character_id)
    
    # ==================== Context Generation ====================
    
    def get_project_context(self) -> str:
        """Generate full project context for prompts."""
        lines = ["## Project Assets\n"]
        
        if self._characters:
            lines.append("### Characters:")
            for char in self._characters.values():
                lines.append(f"\n{char.to_prompt_context()}")
        
        if self._style_guides:
            lines.append("\n### Style Guides:")
            for style in self._style_guides.values():
                lines.append(f"\n{style.to_prompt_context()}")
        
        return "\n".join(lines)
    
    def get_character_context(self, character_id: str) -> str:
        """Generate context for a specific character."""
        char = self.get_character(character_id)
        if not char:
            return ""
        
        context = char.to_prompt_context()
        
        # Add reference images
        if char.reference_images:
            context += f"\nReference Images: {', '.join(char.reference_images[:3])}"
        
        return context
    
    # ==================== Persistence ====================
    
    def _save_to_disk(self) -> None:
        """Save all assets to disk."""
        data = {
            "characters": {k: asdict(v) for k, v in self._characters.items()},
            "style_guides": {k: asdict(v) for k, v in self._style_guides.items()},
            "artifacts": {k: asdict(v) for k, v in self._artifacts.items()}
        }
        
        manifest_path = self.storage_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_from_disk(self) -> None:
        """Load assets from disk."""
        manifest_path = self.storage_dir / "manifest.json"
        
        if not manifest_path.exists():
            return
        
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            
            for k, v in data.get("characters", {}).items():
                self._characters[k] = Character(**v)
            
            for k, v in data.get("style_guides", {}).items():
                self._style_guides[k] = StyleGuide(**v)
            
            for k, v in data.get("artifacts", {}).items():
                self._artifacts[k] = Artifact(**v)
        except Exception as e:
            print(f"Warning: Could not load asset bank: {e}")
    
    def clear(self) -> None:
        """Clear all assets."""
        self._characters.clear()
        self._style_guides.clear()
        self._artifacts.clear()
        self._save_to_disk()
    
    # ==================== Convenience Methods ====================
    
    def create_character(
        self,
        name: str,
        description: str,
        visual_description: str = "",
        personality: str = ""
    ) -> Character:
        """Create and add a character."""
        char_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        character = Character(
            id=char_id,
            name=name,
            description=description,
            visual_description=visual_description,
            personality=personality
        )
        self.add_character(character)
        return character
    
    def create_style_guide(
        self,
        name: str,
        description: str,
        mood: str = "",
        aesthetic: str = "",
        color_palette: Optional[List[str]] = None
    ) -> StyleGuide:
        """Create and add a style guide."""
        style_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        style = StyleGuide(
            id=style_id,
            name=name,
            description=description,
            mood=mood,
            aesthetic=aesthetic,
            color_palette=color_palette or []
        )
        self.add_style_guide(style)
        return style
    
    def create_artifact(
        self,
        type: str,
        name: str,
        description: str,
        url: Optional[str] = None,
        character_id: Optional[str] = None,
        node_id: Optional[str] = None,
        prompt_used: Optional[str] = None
    ) -> Artifact:
        """Create and store an artifact."""
        artifact_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        artifact = Artifact(
            id=artifact_id,
            type=type,
            name=name,
            description=description,
            url=url,
            related_character_id=character_id,
            source_node_id=node_id,
            prompt_used=prompt_used
        )
        self.store_artifact(artifact)
        return artifact


def create_asset_bank(storage_dir: str = ".vibe_assets") -> AssetBank:
    """Create an asset bank with default configuration."""
    return AssetBank(storage_dir)

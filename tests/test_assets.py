"""Tests for Asset Bank."""

import pytest
import tempfile
import shutil
from pathlib import Path

from vibe_aigc.assets import (
    AssetBank,
    Character,
    StyleGuide,
    Artifact,
    create_asset_bank
)


class TestCharacter:
    """Test Character class."""
    
    def test_create_character(self):
        char = Character(
            id="char-001",
            name="Alice",
            description="A curious adventurer",
            visual_description="Young woman, brown hair, blue eyes",
            personality="Curious, brave, kind"
        )
        
        assert char.name == "Alice"
        assert char.id == "char-001"
    
    def test_to_prompt_context(self):
        char = Character(
            id="char-001",
            name="Bob",
            description="A wise mentor",
            visual_description="Elderly man with white beard",
            color_palette=["#333", "#gold"]
        )
        
        context = char.to_prompt_context()
        
        assert "Bob" in context
        assert "wise mentor" in context
        assert "white beard" in context
        assert "#333" in context


class TestStyleGuide:
    """Test StyleGuide class."""
    
    def test_create_style_guide(self):
        style = StyleGuide(
            id="style-001",
            name="Cyberpunk",
            description="Neon-lit dystopian future",
            mood="dark, gritty",
            aesthetic="high-tech low-life",
            color_palette=["#ff00ff", "#00ffff", "#000"]
        )
        
        assert style.name == "Cyberpunk"
        assert style.mood == "dark, gritty"
    
    def test_to_prompt_context(self):
        style = StyleGuide(
            id="style-001",
            name="Minimalist",
            description="Clean and simple",
            mood="calm",
            aesthetic="modern",
            aspect_ratio="16:9"
        )
        
        context = style.to_prompt_context()
        
        assert "Minimalist" in context
        assert "calm" in context
        assert "16:9" in context


class TestArtifact:
    """Test Artifact class."""
    
    def test_create_artifact(self):
        artifact = Artifact(
            id="art-001",
            type="image",
            name="Hero Shot",
            description="Main character portrait",
            url="https://example.com/image.png"
        )
        
        assert artifact.type == "image"
        assert artifact.url is not None


class TestAssetBank:
    """Test AssetBank class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        shutil.rmtree(dir_path)
    
    def test_create_asset_bank(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        assert bank.storage_dir.exists()
    
    def test_add_and_get_character(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        char = Character(
            id="char-001",
            name="Test Character",
            description="A test"
        )
        bank.add_character(char)
        
        retrieved = bank.get_character("char-001")
        
        assert retrieved is not None
        assert retrieved.name == "Test Character"
    
    def test_find_character_by_name(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        char = Character(id="c1", name="Alice", description="Test")
        bank.add_character(char)
        
        found = bank.find_character("alice")  # Case insensitive
        
        assert found is not None
        assert found.id == "c1"
    
    def test_list_characters(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        bank.add_character(Character(id="c1", name="A", description=""))
        bank.add_character(Character(id="c2", name="B", description=""))
        
        chars = bank.list_characters()
        
        assert len(chars) == 2
    
    def test_update_character(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        char = Character(id="c1", name="Old Name", description="")
        bank.add_character(char)
        
        bank.update_character("c1", {"name": "New Name"})
        
        updated = bank.get_character("c1")
        assert updated.name == "New Name"
    
    def test_add_character_reference(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        char = Character(id="c1", name="Test", description="")
        bank.add_character(char)
        
        bank.add_character_reference("c1", "https://example.com/ref.png")
        
        updated = bank.get_character("c1")
        assert len(updated.reference_images) == 1
    
    def test_add_and_get_style_guide(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        style = StyleGuide(id="s1", name="Test Style", description="")
        bank.add_style_guide(style)
        
        retrieved = bank.get_style_guide("s1")
        
        assert retrieved is not None
        assert retrieved.name == "Test Style"
    
    def test_store_and_get_artifact(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        artifact = Artifact(
            id="a1",
            type="image",
            name="Test Image",
            description="",
            url="https://example.com/img.png"
        )
        bank.store_artifact(artifact)
        
        retrieved = bank.get_artifact("a1")
        
        assert retrieved is not None
        assert retrieved.type == "image"
    
    def test_find_artifacts(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        bank.store_artifact(Artifact(id="a1", type="image", name="Img1", description="", related_character_id="c1"))
        bank.store_artifact(Artifact(id="a2", type="image", name="Img2", description="", related_character_id="c1"))
        bank.store_artifact(Artifact(id="a3", type="audio", name="Audio1", description=""))
        
        images = bank.find_artifacts(type="image")
        char_images = bank.find_artifacts(character_id="c1")
        
        assert len(images) == 2
        assert len(char_images) == 2
    
    def test_get_project_context(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        bank.add_character(Character(id="c1", name="Hero", description="Main character"))
        bank.add_style_guide(StyleGuide(id="s1", name="Dark", description="Dark aesthetic"))
        
        context = bank.get_project_context()
        
        assert "Hero" in context
        assert "Dark" in context
    
    def test_persistence(self, temp_dir):
        # Create and populate bank
        bank1 = AssetBank(storage_dir=temp_dir)
        bank1.add_character(Character(id="c1", name="Persistent", description=""))
        
        # Create new bank from same directory
        bank2 = AssetBank(storage_dir=temp_dir)
        
        # Should have loaded the character
        assert bank2.get_character("c1") is not None
        assert bank2.get_character("c1").name == "Persistent"
    
    def test_clear(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        bank.add_character(Character(id="c1", name="Test", description=""))
        bank.clear()
        
        assert len(bank.list_characters()) == 0
    
    def test_create_character_convenience(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        char = bank.create_character(
            name="Quick Character",
            description="Created quickly",
            visual_description="Generic look"
        )
        
        assert char.id is not None
        assert char.name == "Quick Character"
        assert bank.get_character(char.id) is not None
    
    def test_create_style_guide_convenience(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        style = bank.create_style_guide(
            name="Quick Style",
            description="Created quickly",
            mood="happy",
            color_palette=["#fff", "#000"]
        )
        
        assert style.id is not None
        assert style.mood == "happy"
    
    def test_create_artifact_convenience(self, temp_dir):
        bank = AssetBank(storage_dir=temp_dir)
        
        artifact = bank.create_artifact(
            type="image",
            name="Generated Image",
            description="AI generated",
            url="https://example.com/gen.png",
            prompt_used="A beautiful sunset"
        )
        
        assert artifact.id is not None
        assert artifact.prompt_used == "A beautiful sunset"


class TestCreateAssetBank:
    """Test create_asset_bank factory."""
    
    def test_creates_bank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bank = create_asset_bank(storage_dir=temp_dir)
            
            assert isinstance(bank, AssetBank)

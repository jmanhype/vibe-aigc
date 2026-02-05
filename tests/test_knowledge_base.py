"""Tests for Domain-Specific Expert Knowledge Base (Paper Section 5.3)."""

import pytest
from vibe_aigc.knowledge import (
    KnowledgeBase,
    DomainKnowledge,
    create_knowledge_base
)


class TestKnowledgeBase:
    """Test the knowledge base functionality."""
    
    def test_create_knowledge_base(self):
        """Test that default knowledge base is created with built-in domains."""
        kb = create_knowledge_base()
        
        domains = kb.list_domains()
        assert "film" in domains
        assert "writing" in domains
        assert "design" in domains
        assert "music" in domains
    
    def test_query_hitchcockian_suspense(self):
        """Test the Hitchcock example from the paper."""
        kb = create_knowledge_base()
        
        result = kb.query("Hitchcockian suspense thriller")
        
        # Should match the hitchcockian suspense concept
        assert len(result["matched_concepts"]) >= 1
        concept = result["matched_concepts"][0]
        assert concept["concept"] == "hitchcockian suspense"
        assert concept["domain"] == "film"
        
        # Should have technical specs
        assert "camera" in result["technical_specs"]
        assert "lighting" in result["technical_specs"]
        assert "dolly zoom" in result["technical_specs"]["camera"]
    
    def test_query_multiple_concepts(self):
        """Test querying multiple concepts at once."""
        kb = create_knowledge_base()
        
        result = kb.query("noir cinematic video")
        
        # Should match both noir and cinematic
        concepts = [c["concept"] for c in result["matched_concepts"]]
        assert "noir" in concepts
        assert "cinematic" in concepts
    
    def test_query_no_match(self):
        """Test querying with no matching concepts."""
        kb = create_knowledge_base()
        
        result = kb.query("random nonexistent style")
        
        assert len(result["matched_concepts"]) == 0
        assert len(result["technical_specs"]) == 0
    
    def test_query_domain_filter(self):
        """Test filtering query by domain."""
        kb = create_knowledge_base()
        
        # Query for cinematic but only in film domain
        result = kb.query("cinematic", domain="film")
        
        assert len(result["matched_concepts"]) >= 1
        assert all(c["domain"] == "film" for c in result["matched_concepts"])
    
    def test_to_prompt_context(self):
        """Test generating prompt context."""
        kb = create_knowledge_base()
        
        context = kb.to_prompt_context("Create a noir cinematic video")
        
        # Should contain domain knowledge
        assert "Domain Knowledge Context" in context
        assert "Matched Creative Concepts" in context
        assert "Technical Specifications" in context
        
        # Should include specific details
        assert "noir" in context.lower()
        assert "cinematic" in context.lower()
    
    def test_custom_domain_knowledge(self):
        """Test adding custom domain knowledge."""
        kb = create_knowledge_base()
        
        # Create custom domain
        custom = DomainKnowledge(domain="custom")
        custom.add_concept(
            name="my style",
            description="A custom creative style",
            technical_specs={"color": ["red", "blue"]},
            examples=["Example A", "Example B"]
        )
        
        kb.register_domain(custom)
        
        # Should be queryable
        result = kb.query("my style")
        assert len(result["matched_concepts"]) >= 1
        assert result["matched_concepts"][0]["domain"] == "custom"
    
    def test_list_concepts(self):
        """Test listing all concepts."""
        kb = create_knowledge_base()
        
        concepts = kb.list_concepts()
        
        assert len(concepts) > 0
        # Format should be domain:concept
        assert all(":" in c for c in concepts)
        assert "film:hitchcockian suspense" in concepts
    
    def test_get_domain(self):
        """Test getting a specific domain."""
        kb = create_knowledge_base()
        
        film = kb.get_domain("film")
        
        assert film is not None
        assert film.domain == "film"
        assert "hitchcockian suspense" in film.concepts


class TestDomainKnowledge:
    """Test DomainKnowledge class."""
    
    def test_add_concept(self):
        """Test adding a concept."""
        dk = DomainKnowledge(domain="test")
        
        dk.add_concept(
            name="Test Concept",
            description="A test concept",
            technical_specs={"key": ["value1", "value2"]},
            examples=["Ex1"]
        )
        
        assert "test concept" in dk.concepts
        assert dk.concepts["test concept"]["description"] == "A test concept"
    
    def test_add_technique(self):
        """Test adding a technique."""
        dk = DomainKnowledge(domain="test")
        
        dk.add_technique("my technique", ["Step 1", "Step 2", "Step 3"])
        
        assert "my technique" in dk.techniques
        assert len(dk.techniques["my technique"]) == 3
    
    def test_add_constraint(self):
        """Test adding constraints."""
        dk = DomainKnowledge(domain="test")
        
        dk.add_constraint("professional", ["Rule 1", "Rule 2"])
        
        assert "professional" in dk.constraints
        assert len(dk.constraints["professional"]) == 2

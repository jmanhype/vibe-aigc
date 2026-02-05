"""
Domain-Specific Expert Knowledge Base.

As defined in Paper Section 5.1 and 5.3:
- Stores professional skills, experiential knowledge
- Maps abstract creative intent to concrete technical constraints
- Helps MetaPlanner interpret "vibes" correctly
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class DomainKnowledge:
    """Knowledge entry for a specific creative domain."""
    
    domain: str
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    techniques: Dict[str, List[str]] = field(default_factory=dict)
    constraints: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_concept(
        self,
        name: str,
        description: str,
        technical_specs: Dict[str, Any],
        examples: Optional[List[str]] = None
    ) -> None:
        """Add a creative concept with its technical mapping."""
        self.concepts[name.lower()] = {
            "description": description,
            "technical_specs": technical_specs,
            "examples": examples or []
        }
    
    def add_technique(self, name: str, steps: List[str]) -> None:
        """Add a technique with implementation steps."""
        self.techniques[name.lower()] = steps
    
    def add_constraint(self, context: str, rules: List[str]) -> None:
        """Add constraints for a specific context."""
        self.constraints[context.lower()] = rules


class KnowledgeBase:
    """
    Domain-Specific Expert Knowledge Base.
    
    From Paper Section 5.3:
    "The Planner begins by querying the creative expert knowledge modules,
    which encapsulate a vast array of multi-disciplinary expertise."
    """
    
    def __init__(self):
        self._domains: Dict[str, DomainKnowledge] = {}
        self._load_builtin_knowledge()
    
    def _load_builtin_knowledge(self) -> None:
        """Load built-in knowledge for common creative domains."""
        self._load_film_knowledge()
        self._load_writing_knowledge()
        self._load_design_knowledge()
        self._load_music_knowledge()
        self._load_visual_generation_knowledge()
    
    def _load_film_knowledge(self) -> None:
        """Film/video production knowledge."""
        film = DomainKnowledge(domain="film")
        
        # Cinematic styles (from paper's Hitchcock example)
        film.add_concept(
            name="hitchcockian suspense",
            description="Suspense technique pioneered by Alfred Hitchcock",
            technical_specs={
                "camera": ["dolly zoom", "close-ups", "point-of-view shots"],
                "lighting": ["high-contrast", "low-key", "dramatic shadows"],
                "editing": ["slow build", "information asymmetry", "delayed reveal"],
                "audio": ["dissonant intervals", "silence before shock", "diegetic tension"]
            },
            examples=["Vertigo", "Psycho", "Rear Window"]
        )
        
        film.add_concept(
            name="cinematic",
            description="Film-quality visual aesthetic",
            technical_specs={
                "aspect_ratio": "2.39:1 or 1.85:1",
                "frame_rate": "24fps",
                "color_grading": ["filmic LUT", "crushed blacks", "highlight rolloff"],
                "depth_of_field": "shallow, subject isolation"
            }
        )
        
        film.add_concept(
            name="documentary",
            description="Non-fiction film style emphasizing authenticity",
            technical_specs={
                "camera": ["handheld", "natural movement", "observational"],
                "lighting": "natural/available light",
                "audio": ["ambient sound", "interview format"],
                "editing": ["talking heads", "b-roll cutaways"]
            }
        )
        
        film.add_concept(
            name="noir",
            description="Dark, moody visual style from 1940s crime films",
            technical_specs={
                "lighting": ["low-key", "venetian blind shadows", "single source"],
                "color": ["desaturated", "high contrast", "black and white optional"],
                "composition": ["dutch angles", "deep shadows", "silhouettes"],
                "mood": ["cynical", "fatalistic", "morally ambiguous"]
            }
        )
        
        # Techniques
        film.add_technique("three-point lighting", [
            "Position key light at 45° to subject",
            "Add fill light opposite key at lower intensity",
            "Place back light behind subject for separation"
        ])
        
        film.add_technique("shot-reverse-shot", [
            "Establish two subjects in conversation",
            "Cut to over-the-shoulder of subject A looking at B",
            "Cut to over-the-shoulder of subject B looking at A",
            "Maintain 180° rule throughout"
        ])
        
        self._domains["film"] = film
    
    def _load_writing_knowledge(self) -> None:
        """Writing and content creation knowledge."""
        writing = DomainKnowledge(domain="writing")
        
        writing.add_concept(
            name="viral",
            description="Content optimized for social sharing",
            technical_specs={
                "hook": "First line must create curiosity gap",
                "length": "Platform-specific (Twitter: concise, LinkedIn: medium)",
                "structure": ["hook", "value", "call-to-action"],
                "emotion": ["surprise", "awe", "humor", "outrage"]
            }
        )
        
        writing.add_concept(
            name="technical",
            description="Documentation and technical writing",
            technical_specs={
                "structure": ["overview", "prerequisites", "steps", "examples", "troubleshooting"],
                "voice": "second person, active voice",
                "formatting": ["code blocks", "numbered lists", "clear headings"],
                "assumptions": "state explicitly"
            }
        )
        
        writing.add_concept(
            name="storytelling",
            description="Narrative-driven content",
            technical_specs={
                "structure": ["hook", "context", "conflict", "resolution", "lesson"],
                "elements": ["protagonist", "stakes", "transformation"],
                "pacing": "vary sentence length for rhythm"
            }
        )
        
        writing.add_concept(
            name="persuasive",
            description="Content designed to convince",
            technical_specs={
                "structure": ["problem", "agitation", "solution"],
                "techniques": ["social proof", "scarcity", "authority", "reciprocity"],
                "objections": "address proactively"
            }
        )
        
        self._domains["writing"] = writing
    
    def _load_design_knowledge(self) -> None:
        """Graphic design and visual knowledge."""
        design = DomainKnowledge(domain="design")
        
        design.add_concept(
            name="minimalist",
            description="Clean, simple aesthetic with maximum whitespace",
            technical_specs={
                "whitespace": "generous, 40%+ of canvas",
                "colors": "limited palette, 2-3 colors max",
                "typography": "single font family, clear hierarchy",
                "elements": "only essential, remove until it breaks"
            }
        )
        
        design.add_concept(
            name="bold",
            description="High-impact, attention-grabbing design",
            technical_specs={
                "colors": ["high saturation", "high contrast", "complementary pairs"],
                "typography": ["large headlines", "heavy weights", "tight tracking"],
                "composition": ["asymmetric", "overlapping elements", "breaking grid"]
            }
        )
        
        design.add_concept(
            name="professional",
            description="Corporate, trustworthy aesthetic",
            technical_specs={
                "colors": ["blues", "grays", "conservative palette"],
                "typography": ["sans-serif", "readable", "consistent"],
                "imagery": ["authentic photos", "no clip art", "diverse representation"],
                "layout": ["grid-based", "aligned", "balanced"]
            }
        )
        
        # Layout techniques
        design.add_technique("rule of thirds", [
            "Divide canvas into 3x3 grid",
            "Place key elements along gridlines",
            "Position focal point at intersection"
        ])
        
        design.add_technique("visual hierarchy", [
            "Determine information priority",
            "Use size: larger = more important",
            "Use contrast: high contrast = attention",
            "Use position: top-left reads first (LTR languages)"
        ])
        
        self._domains["design"] = design
    
    def _load_music_knowledge(self) -> None:
        """Music and audio production knowledge."""
        music = DomainKnowledge(domain="music")
        
        music.add_concept(
            name="upbeat",
            description="Energetic, positive musical mood",
            technical_specs={
                "tempo": "120-140 BPM",
                "key": "major key",
                "rhythm": ["syncopation", "driving beat"],
                "instrumentation": ["bright tones", "prominent percussion"]
            }
        )
        
        music.add_concept(
            name="melancholic",
            description="Sad, reflective musical mood",
            technical_specs={
                "tempo": "60-80 BPM",
                "key": "minor key",
                "harmony": ["suspended chords", "7ths", "descending progressions"],
                "instrumentation": ["piano", "strings", "sparse arrangement"]
            }
        )
        
        music.add_concept(
            name="epic",
            description="Grand, cinematic musical style",
            technical_specs={
                "dynamics": "quiet build to loud climax",
                "instrumentation": ["full orchestra", "choir", "percussion hits"],
                "structure": ["slow intro", "building layers", "climactic peak", "resolution"],
                "techniques": ["ostinato", "layering", "brass stabs"]
            }
        )
        
        self._domains["music"] = music
    
    def _load_visual_generation_knowledge(self) -> None:
        """Visual/image generation knowledge for Stable Diffusion & ComfyUI."""
        visual = DomainKnowledge(domain="visual")
        
        # === AESTHETIC STYLES ===
        visual.add_concept(
            name="cyberpunk",
            description="Futuristic dystopian aesthetic with neon and tech",
            technical_specs={
                "sd_prompt_tags": ["cyberpunk", "neon lights", "futuristic city", "rain", 
                                   "holographic displays", "chrome", "dystopian"],
                "color_palette": ["#FF00FF", "#00FFFF", "#FF0080", "#00FF80", "#1a1a2e"],
                "lighting": ["neon glow", "rim lighting", "volumetric fog", "light rays"],
                "composition": ["low angle", "wide shot", "reflective surfaces"],
                "negative_prompt": ["natural lighting", "daytime", "pastoral", "bright colors"]
            },
            examples=["Blade Runner", "Ghost in the Shell", "Akira"]
        )
        
        visual.add_concept(
            name="neon noir",
            description="Modern noir with neon lighting and rain-slicked streets",
            technical_specs={
                "sd_prompt_tags": ["neon noir", "rain", "night city", "wet streets",
                                   "reflections", "moody", "cinematic lighting"],
                "color_palette": ["#FF00FF", "#00FFFF", "#1a1a2e", "#2d132c"],
                "lighting": ["neon signs", "puddle reflections", "dramatic shadows", "rim light"],
                "composition": ["dutch angle", "silhouettes", "deep shadows"],
                "negative_prompt": ["bright", "cheerful", "daytime", "sunny"]
            }
        )
        
        visual.add_concept(
            name="anime",
            description="Japanese animation style",
            technical_specs={
                "sd_prompt_tags": ["anime", "cel shading", "vibrant colors", 
                                   "detailed eyes", "dramatic pose"],
                "models": ["anything-v5", "counterfeit-v3", "waifu-diffusion"],
                "cfg_scale": 7,
                "negative_prompt": ["realistic", "photographic", "3d render", "western"]
            }
        )
        
        visual.add_concept(
            name="photorealistic",
            description="Realistic, photo-quality images",
            technical_specs={
                "sd_prompt_tags": ["photorealistic", "hyperrealistic", "8k", "uhd",
                                   "RAW photo", "dslr", "soft lighting", "high detail"],
                "models": ["realistic-vision", "deliberate", "photon"],
                "cfg_scale": 5,
                "steps": 30,
                "negative_prompt": ["cartoon", "anime", "painting", "illustration", 
                                   "drawing", "artificial", "fake"]
            }
        )
        
        visual.add_concept(
            name="cinematic",
            description="Movie-quality visuals with dramatic lighting",
            technical_specs={
                "sd_prompt_tags": ["cinematic", "dramatic lighting", "film grain",
                                   "anamorphic", "depth of field", "bokeh", "lens flare"],
                "aspect_ratio": "21:9 or 16:9",
                "lighting": ["three-point lighting", "golden hour", "blue hour", "volumetric"],
                "composition": ["rule of thirds", "leading lines", "depth layers"]
            }
        )
        
        visual.add_concept(
            name="concept art",
            description="Professional concept art style for games/films",
            technical_specs={
                "sd_prompt_tags": ["concept art", "digital painting", "artstation",
                                   "trending on artstation", "matte painting", "detailed"],
                "artists": ["greg rutkowski", "craig mullins", "feng zhu"],
                "negative_prompt": ["photo", "realistic", "amateur"]
            }
        )
        
        visual.add_concept(
            name="portrait",
            description="Character portrait focus",
            technical_specs={
                "sd_prompt_tags": ["portrait", "headshot", "face focus", "detailed face",
                                   "sharp focus", "studio lighting"],
                "composition": ["centered", "eye level", "close-up"],
                "lighting": ["rembrandt lighting", "butterfly lighting", "split lighting"],
                "negative_prompt": ["full body", "wide shot", "multiple people", "crowd"]
            }
        )
        
        # === QUALITY MODIFIERS ===
        visual.add_concept(
            name="high quality",
            description="Quality boosting prompt modifiers",
            technical_specs={
                "sd_prompt_tags": ["masterpiece", "best quality", "highly detailed", 
                                   "sharp focus", "intricate details", "professional"],
                "negative_prompt": ["worst quality", "low quality", "blurry", "jpeg artifacts",
                                   "watermark", "signature", "text", "error", "cropped"]
            }
        )
        
        # === LIGHTING TECHNIQUES ===
        visual.add_technique("dramatic lighting", [
            "Add 'dramatic lighting' or 'cinematic lighting' to prompt",
            "Include specific light source (neon, golden hour, etc.)",
            "Add 'volumetric lighting' or 'god rays' for atmosphere",
            "Use high contrast in composition"
        ])
        
        visual.add_technique("neon lighting", [
            "Specify neon colors: 'pink neon', 'cyan neon', 'purple neon'",
            "Add 'neon glow', 'neon signs', 'neon reflections'",
            "Include 'wet streets' or 'rain' for reflections",
            "Set scene at night for contrast"
        ])
        
        # === COMFYUI WORKFLOW KNOWLEDGE ===
        visual.add_concept(
            name="txt2img",
            description="Text to image generation workflow",
            technical_specs={
                "nodes": ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", 
                         "VAEDecode", "SaveImage"],
                "optimal_settings": {
                    "steps": "20-30 for quality, 10-15 for speed",
                    "cfg": "7-8 for balanced, 5-6 for creative, 10+ for strict",
                    "sampler": "euler_ancestral for variety, dpm++ for quality"
                }
            }
        )
        
        visual.add_concept(
            name="img2img",
            description="Image to image transformation workflow",
            technical_specs={
                "nodes": ["LoadImage", "VAEEncode", "KSampler", "VAEDecode", "SaveImage"],
                "optimal_settings": {
                    "denoise": "0.4-0.6 for subtle changes, 0.7-0.9 for major changes",
                    "steps": "20-30",
                    "cfg": "7-10"
                }
            }
        )
        
        visual.add_concept(
            name="character consistency",
            description="Maintaining character appearance across generations",
            technical_specs={
                "techniques": ["IPAdapter", "reference image", "trained LoRA"],
                "nodes": ["IPAdapter", "IPAdapterFaceID", "ControlNet"],
                "tips": ["Use same seed for similar poses", "Train character LoRA for best results"]
            }
        )
        
        # === CONSTRAINTS ===
        visual.add_constraint("8gb vram", [
            "Use SD 1.5 or SDXL with optimizations",
            "Enable --lowvram or --medvram flags",
            "Max resolution ~768x768 for SD1.5, ~1024x1024 for SDXL",
            "Use fp8 for Flux models",
            "Avoid large batch sizes"
        ])
        
        visual.add_constraint("video generation", [
            "AnimateDiff for short clips (16-32 frames)",
            "CogVideoX for longer content (requires more VRAM)",
            "Keep consistent seed and prompt for coherence",
            "Use motion LoRA for specific movements"
        ])
        
        self._domains["visual"] = visual
    
    def register_domain(self, knowledge: DomainKnowledge) -> None:
        """Register a custom domain knowledge module."""
        self._domains[knowledge.domain.lower()] = knowledge
    
    def query(
        self,
        intent: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge base for relevant expertise.
        
        Args:
            intent: The creative intent or vibe description
            domain: Optional domain to search (searches all if None)
        
        Returns:
            Dictionary of relevant knowledge
        """
        results = {
            "matched_concepts": [],
            "relevant_techniques": [],
            "constraints": [],
            "technical_specs": {}
        }
        
        intent_lower = intent.lower()
        domains_to_search = [self._domains[domain.lower()]] if domain else self._domains.values()
        
        for dk in domains_to_search:
            # Search concepts
            for concept_name, concept_data in dk.concepts.items():
                if concept_name in intent_lower or any(
                    word in intent_lower for word in concept_name.split()
                ):
                    results["matched_concepts"].append({
                        "domain": dk.domain,
                        "concept": concept_name,
                        **concept_data
                    })
                    # Merge technical specs
                    for key, value in concept_data.get("technical_specs", {}).items():
                        if key not in results["technical_specs"]:
                            results["technical_specs"][key] = value
                        elif isinstance(value, list):
                            existing = results["technical_specs"][key]
                            if isinstance(existing, list):
                                results["technical_specs"][key] = list(set(existing + value))
            
            # Search techniques
            for tech_name, steps in dk.techniques.items():
                if tech_name in intent_lower:
                    results["relevant_techniques"].append({
                        "domain": dk.domain,
                        "technique": tech_name,
                        "steps": steps
                    })
            
            # Include constraints
            for context, rules in dk.constraints.items():
                if context in intent_lower:
                    results["constraints"].extend(rules)
        
        return results
    
    def get_domain(self, domain: str) -> Optional[DomainKnowledge]:
        """Get a specific domain's knowledge."""
        return self._domains.get(domain.lower())
    
    def list_domains(self) -> List[str]:
        """List all available domains."""
        return list(self._domains.keys())
    
    def list_concepts(self, domain: Optional[str] = None) -> List[str]:
        """List all concepts, optionally filtered by domain."""
        concepts = []
        domains = [self._domains[domain.lower()]] if domain else self._domains.values()
        for dk in domains:
            concepts.extend([f"{dk.domain}:{c}" for c in dk.concepts.keys()])
        return concepts
    
    def to_prompt_context(self, intent: str) -> str:
        """
        Generate prompt context from knowledge base query.
        
        Used by MetaPlanner to enhance intent understanding.
        """
        knowledge = self.query(intent)
        
        if not knowledge["matched_concepts"] and not knowledge["technical_specs"]:
            return ""
        
        lines = ["## Domain Knowledge Context\n"]
        
        if knowledge["matched_concepts"]:
            lines.append("### Matched Creative Concepts:")
            for concept in knowledge["matched_concepts"]:
                lines.append(f"\n**{concept['concept'].title()}** ({concept['domain']})")
                lines.append(f"  {concept.get('description', '')}")
                if concept.get("examples"):
                    lines.append(f"  Examples: {', '.join(concept['examples'])}")
        
        if knowledge["technical_specs"]:
            lines.append("\n### Technical Specifications:")
            for key, value in knowledge["technical_specs"].items():
                if isinstance(value, list):
                    lines.append(f"- **{key}**: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"- **{key}**: {value}")
        
        if knowledge["relevant_techniques"]:
            lines.append("\n### Relevant Techniques:")
            for tech in knowledge["relevant_techniques"]:
                lines.append(f"- **{tech['technique']}**: {' → '.join(tech['steps'])}")
        
        if knowledge["constraints"]:
            lines.append("\n### Constraints:")
            for constraint in knowledge["constraints"]:
                lines.append(f"- {constraint}")
        
        return "\n".join(lines)


# Convenience function
def create_knowledge_base() -> KnowledgeBase:
    """Create a knowledge base with built-in domain knowledge."""
    return KnowledgeBase()

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
        self._load_animation_knowledge()
        self._load_audio_knowledge()
        self._load_character_knowledge()
        self._load_environment_knowledge()
    
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
    
    def _load_animation_knowledge(self) -> None:
        """Animation and motion knowledge."""
        anim = DomainKnowledge(domain="animation")
        
        anim.add_concept(
            name="sakuga",
            description="High-quality Japanese animation with fluid motion",
            technical_specs={
                "frame_rate": "24fps with variable timing (1s, 2s, 3s)",
                "key_poses": ["anticipation", "follow-through", "squash and stretch"],
                "effects": ["impact frames", "smear frames", "speed lines"],
                "artists": ["Yutaka Nakamura", "Norio Matsumoto", "Hiroyuki Imaishi"]
            },
            examples=["One Punch Man", "Mob Psycho 100", "Demon Slayer"]
        )
        
        anim.add_concept(
            name="disney style",
            description="Classic Western animation principles",
            technical_specs={
                "principles": ["squash and stretch", "anticipation", "staging", 
                              "follow through", "slow in/out", "arcs", "secondary action",
                              "timing", "exaggeration", "solid drawing", "appeal"],
                "frame_rate": "24fps on ones or twos",
                "characteristics": ["smooth motion", "expressive faces", "musical timing"]
            },
            examples=["The Little Mermaid", "Beauty and the Beast", "Aladdin"]
        )
        
        anim.add_concept(
            name="rotoscope",
            description="Animation traced from live-action footage",
            technical_specs={
                "workflow": ["film reference footage", "trace key frames", "interpolate"],
                "use_cases": ["realistic motion", "dance sequences", "complex action"],
                "tools": ["reference video", "onion skinning"]
            },
            examples=["A Scanner Darkly", "Waking Life", "Undone"]
        )
        
        anim.add_concept(
            name="limited animation",
            description="Cost-effective animation with fewer frames",
            technical_specs={
                "techniques": ["held frames", "mouth flaps only", "sliding cels", 
                              "repeated cycles", "still backgrounds"],
                "frame_rate": "8-12 fps effective",
                "style": "emphasizes design over motion"
            },
            examples=["South Park", "Archer", "early Hanna-Barbera"]
        )
        
        anim.add_concept(
            name="smooth animation",
            description="Fluid, high frame-rate animation",
            technical_specs={
                "frame_rate": "24-60fps on ones",
                "interpolation": "ease in/out on all movements",
                "motion_blur": "natural blur on fast movements",
                "animatediff_settings": {"frames": 16, "fps": 8, "motion_scale": 1.0}
            }
        )
        
        # Techniques
        anim.add_technique("walk cycle", [
            "Contact pose: heel strikes ground",
            "Down pose: body at lowest point",
            "Passing pose: legs cross",
            "Up pose: body at highest point",
            "Loop seamlessly (8-16 frames typical)"
        ])
        
        anim.add_technique("anticipation", [
            "Move opposite to intended action direction",
            "Compress/wind up before release",
            "Holds for 2-4 frames before main action",
            "Bigger anticipation = more powerful action"
        ])
        
        self._domains["animation"] = anim
    
    def _load_audio_knowledge(self) -> None:
        """Audio and sound design knowledge."""
        audio = DomainKnowledge(domain="audio")
        
        audio.add_concept(
            name="ambient",
            description="Background environmental soundscapes",
            technical_specs={
                "elements": ["room tone", "environmental sounds", "distant activity"],
                "mixing": "low in mix, constant, non-distracting",
                "layers": ["base drone", "mid-frequency detail", "occasional accents"],
                "music_gen_prompt": "ambient, atmospheric, ethereal, minimal, spacious"
            },
            examples=["Brian Eno", "Aphex Twin Selected Ambient Works"]
        )
        
        audio.add_concept(
            name="lo-fi",
            description="Low-fidelity aesthetic with warmth and imperfection",
            technical_specs={
                "characteristics": ["vinyl crackle", "tape hiss", "bit reduction", 
                                   "low-pass filter", "subtle pitch wobble"],
                "tempo": "70-90 BPM",
                "elements": ["jazz chords", "hip-hop beats", "ambient pads"],
                "music_gen_prompt": "lo-fi hip hop, chill, relaxing, vinyl crackle, jazz piano"
            },
            examples=["ChilledCow", "Nujabes", "J Dilla"]
        )
        
        audio.add_concept(
            name="horror",
            description="Unsettling, tension-building audio",
            technical_specs={
                "techniques": ["dissonance", "sudden silence", "infrasound", 
                              "reversed audio", "distorted voices"],
                "elements": ["drones", "metallic scrapes", "heartbeat", "breathing"],
                "music_gen_prompt": "dark ambient, horror, unsettling, dissonant, tension"
            },
            examples=["Akira Yamaoka", "Goblin", "John Carpenter"]
        )
        
        audio.add_concept(
            name="epic orchestral",
            description="Grand cinematic orchestral sound",
            technical_specs={
                "instruments": ["full orchestra", "choir", "taiko drums", "brass fanfares"],
                "dynamics": "piano to fortissimo swells",
                "structure": ["quiet intro", "building tension", "climactic peak"],
                "music_gen_prompt": "epic orchestral, cinematic, heroic, brass, choir, drums"
            },
            examples=["Hans Zimmer", "Two Steps From Hell", "Audiomachine"]
        )
        
        audio.add_concept(
            name="synthwave",
            description="Retro 80s synthesizer aesthetic",
            technical_specs={
                "instruments": ["analog synths", "arpeggiated basslines", "gated reverb drums"],
                "tempo": "100-120 BPM",
                "effects": ["heavy reverb", "chorus", "side-chain compression"],
                "music_gen_prompt": "synthwave, retro, 80s, neon, analog synth, driving beat"
            },
            examples=["Kavinsky", "Carpenter Brut", "Perturbator"]
        )
        
        # Techniques
        audio.add_technique("sound layering", [
            "Start with base/fundamental sound",
            "Add mid-frequency body layer",
            "Add high-frequency detail/air",
            "Add transient/attack layer if needed",
            "EQ each layer to its own frequency space"
        ])
        
        audio.add_technique("tension building", [
            "Start sparse and quiet",
            "Gradually add layers",
            "Increase tempo or rhythm density",
            "Rise in pitch/frequency",
            "Cut to silence before climax"
        ])
        
        self._domains["audio"] = audio
    
    def _load_character_knowledge(self) -> None:
        """Character design and archetype knowledge."""
        char = DomainKnowledge(domain="character")
        
        char.add_concept(
            name="protagonist",
            description="Main character the audience follows",
            technical_specs={
                "visual_traits": ["distinctive silhouette", "warm colors", "open posture"],
                "design_principles": ["relatable flaws", "clear goals", "room to grow"],
                "sd_prompt_tags": ["protagonist", "hero", "determined expression", "dynamic pose"]
            }
        )
        
        char.add_concept(
            name="antagonist",
            description="Character opposing the protagonist",
            technical_specs={
                "visual_traits": ["angular shapes", "cool/dark colors", "imposing silhouette"],
                "design_principles": ["mirror protagonist", "understandable motivation", "threatening presence"],
                "sd_prompt_tags": ["villain", "antagonist", "menacing", "dark colors", "imposing"]
            }
        )
        
        char.add_concept(
            name="femme fatale",
            description="Mysterious, seductive dangerous woman archetype",
            technical_specs={
                "visual_traits": ["elegant silhouette", "red/black palette", "confident pose"],
                "characteristics": ["mysterious", "intelligent", "morally ambiguous"],
                "sd_prompt_tags": ["femme fatale", "elegant", "mysterious", "noir", "confident"]
            },
            examples=["Basic Instinct", "Double Indemnity", "LA Confidential"]
        )
        
        char.add_concept(
            name="wise mentor",
            description="Guide figure who helps protagonist",
            technical_specs={
                "visual_traits": ["aged appearance", "calm expression", "simple clothing"],
                "characteristics": ["experienced", "cryptic", "sacrificial"],
                "sd_prompt_tags": ["wise old man", "mentor", "sage", "calm", "knowing expression"]
            },
            examples=["Gandalf", "Obi-Wan", "Dumbledore"]
        )
        
        char.add_concept(
            name="kawaii",
            description="Japanese cute aesthetic for characters",
            technical_specs={
                "visual_traits": ["large eyes", "small nose/mouth", "round face", "pastel colors"],
                "proportions": ["large head", "small body", "stubby limbs"],
                "sd_prompt_tags": ["kawaii", "cute", "chibi", "big eyes", "pastel colors", "adorable"]
            }
        )
        
        char.add_concept(
            name="gritty realistic",
            description="Grounded, realistic character design",
            technical_specs={
                "visual_traits": ["weathered appearance", "practical clothing", "visible wear"],
                "characteristics": ["flawed", "scarred", "tired eyes"],
                "sd_prompt_tags": ["realistic", "gritty", "weathered", "detailed skin", "imperfect"]
            },
            examples=["The Last of Us", "Mad Max", "Logan"]
        )
        
        # Techniques
        char.add_technique("character silhouette", [
            "Design should be recognizable as black silhouette",
            "Unique shape language per character",
            "Avoid generic proportions",
            "Test: can you identify character from shadow alone?"
        ])
        
        char.add_technique("color coding", [
            "Assign primary color to character",
            "Color reflects personality (red=passionate, blue=calm)",
            "Maintain consistency across appearances",
            "Use color to show character development"
        ])
        
        self._domains["character"] = char
    
    def _load_environment_knowledge(self) -> None:
        """Environment and setting knowledge."""
        env = DomainKnowledge(domain="environment")
        
        env.add_concept(
            name="urban decay",
            description="Deteriorating city environments",
            technical_specs={
                "elements": ["graffiti", "broken windows", "overgrown plants", "rust", "debris"],
                "lighting": ["harsh shadows", "flickering lights", "orange sodium lamps"],
                "atmosphere": ["hazy", "dusty", "smoggy"],
                "sd_prompt_tags": ["urban decay", "abandoned", "overgrown", "graffiti", 
                                   "broken windows", "rust", "dystopian"]
            },
            examples=["The Last of Us", "Fallout", "District 9"]
        )
        
        env.add_concept(
            name="cozy interior",
            description="Warm, inviting indoor spaces",
            technical_specs={
                "elements": ["soft lighting", "warm colors", "plants", "books", "textiles"],
                "lighting": ["golden hour", "lamp light", "fireplace glow"],
                "atmosphere": ["intimate", "safe", "lived-in"],
                "sd_prompt_tags": ["cozy", "warm lighting", "interior", "comfortable", 
                                   "plants", "books", "soft textures"]
            },
            examples=["Studio Ghibli interiors", "Hygge aesthetic"]
        )
        
        env.add_concept(
            name="alien landscape",
            description="Otherworldly, non-Earth environments",
            technical_specs={
                "elements": ["unusual rock formations", "strange vegetation", "multiple moons",
                            "bioluminescence", "impossible geometry"],
                "colors": ["non-natural palette", "purple skies", "green suns"],
                "atmosphere": ["mysterious", "vast", "inhospitable"],
                "sd_prompt_tags": ["alien planet", "otherworldly", "sci-fi landscape",
                                   "strange vegetation", "multiple moons", "bioluminescent"]
            },
            examples=["Avatar Pandora", "Dune Arrakis", "No Man's Sky"]
        )
        
        env.add_concept(
            name="gothic architecture",
            description="Medieval European dark architectural style",
            technical_specs={
                "elements": ["pointed arches", "flying buttresses", "gargoyles", 
                            "stained glass", "vaulted ceilings", "spires"],
                "lighting": ["dramatic shadows", "candlelight", "shafts of light"],
                "atmosphere": ["imposing", "sacred", "mysterious"],
                "sd_prompt_tags": ["gothic architecture", "cathedral", "dark fantasy",
                                   "stone walls", "stained glass", "dramatic lighting"]
            },
            examples=["Dark Souls", "Castlevania", "Notre-Dame"]
        )
        
        env.add_concept(
            name="neon city",
            description="Futuristic urban nightscape with neon lighting",
            technical_specs={
                "elements": ["neon signs", "holograms", "flying vehicles", "rain",
                            "crowded streets", "towering buildings"],
                "lighting": ["neon glow", "reflections", "volumetric fog"],
                "atmosphere": ["busy", "overwhelming", "anonymous"],
                "sd_prompt_tags": ["neon city", "cyberpunk", "night city", "rain",
                                   "neon signs", "futuristic", "crowded streets"]
            },
            examples=["Blade Runner", "Ghost in the Shell", "Akira"]
        )
        
        env.add_concept(
            name="pastoral",
            description="Idyllic countryside and nature",
            technical_specs={
                "elements": ["rolling hills", "meadows", "streams", "cottages", "farmland"],
                "lighting": ["golden hour", "soft clouds", "dappled sunlight"],
                "atmosphere": ["peaceful", "nostalgic", "timeless"],
                "sd_prompt_tags": ["pastoral", "countryside", "meadow", "golden hour",
                                   "peaceful", "nature", "idyllic"]
            },
            examples=["Studio Ghibli landscapes", "Constable paintings"]
        )
        
        env.add_concept(
            name="underwater",
            description="Subaquatic environments",
            technical_specs={
                "elements": ["coral", "fish", "caustic light patterns", "bubbles", "kelp"],
                "lighting": ["blue/green tint", "god rays from surface", "bioluminescence"],
                "atmosphere": ["weightless", "mysterious", "alien"],
                "sd_prompt_tags": ["underwater", "ocean", "coral reef", "fish",
                                   "caustic lighting", "blue", "subaquatic"]
            },
            examples=["Finding Nemo", "Subnautica", "The Abyss"]
        )
        
        # Techniques
        env.add_technique("establishing atmosphere", [
            "Start with lighting direction and color temperature",
            "Add atmospheric effects (fog, dust, rain)",
            "Include environmental storytelling details",
            "Layer foreground, midground, background elements"
        ])
        
        env.add_technique("depth cues", [
            "Overlap elements front to back",
            "Use atmospheric perspective (distant = hazier)",
            "Decrease detail with distance",
            "Cool colors recede, warm colors advance"
        ])
        
        self._domains["environment"] = env
    
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

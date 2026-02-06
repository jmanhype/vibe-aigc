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
                "lighting": ["low-key", "venetian blind shadows", "single source", "chiaroscuro"],
                "color_grading": ["black and white", "high contrast", "crushed blacks", "silver gelatin look"],
                "composition": ["dutch angles", "deep shadows", "silhouettes", "framing through doorways"],
                "mood": ["cynical", "fatalistic", "morally ambiguous"],
                "sd_prompt_tags": ["film noir", "black and white", "dramatic shadows", "venetian blinds",
                                   "1940s", "detective", "femme fatale", "high contrast"]
            },
            examples=["Double Indemnity", "The Maltese Falcon", "The Third Man", "Chinatown"]
        )
        
        film.add_concept(
            name="blade runner",
            description="Neo-noir sci-fi aesthetic with neon-soaked rain and existential themes",
            technical_specs={
                "lighting": ["neon glow", "rain reflections", "volumetric fog", "teal-orange contrast"],
                "color_grading": ["teal and orange", "high contrast", "crushed shadows", "neon highlights"],
                "composition": ["wide establishing shots", "claustrophobic interiors", "eye-level conversations"],
                "atmosphere": ["rain", "smoke", "fog", "steam vents"],
                "sd_prompt_tags": ["blade runner", "neo-noir", "neon rain", "teal orange", "cyberpunk noir",
                                   "dystopian", "replicant", "futuristic noir", "wet streets"]
            },
            examples=["Blade Runner", "Blade Runner 2049", "Dark City", "The Crow"]
        )
        
        film.add_concept(
            name="neo-noir",
            description="Modern interpretation of noir with color and contemporary settings",
            technical_specs={
                "lighting": ["neon sources", "practical lights", "motivated shadows", "rim lighting"],
                "color_grading": ["teal-orange", "desaturated midtones", "neon accents", "high contrast"],
                "composition": ["urban landscapes", "reflective surfaces", "rain-slicked streets"],
                "themes": ["moral ambiguity", "corruption", "doomed protagonists"],
                "sd_prompt_tags": ["neo-noir", "modern noir", "neon noir", "crime drama",
                                   "night city", "rain", "moody", "cinematic"]
            },
            examples=["Drive", "Collateral", "Nightcrawler", "Sin City"]
        )
        
        film.add_concept(
            name="cyberpunk",
            description="High-tech dystopian future with stark class divides",
            technical_specs={
                "lighting": ["neon signs", "holographic displays", "harsh artificial light", "no natural light"],
                "color_grading": ["cyan-magenta", "neon pink", "electric blue", "toxic green"],
                "composition": ["dense vertical cities", "low angles", "overwhelming scale"],
                "elements": ["chrome", "holograms", "augmented humans", "corporate logos"],
                "sd_prompt_tags": ["cyberpunk", "neon city", "futuristic", "dystopian", "chrome",
                                   "holograms", "night", "dense urban", "high tech low life"]
            },
            examples=["Ghost in the Shell", "Akira", "The Matrix", "Altered Carbon"]
        )
        
        film.add_concept(
            name="anime",
            description="Japanese animation aesthetic and cinematic techniques",
            technical_specs={
                "lighting": ["rim light halos", "dramatic backlighting", "colored shadows"],
                "color_grading": ["vibrant saturation", "bold primaries", "gradient skies"],
                "composition": ["speed lines", "impact frames", "dramatic poses", "reaction shots"],
                "editing": ["still frames for impact", "rapid cuts", "held dramatic moments"],
                "sd_prompt_tags": ["anime", "cel shaded", "vibrant colors", "dynamic pose",
                                   "speed lines", "dramatic lighting", "Japanese animation"]
            },
            examples=["Akira", "Your Name", "Spirited Away", "Attack on Titan"]
        )
        
        film.add_concept(
            name="ghibli",
            description="Studio Ghibli's signature warm, whimsical aesthetic",
            technical_specs={
                "lighting": ["soft natural light", "golden hour warmth", "dappled sunlight through leaves"],
                "color_grading": ["soft pastels", "warm earth tones", "watercolor palette", "gentle saturation"],
                "composition": ["wide nature shots", "detailed backgrounds", "character in environment"],
                "atmosphere": ["peaceful", "nostalgic", "magical realism"],
                "sd_prompt_tags": ["studio ghibli", "ghibli style", "hand painted", "whimsical",
                                   "soft colors", "nature", "miyazaki", "totoro style"]
            },
            examples=["My Neighbor Totoro", "Spirited Away", "Howl's Moving Castle", "Princess Mononoke"]
        )
        
        film.add_concept(
            name="horror",
            description="Unsettling visual style designed to create fear and dread",
            technical_specs={
                "lighting": ["underlit faces", "flickering sources", "motivated darkness", "single harsh source"],
                "color_grading": ["desaturated", "sickly greens", "cold blues", "high contrast"],
                "composition": ["dutch angles", "negative space", "obscured threats", "deep shadows"],
                "techniques": ["fog", "practical effects", "sudden reveals", "slow builds"],
                "sd_prompt_tags": ["horror", "creepy", "dark", "fog", "desaturated",
                                   "dutch angle", "atmospheric horror", "dread"]
            },
            examples=["The Shining", "Hereditary", "The Exorcist", "A Quiet Place"]
        )
        
        film.add_concept(
            name="vaporwave",
            description="Retro-futuristic aesthetic blending 80s/90s nostalgia with surrealism",
            technical_specs={
                "lighting": ["neon pink/purple/cyan", "sunset gradients", "retro CRT glow"],
                "color_grading": ["pink-purple-cyan triad", "saturated pastels", "chrome reflections"],
                "composition": ["geometric grids", "floating objects", "Greek statues", "retro tech"],
                "elements": ["VHS artifacts", "glitch effects", "palm trees", "dolphins", "sunsets"],
                "sd_prompt_tags": ["vaporwave", "aesthetic", "retrowave", "80s", "pink purple cyan",
                                   "neon", "glitch", "geometric", "retro futurism"]
            },
            examples=["Hotline Miami", "Far Cry 3: Blood Dragon", "Kung Fury"]
        )
        
        film.add_concept(
            name="photorealistic",
            description="Ultra-realistic visual presentation indistinguishable from reality",
            technical_specs={
                "lighting": ["natural light behavior", "global illumination", "subsurface scattering"],
                "color_grading": ["minimal grading", "accurate skin tones", "natural contrast"],
                "technical": ["high resolution (4K/8K)", "high frame rate option", "detailed textures"],
                "composition": ["natural framing", "realistic depth of field", "authentic motion blur"],
                "sd_prompt_tags": ["photorealistic", "hyperrealistic", "8k uhd", "RAW photo",
                                   "detailed skin", "subsurface scattering", "ray tracing"]
            },
            examples=["Avatar", "The Lion King (2019)", "Planet Earth documentaries"]
        )
        
        film.add_concept(
            name="impressionist",
            description="Painterly visual style emphasizing light and atmosphere over detail",
            technical_specs={
                "lighting": ["diffused light", "color-in-shadow", "light as subject"],
                "color_grading": ["visible color strokes", "complementary shadows", "vibrant but soft"],
                "composition": ["en plein air framing", "everyday scenes", "emphasis on atmosphere"],
                "texture": ["visible brushstrokes", "soft edges", "broken color"],
                "sd_prompt_tags": ["impressionist", "painterly", "visible brushstrokes", "monet style",
                                   "soft light", "atmospheric", "oil painting", "plein air"]
            },
            examples=["Loving Vincent", "What Dreams May Come", "Monet/Renoir paintings"]
        )
        
        film.add_concept(
            name="surrealist",
            description="Dreamlike imagery that defies logical reality",
            technical_specs={
                "lighting": ["impossible light sources", "dramatic shadows", "theatrical"],
                "color_grading": ["heightened reality", "symbolic color", "dreamlike palette"],
                "composition": ["impossible geometry", "scale distortion", "juxtaposition"],
                "elements": ["melting objects", "floating elements", "recursive imagery", "symbolic objects"],
                "sd_prompt_tags": ["surrealist", "surrealism", "dreamlike", "dali style",
                                   "impossible geometry", "melting", "floating", "ethereal"]
            },
            examples=["Eternal Sunshine of the Spotless Mind", "Pan's Labyrinth", "Mulholland Drive", "Dali paintings"]
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
            description="Futuristic dystopian aesthetic with neon, chrome, and high-tech urban decay",
            technical_specs={
                "sd_prompt_tags": ["cyberpunk", "neon lights", "futuristic city", "rain", 
                                   "holographic displays", "chrome", "dystopian", "night city",
                                   "dense urban", "neon signs", "megacity", "high tech low life",
                                   "augmented human", "corporate dystopia", "flying cars"],
                "color_palette": ["#FF00FF", "#00FFFF", "#FF0080", "#00FF80", "#1a1a2e", "#39FF14"],
                "lighting": ["neon glow", "rim lighting", "volumetric fog", "light rays",
                            "holographic light", "advertisement screens", "no natural light"],
                "composition": ["low angle", "wide shot", "reflective surfaces", "vertical cities",
                               "overwhelming scale", "crowded streets", "layered depth"],
                "elements": ["chrome surfaces", "holograms", "neon signs", "rain", "steam",
                            "cables and wires", "augmented reality overlays"],
                "negative_prompt": ["natural lighting", "daytime", "pastoral", "clean", "rural",
                                   "historical", "medieval", "nature"]
            },
            examples=["Blade Runner", "Ghost in the Shell", "Akira", "Cyberpunk 2077", "The Matrix"]
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
            name="blade runner",
            description="Neo-noir sci-fi with rain, neon, and teal-orange color grading",
            technical_specs={
                "sd_prompt_tags": ["blade runner", "neo-noir", "neon rain", "cyberpunk noir",
                                   "wet streets", "neon reflections", "teal orange color grading",
                                   "dystopian city", "replicant", "futuristic noir", "smog",
                                   "holographic advertisements", "flying cars"],
                "color_palette": ["#008080", "#FF6B35", "#1a1a2e", "#FF00FF", "#00FFFF"],
                "lighting": ["neon glow", "rain reflections", "volumetric fog", "teal-orange contrast",
                            "backlit silhouettes", "practical neon sources"],
                "composition": ["wide establishing shots", "eye-level dialogue", "reflective puddles",
                               "steam vents", "layered depth"],
                "negative_prompt": ["bright", "sunny", "daytime", "cheerful", "clean", "pastoral"]
            },
            examples=["Blade Runner", "Blade Runner 2049", "Dark City"]
        )
        
        visual.add_concept(
            name="anime",
            description="Japanese animation style with cel shading and dynamic energy",
            technical_specs={
                "sd_prompt_tags": ["anime", "cel shading", "vibrant colors", 
                                   "detailed eyes", "dramatic pose", "speed lines",
                                   "dynamic action", "anime lighting", "manga style"],
                "models": ["anything-v5", "counterfeit-v3", "waifu-diffusion", "animagine-xl"],
                "cfg_scale": 7,
                "lighting": ["rim light halos", "dramatic backlighting", "colored shadows",
                            "anime glow", "light rays"],
                "composition": ["dynamic angles", "action poses", "impact frames", "close-up reactions"],
                "negative_prompt": ["realistic", "photographic", "3d render", "western", "blurry"]
            },
            examples=["Demon Slayer", "Attack on Titan", "My Hero Academia"]
        )
        
        visual.add_concept(
            name="ghibli",
            description="Studio Ghibli's soft, whimsical, hand-painted aesthetic",
            technical_specs={
                "sd_prompt_tags": ["studio ghibli", "ghibli style", "hayao miyazaki",
                                   "hand painted", "soft colors", "whimsical", "nature",
                                   "watercolor", "peaceful", "magical", "totoro style",
                                   "detailed background", "cozy"],
                "color_palette": ["#87CEEB", "#90EE90", "#F5DEB3", "#FFB6C1", "#E6E6FA"],
                "lighting": ["soft natural light", "golden hour", "dappled sunlight",
                            "gentle shadows", "warm diffused light"],
                "composition": ["wide landscape shots", "character in nature", "detailed environments",
                               "peaceful scenes", "flying sequences"],
                "cfg_scale": 7,
                "negative_prompt": ["dark", "gritty", "realistic", "violent", "horror", "cyberpunk"]
            },
            examples=["Spirited Away", "My Neighbor Totoro", "Howl's Moving Castle", "Kiki's Delivery Service"]
        )
        
        visual.add_concept(
            name="noir",
            description="Classic film noir with black and white, dramatic shadows",
            technical_specs={
                "sd_prompt_tags": ["film noir", "black and white", "monochrome",
                                   "dramatic shadows", "venetian blind shadows", "1940s",
                                   "detective", "femme fatale", "high contrast", "chiaroscuro",
                                   "moody", "cigarette smoke", "fedora"],
                "color_palette": ["#000000", "#FFFFFF", "#333333", "#666666", "#999999"],
                "lighting": ["low-key lighting", "single source", "venetian blinds",
                            "hard shadows", "silhouettes", "rim light"],
                "composition": ["dutch angles", "framing through doorways", "long shadows",
                               "reflections", "staircase shots"],
                "negative_prompt": ["color", "colorful", "bright", "cheerful", "daytime", "sunny"]
            },
            examples=["Double Indemnity", "The Maltese Falcon", "The Third Man", "Sin City"]
        )
        
        visual.add_concept(
            name="horror",
            description="Unsettling horror aesthetic with desaturation and dread",
            technical_specs={
                "sd_prompt_tags": ["horror", "creepy", "dark", "atmospheric horror",
                                   "fog", "mist", "desaturated", "gloomy", "ominous",
                                   "dread", "unsettling", "eerie", "sinister"],
                "color_palette": ["#1a1a2e", "#2d3436", "#636e72", "#74b9ff", "#a29bfe"],
                "lighting": ["underlit faces", "flickering light", "harsh single source",
                            "motivated darkness", "cold blue light", "practical candles"],
                "composition": ["dutch angles", "negative space", "obscured threats",
                               "deep shadows", "off-center framing", "low angles"],
                "atmosphere": ["fog", "mist", "rain", "decay", "abandoned"],
                "negative_prompt": ["bright", "cheerful", "colorful", "sunny", "happy", "vibrant"]
            },
            examples=["The Shining", "Hereditary", "Silent Hill", "Resident Evil"]
        )
        
        visual.add_concept(
            name="vaporwave",
            description="Retro 80s/90s aesthetic with pink, purple, cyan, and glitch effects",
            technical_specs={
                "sd_prompt_tags": ["vaporwave", "aesthetic", "retrowave", "80s aesthetic",
                                   "neon pink", "neon purple", "cyan", "glitch art",
                                   "geometric", "grid", "palm trees", "sunset gradient",
                                   "greek statue", "VHS", "retro futurism", "synthwave"],
                "color_palette": ["#FF6AD5", "#C774E8", "#AD8CFF", "#8795E8", "#94D0FF", "#00FFFF"],
                "lighting": ["neon glow", "sunset gradient", "CRT screen glow",
                            "chrome reflections", "rim lighting"],
                "composition": ["geometric grids", "floating objects", "perspective grids",
                               "tiled patterns", "chrome spheres", "sunset backgrounds"],
                "effects": ["VHS artifacts", "scan lines", "glitch", "chromatic aberration"],
                "negative_prompt": ["realistic", "natural", "muted colors", "organic", "modern"]
            },
            examples=["Hotline Miami", "Far Cry 3: Blood Dragon", "Kung Fury"]
        )
        
        visual.add_concept(
            name="impressionist",
            description="Painterly style with visible brushstrokes and light effects",
            technical_specs={
                "sd_prompt_tags": ["impressionist", "impressionism", "oil painting",
                                   "visible brushstrokes", "monet style", "renoir style",
                                   "plein air", "soft light", "atmospheric", "painterly",
                                   "dappled light", "broken color"],
                "color_palette": ["#87CEEB", "#F0E68C", "#DDA0DD", "#98FB98", "#FFB6C1"],
                "lighting": ["diffused natural light", "dappled sunlight", "golden hour",
                            "reflected light", "color in shadows"],
                "composition": ["en plein air framing", "everyday scenes", "landscapes",
                               "water reflections", "garden scenes"],
                "texture": ["thick impasto", "broken color", "soft edges", "atmospheric blur"],
                "negative_prompt": ["sharp", "photorealistic", "digital", "clean lines", "flat color"]
            },
            examples=["Claude Monet", "Pierre-Auguste Renoir", "Loving Vincent"]
        )
        
        visual.add_concept(
            name="surrealist",
            description="Dreamlike imagery with impossible geometry and Dali-esque elements",
            technical_specs={
                "sd_prompt_tags": ["surrealist", "surrealism", "dali style", "dreamlike",
                                   "impossible geometry", "melting", "floating objects",
                                   "ethereal", "magritte style", "escher style",
                                   "recursive", "distorted reality", "symbolic"],
                "color_palette": ["#E6D5B8", "#87CEEB", "#F4A460", "#DEB887", "#C0C0C0"],
                "lighting": ["dramatic shadows", "impossible light sources", "theatrical",
                            "dreamlike glow", "multiple light sources"],
                "composition": ["impossible perspectives", "scale distortion", "juxtaposition",
                               "floating elements", "infinite landscapes", "recursive patterns"],
                "elements": ["melting clocks", "floating stones", "impossible architecture",
                            "symbolic objects", "metamorphosis"],
                "negative_prompt": ["realistic", "normal", "ordinary", "mundane", "photographic"]
            },
            examples=["Salvador Dali", "Rene Magritte", "M.C. Escher", "Pan's Labyrinth"]
        )
        
        visual.add_concept(
            name="photorealistic",
            description="Ultra-realistic photo-quality images indistinguishable from photographs",
            technical_specs={
                "sd_prompt_tags": ["photorealistic", "hyperrealistic", "8k uhd", "RAW photo",
                                   "dslr", "high detail", "sharp focus", "detailed skin texture",
                                   "subsurface scattering", "ray tracing", "global illumination",
                                   "studio lighting", "professional photography"],
                "models": ["realistic-vision", "deliberate", "photon", "juggernaut-xl"],
                "cfg_scale": 5,
                "steps": 30,
                "lighting": ["natural light", "studio lighting", "soft diffused light",
                            "golden hour", "rim lighting", "catch lights in eyes"],
                "composition": ["shallow depth of field", "bokeh", "eye-level perspective",
                               "professional framing", "natural poses"],
                "color_palette": ["natural skin tones", "accurate colors", "subtle grading"],
                "negative_prompt": ["cartoon", "anime", "painting", "illustration", 
                                   "drawing", "artificial", "fake", "oversaturated",
                                   "plastic skin", "doll-like", "cgi"]
            },
            examples=["Stock photography", "Portrait photography", "Product photography"]
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
        """Audio, sound design, and film scoring knowledge."""
        audio = DomainKnowledge(domain="audio")
        
        # === MUSICAL STYLES ===
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
                "intervals": ["minor 2nd", "tritone", "diminished chords"],
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
        
        # === FILM SCORING CONCEPTS ===
        audio.add_concept(
            name="leitmotif",
            description="Recurring musical theme associated with a character, place, or idea",
            technical_specs={
                "usage": ["character introduction", "emotional callback", "thematic unity"],
                "implementation": ["short melodic phrase", "distinctive instrumentation", 
                                  "variations for mood changes"],
                "variations": ["major/minor for hero/villain", "tempo for tension",
                              "orchestration for scale"],
                "music_gen_prompt": "memorable melody, thematic, cinematic, character theme"
            },
            examples=["Imperial March (Vader)", "Hedwig's Theme (Harry Potter)", 
                     "The Shire (Lord of the Rings)", "Jaws theme"]
        )
        
        audio.add_concept(
            name="diegetic",
            description="Sound that exists within the story world (characters can hear it)",
            technical_specs={
                "types": ["radio music", "live performance", "ambient sounds", "dialogue"],
                "mixing": ["room acoustics", "distance attenuation", "occlusion"],
                "contrast": "Can transition to non-diegetic for emotional effect",
                "examples_in_film": ["record player scene", "band at party", "car radio"]
            },
            examples=["Guardians of the Galaxy Awesome Mix", "Baby Driver car scenes"]
        )
        
        audio.add_concept(
            name="non-diegetic",
            description="Sound added for audience only (score, narration, sound effects)",
            technical_specs={
                "types": ["underscore", "voiceover narration", "emotional sound design"],
                "purpose": ["guide emotion", "build tension", "signal genre"],
                "mixing": ["full stereo/surround", "clean signal", "no room tone"],
                "technique": "Often starts diegetic then swells to non-diegetic"
            },
            examples=["Film scores", "Documentary narration", "Laugh tracks"]
        )
        
        audio.add_concept(
            name="underscore",
            description="Background music supporting scenes without being noticed",
            technical_specs={
                "characteristics": ["subtle", "supports dialogue", "emotionally guiding"],
                "mixing": ["ducking under dialogue", "sparse arrangement", "low register"],
                "music_gen_prompt": "subtle underscore, minimal, background, cinematic, gentle"
            }
        )
        
        audio.add_concept(
            name="stinger",
            description="Short, sharp musical accent for dramatic effect",
            technical_specs={
                "usage": ["jump scares", "reveals", "dramatic moments", "scene transitions"],
                "characteristics": ["sudden", "dissonant", "brief (1-3 seconds)"],
                "instruments": ["brass hits", "string stabs", "percussion hits", "synth blasts"],
                "music_gen_prompt": "dramatic hit, stinger, impact, brass, sudden"
            },
            examples=["Horror jump scares", "Netflix 'tudum'", "Law & Order 'dun dun'"]
        )
        
        # === GENRE MOODS ===
        audio.add_concept(
            name="action score",
            description="High-energy music for action sequences",
            technical_specs={
                "tempo": "120-160 BPM",
                "elements": ["driving percussion", "brass stabs", "string ostinatos",
                            "synth bass", "rhythmic urgency"],
                "dynamics": ["loud overall", "impact hits", "brief quiet for contrast"],
                "structure": ["building intensity", "peaks at action climax"],
                "music_gen_prompt": "action, intense, percussion, brass, driving beat, cinematic"
            },
            examples=["Mad Max Fury Road", "John Wick", "Mission Impossible"]
        )
        
        audio.add_concept(
            name="romantic score",
            description="Emotional, sweeping music for love scenes",
            technical_specs={
                "tempo": "60-80 BPM",
                "elements": ["strings", "piano", "woodwinds", "harp", "warm pads"],
                "harmony": ["major keys", "suspended chords", "rising progressions"],
                "dynamics": ["soft to swelling", "intimate to grand"],
                "music_gen_prompt": "romantic, emotional, strings, piano, sweeping, tender"
            },
            examples=["Titanic", "The Notebook", "Pride and Prejudice"]
        )
        
        audio.add_concept(
            name="thriller score",
            description="Suspenseful music building tension and unease",
            technical_specs={
                "tempo": "variable, often accelerating",
                "elements": ["ostinato patterns", "rising pitch", "dissonant intervals",
                            "sparse textures", "electronic elements"],
                "techniques": ["silence", "sudden dynamics", "heartbeat rhythms"],
                "music_gen_prompt": "suspense, tension, thriller, dark, building, ominous"
            },
            examples=["Inception", "Sicario", "Gone Girl"]
        )
        
        audio.add_concept(
            name="comedy score",
            description="Light, playful music for comedic moments",
            technical_specs={
                "tempo": "100-140 BPM (often bouncy)",
                "elements": ["pizzicato strings", "woodwinds", "xylophone", "light percussion"],
                "characteristics": ["staccato", "major keys", "playful melodies", "cartoonish"],
                "techniques": ["mickey-mousing (matching action)", "awkward pauses"],
                "music_gen_prompt": "comedy, playful, light, bouncy, whimsical, quirky"
            },
            examples=["Home Alone", "The Grand Budapest Hotel", "Pixar films"]
        )
        
        audio.add_concept(
            name="western score",
            description="Classic American frontier sound",
            technical_specs={
                "elements": ["solo trumpet", "harmonica", "acoustic guitar", "whistling",
                            "twangy electric guitar", "orchestral swells"],
                "characteristics": ["wide open spaces", "heroic themes", "lonely melodies"],
                "music_gen_prompt": "western, frontier, harmonica, acoustic guitar, cinematic"
            },
            examples=["Ennio Morricone", "Django Unchained", "No Country for Old Men"]
        )
        
        # === PACING CONCEPTS ===
        audio.add_concept(
            name="musical pacing",
            description="Using tempo and rhythm to control scene energy",
            technical_specs={
                "slow_scene": {"tempo": "40-70 BPM", "usage": "emotional, reflective, intimate"},
                "medium_scene": {"tempo": "80-110 BPM", "usage": "dialogue, exploration, transition"},
                "fast_scene": {"tempo": "120-180 BPM", "usage": "action, chase, climax"},
                "techniques": ["tempo acceleration for building tension",
                              "tempo deceleration for resolution",
                              "rubato for emotional emphasis"]
            }
        )
        
        audio.add_concept(
            name="dynamic arc",
            description="Volume and intensity shaping over time",
            technical_specs={
                "shapes": {
                    "crescendo": "quiet to loud, building anticipation",
                    "decrescendo": "loud to quiet, resolution or dread",
                    "swell": "quiet-loud-quiet, emotional peaks",
                    "plateau": "sustained intensity for action"
                },
                "mixing": ["automate volume", "add/remove layers", "EQ brightness"]
            }
        )
        
        audio.add_concept(
            name="silence",
            description="Strategic use of no music for dramatic effect",
            technical_specs={
                "usage": ["before jump scare", "after revelation", "intimate dialogue",
                         "documentary realism", "uncomfortable tension"],
                "duration": {"brief": "1-3 sec impact", "extended": "10+ sec dread/realism"},
                "techniques": ["cut music abruptly for shock", "fade out for resolution"]
            },
            examples=["No Country for Old Men (mostly no score)", "A Quiet Place"]
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
        
        audio.add_technique("spotting", [
            "Watch scene without music to understand natural rhythm",
            "Identify key emotional beats and hit points",
            "Decide diegetic vs non-diegetic needs",
            "Note dialogue that needs music ducking",
            "Plan music entry/exit points at natural scene breaks"
        ])
        
        audio.add_technique("mickey-mousing", [
            "Match music directly to on-screen action",
            "Accent movements with stings or motifs",
            "Use for comedy or heightened stylization",
            "Avoid for realistic drama (too cartoonish)"
        ])
        
        # Constraints
        audio.add_constraint("dialogue scenes", [
            "Keep underscore minimal and low-register",
            "Avoid melodic content that competes with speech",
            "Duck music volume during dialogue",
            "Enter/exit music during pauses when possible"
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

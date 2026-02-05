"""
Workflow Composer - AI agent that edits ComfyUI workflows based on VLM feedback.

Uses:
- Gemini VLM to analyze generated outputs
- Comfy-Pilot to modify workflows
- Iterative improvement loop

The AI can SEE what it generates and MODIFY the workflow to improve it.
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Add comfy-pilot to path
COMFY_PILOT_PATH = Path("C:/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfy-pilot")
if COMFY_PILOT_PATH.exists():
    sys.path.insert(0, str(COMFY_PILOT_PATH))

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


@dataclass
class ComposerConfig:
    """Configuration for the workflow composer."""
    comfyui_url: str = "http://127.0.0.1:8188"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-flash-latest"
    perplexity_api_key: Optional[str] = None
    max_iterations: int = 3
    quality_threshold: float = 8.0  # Stop if quality >= this
    output_dir: Path = Path("./composed_outputs")


class WorkflowComposer:
    """
    AI agent that composes and refines ComfyUI workflows.
    
    Loop:
    1. Execute workflow
    2. Analyze output with Gemini VLM
    3. Get improvement suggestions
    4. Modify workflow using Comfy-Pilot patterns
    5. Repeat until quality threshold or max iterations
    """
    
    def __init__(self, config: Optional[ComposerConfig] = None):
        self.config = config or ComposerConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Gemini if available
        if HAS_GENAI and self.config.gemini_api_key:
            import warnings
            warnings.filterwarnings('ignore')
            genai.configure(api_key=self.config.gemini_api_key)
            self.vlm = genai.GenerativeModel(self.config.gemini_model)
        else:
            self.vlm = None
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.history: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_available_nodes(self) -> Dict[str, Any]:
        """Get all available ComfyUI nodes."""
        async with self.session.get(f"{self.config.comfyui_url}/object_info") as resp:
            return await resp.json()
    
    async def research(self, topic: str) -> str:
        """Use Perplexity to research a topic for better prompts."""
        if not self.config.perplexity_api_key:
            return ""
        
        import urllib.request
        
        data = json.dumps({
            "model": "sonar",
            "messages": [{
                "role": "user", 
                "content": f"For AI image generation, describe the visual characteristics of: {topic}. Focus on colors, textures, lighting, composition. Be concise."
            }]
        }).encode()
        
        req = urllib.request.Request(
            "https://api.perplexity.ai/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.perplexity_api_key}",
                "Content-Type": "application/json"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Research error: {e}")
            return ""
    
    async def execute_workflow(self, workflow: Dict) -> Dict[str, Any]:
        """Execute a workflow and return results."""
        async with self.session.post(
            f"{self.config.comfyui_url}/prompt",
            json={"prompt": workflow, "client_id": "composer"}
        ) as resp:
            result = await resp.json()
            if "error" in result:
                return {"error": result}
            prompt_id = result["prompt_id"]
        
        # Wait for completion
        for _ in range(120):
            await asyncio.sleep(1)
            async with self.session.get(
                f"{self.config.comfyui_url}/history/{prompt_id}"
            ) as resp:
                hist = await resp.json()
                if prompt_id in hist:
                    status = hist[prompt_id].get("status", {}).get("status_str")
                    if status == "success":
                        return {
                            "prompt_id": prompt_id,
                            "outputs": hist[prompt_id].get("outputs", {})
                        }
                    elif status == "error":
                        return {"error": hist[prompt_id]}
        
        return {"error": "Timeout waiting for workflow"}
    
    async def download_image(self, filename: str) -> Optional[Path]:
        """Download an image from ComfyUI."""
        url = f"{self.config.comfyui_url}/view?filename={filename}"
        async with self.session.get(url) as resp:
            if resp.status == 200:
                path = self.config.output_dir / filename
                path.write_bytes(await resp.read())
                return path
        return None
    
    def analyze_image(self, image_path: Path, context: str = "") -> Dict[str, Any]:
        """Use Gemini VLM to analyze an image."""
        if not self.vlm or not HAS_PIL:
            return {"error": "VLM not available"}
        
        img = Image.open(image_path)
        
        prompt = f"""You are an AI art director analyzing generated images.

Context: {context}

Analyze this image and respond in JSON format:
{{
    "quality_score": <1-10>,
    "description": "<what you see>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "prompt_improvements": ["<specific prompt addition>", ...],
    "parameter_changes": {{
        "cfg": <suggested cfg or null>,
        "steps": <suggested steps or null>,
        "sampler": "<suggested sampler or null>"
    }}
}}

Be specific about what to ADD to the prompt to fix issues.
Focus on actionable improvements."""

        response = self.vlm.generate_content([prompt, img])
        
        # Parse JSON from response
        try:
            text = response.text
            # Extract JSON from markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return {
                "quality_score": 5,
                "description": response.text[:500],
                "raw_response": response.text
            }
    
    # Valid ComfyUI sampler names
    VALID_SAMPLERS = [
        "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
        "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
        "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
        "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc",
        "uni_pc_bh2"
    ]
    
    def modify_workflow(
        self, 
        workflow: Dict, 
        analysis: Dict[str, Any]
    ) -> Dict:
        """Modify workflow based on VLM analysis."""
        modified = json.loads(json.dumps(workflow))  # Deep copy
        
        # Find prompt nodes (CLIPTextEncode)
        for node_id, node in modified.items():
            if node.get("class_type") == "CLIPTextEncode":
                current_text = node["inputs"].get("text", "")
                
                # Check if this is positive prompt (not negative)
                if "negative" not in node_id.lower() and not any(
                    neg in current_text.lower() 
                    for neg in ["blur", "bad", "ugly", "deform", "low quality"]
                ):
                    # Add improvements to prompt
                    improvements = analysis.get("prompt_improvements", [])
                    if improvements:
                        additions = ", ".join(improvements[:3])
                        node["inputs"]["text"] = f"{current_text}, {additions}"
        
        # Modify KSampler parameters if suggested
        param_changes = analysis.get("parameter_changes", {})
        for node_id, node in modified.items():
            if node.get("class_type") == "KSampler":
                if param_changes.get("cfg"):
                    node["inputs"]["cfg"] = param_changes["cfg"]
                if param_changes.get("steps"):
                    node["inputs"]["steps"] = min(param_changes["steps"], 30)  # Cap steps
                if param_changes.get("sampler"):
                    # Validate sampler name
                    sampler = param_changes["sampler"].lower().replace(" ", "_").replace("+", "p")
                    if sampler in self.VALID_SAMPLERS:
                        node["inputs"]["sampler_name"] = sampler
                    elif "dpm" in sampler:
                        node["inputs"]["sampler_name"] = "dpmpp_2m"  # Safe default
                # Always change seed for variation
                node["inputs"]["seed"] = node["inputs"].get("seed", 0) + 1
        
        return modified
    
    async def compose(
        self, 
        initial_workflow: Dict,
        goal: str = "Generate a high-quality image"
    ) -> Dict[str, Any]:
        """
        Main composition loop.
        
        Args:
            initial_workflow: Starting workflow
            goal: What we're trying to achieve
        
        Returns:
            Final results with history
        """
        workflow = initial_workflow
        self.history = []
        
        for iteration in range(self.config.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.config.max_iterations} ===")
            
            # Execute workflow
            print("Executing workflow...")
            result = await self.execute_workflow(workflow)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                break
            
            # Find and download output image
            image_path = None
            for node_id, outputs in result.get("outputs", {}).items():
                if "images" in outputs:
                    for img_info in outputs["images"]:
                        image_path = await self.download_image(img_info["filename"])
                        break
            
            if not image_path:
                print("No image output found")
                break
            
            print(f"Generated: {image_path.name}")
            
            # Analyze with VLM
            print("Analyzing with Gemini VLM...")
            analysis = self.analyze_image(image_path, goal)
            
            quality = analysis.get("quality_score", 0)
            print(f"Quality: {quality}/10")
            print(f"Description: {analysis.get('description', 'N/A')[:100]}...")
            
            # Record history
            self.history.append({
                "iteration": iteration + 1,
                "image": str(image_path),
                "analysis": analysis,
                "workflow_snapshot": workflow
            })
            
            # Check if we've reached quality threshold
            if quality >= self.config.quality_threshold:
                print(f"✓ Quality threshold reached ({quality} >= {self.config.quality_threshold})")
                break
            
            # Modify workflow for next iteration
            if iteration < self.config.max_iterations - 1:
                print("Modifying workflow based on feedback...")
                workflow = self.modify_workflow(workflow, analysis)
                
                # Show what changed
                improvements = analysis.get("prompt_improvements", [])
                if improvements:
                    print(f"  Adding to prompt: {', '.join(improvements[:3])}")
        
        return {
            "final_image": str(image_path) if image_path else None,
            "final_quality": analysis.get("quality_score", 0),
            "iterations": len(self.history),
            "history": self.history
        }


async def demo():
    """Demo the workflow composer."""
    
    # Load API keys
    config_path = Path("C:/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Gemini/config.json")
    gemini_key = None
    if config_path.exists():
        with open(config_path) as f:
            gemini_key = json.load(f).get("GEMINI_API_KEY")
    
    if not gemini_key:
        print("No Gemini API key found!")
        return
    
    # Perplexity key for research
    perplexity_key = "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"
    
    config = ComposerConfig(
        gemini_api_key=gemini_key,
        perplexity_api_key=perplexity_key,
        max_iterations=3,
        quality_threshold=8.0,
        output_dir=Path("./composed_outputs")
    )
    
    # Simple test workflow
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "a cyberpunk android in neon city"
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "blurry, low quality, deformed"
            }
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 512, "width": 512}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7,
                "denoise": 1,
                "latent_image": ["4", 0],
                "model": ["1", 0],
                "negative": ["3", 0],
                "positive": ["2", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 12345,
                "steps": 15
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "composed", "images": ["6", 0]}
        }
    }
    
    async with WorkflowComposer(config) as composer:
        print("=== WORKFLOW COMPOSER DEMO ===")
        print("Goal: Generate a high-quality cyberpunk image")
        print("Max iterations:", config.max_iterations)
        print("Quality threshold:", config.quality_threshold)
        
        result = await composer.compose(
            workflow, 
            goal="Generate a detailed, high-quality cyberpunk android image"
        )
        
        print("\n=== FINAL RESULTS ===")
        print(f"Final image: {result['final_image']}")
        print(f"Final quality: {result['final_quality']}/10")
        print(f"Iterations: {result['iterations']}")
        
        # Show improvement history
        print("\nImprovement History:")
        for h in result["history"]:
            score = h["analysis"].get("quality_score", "?")
            improvements = h["analysis"].get("prompt_improvements", [])
            print(f"  Iteration {h['iteration']}: {score}/10")
            if improvements:
                print(f"    -> Added: {', '.join(improvements[:2])}")


if __name__ == "__main__":
    asyncio.run(demo())

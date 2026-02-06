"""Debug VLM feedback to see raw responses."""

import os
from pathlib import Path
from vibe_aigc.vlm_feedback import VLMFeedback

# Initialize VLM
vlm = VLMFeedback()
print(f"VLM available: {vlm.available}")

# Analyze test image
image_path = Path("temp_image.png")
if image_path.exists():
    print(f"\nAnalyzing: {image_path}")
    
    context = "cyberpunk samurai warrior standing in neon-lit Tokyo street, rain, cinematic lighting, detailed armor, moody atmosphere"
    
    result = vlm.analyze_image(image_path, context)
    
    print("\n=== RAW RESULT ===")
    print(f"Quality Score: {result.quality_score}")
    print(f"Description: {result.description}")
    print(f"Strengths: {result.strengths}")
    print(f"Weaknesses: {result.weaknesses}")
    print(f"Prompt Improvements: {result.prompt_improvements}")
    print(f"Parameter Changes: {result.parameter_changes}")
    print(f"\nRaw Response:\n{result.raw_response}")
else:
    print("temp_image.png not found!")

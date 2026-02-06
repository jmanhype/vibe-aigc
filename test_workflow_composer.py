"""Test the workflow composer."""

import json
from vibe_aigc.workflow_composer import (
    WorkflowComposer, Tools,
    compose_txt2img, compose_wan_video
)

print("=" * 60)
print("WORKFLOW COMPOSER TEST")
print("=" * 60)

# Test 1: Manual composition
print("\n1. Manual Composition (txt2img):")
print("-" * 40)

c = WorkflowComposer()

c.add("loader", Tools.model_loader("checkpoint"), ckpt_name="v1-5-pruned.safetensors")
c.add("pos", Tools.text_encode(), text="a cyberpunk cat")
c.add("neg", Tools.text_encode(), text="bad quality")
c.add("latent", Tools.empty_latent(), width=512, height=512)
c.add("sampler", Tools.ksampler(), seed=42, steps=20, cfg=7.0)
c.add("decode", Tools.vae_decode())
c.add("save", Tools.save_image())

c.wire("loader", "clip", "pos", "clip")
c.wire("loader", "clip", "neg", "clip")
c.wire("loader", "model", "sampler", "model")
c.wire("pos", "conditioning", "sampler", "positive")
c.wire("neg", "conditioning", "sampler", "negative")
c.wire("latent", "latent", "sampler", "latent_image")
c.wire("sampler", "latent", "decode", "samples")
c.wire("loader", "vae", "decode", "vae")
c.wire("decode", "image", "save", "images")

# Validate
issues = c.validate()
if issues:
    print("Issues:", issues)
else:
    print("[OK] Validation passed")

# Build
workflow = c.build()
print(f"[OK] Built workflow with {len(workflow)} nodes")
print(f"  Node types: {[n['class_type'] for n in workflow.values()]}")

# Test 2: Template composition
print("\n2. Template Composition (txt2img):")
print("-" * 40)

workflow = compose_txt2img(
    model_file="v1-5-pruned.safetensors",
    prompt="a samurai in neon rain",
    negative_prompt="ugly, blurry",
    width=768,
    height=512,
    steps=25,
    cfg=8.0,
    seed=123
)
print(f"[OK] Built workflow with {len(workflow)} nodes")

# Test 3: Wan video composition
print("\n3. Wan Video Composition:")
print("-" * 40)

workflow = compose_wan_video(
    model_file="Wan2_2-I2V-A14B-HIGH_fp8.safetensors",
    clip_file="umt5_xxl_fp8.safetensors",
    vae_file="wan_2.1_vae.safetensors",
    prompt="cyberpunk samurai walking through neon city",
    negative_prompt="static, blurry, distorted",
    width=512,
    height=512,
    frames=24,
    steps=6,
    cfg=6.0,
    seed=42
)
print(f"[OK] Built workflow with {len(workflow)} nodes")
print(f"  Node types: {[n['class_type'] for n in workflow.values()]}")

# Show the workflow structure
print("\n4. Workflow Structure (Wan Video):")
print("-" * 40)
for node_id, node in workflow.items():
    inputs_summary = []
    for k, v in node.get('inputs', {}).items():
        if isinstance(v, list):
            inputs_summary.append(f"{k}←node{v[0]}")
        elif isinstance(v, str) and len(v) > 20:
            inputs_summary.append(f"{k}=\"{v[:17]}...\"")
        else:
            inputs_summary.append(f"{k}={v}")
    print(f"  [{node_id}] {node['class_type']}")
    if inputs_summary:
        print(f"      {', '.join(inputs_summary[:3])}")

print("\n" + "=" * 60)
print("SUCCESS: Composer can build workflows from atomic tools")
print("=" * 60)

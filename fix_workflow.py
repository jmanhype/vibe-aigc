import json

# Load the dump
with open('workflows/wan22_i2v_svi_pro.json') as f:
    data = json.load(f)

# Extract just the workflow part
workflow = data.get('workflow', data)

# Add metadata
workflow['_vibe_metadata'] = {
    'name': 'wan22_i2v_svi_pro',
    'description': 'Wan 2.2 Image-to-Video with SVI Pro LoRAs (working 3090 config)',
    'capabilities': ['image_to_video', 'text_to_video'],
    'tags': ['wan', 'video', 'svi', 'lora', '3090'],
    'version': '1.0'
}

# Save
with open('workflows/wan22_i2v_svi_pro.json', 'w') as f:
    json.dump(workflow, f, indent=2)

print('Fixed workflow format')
print(f'Nodes: {len(workflow.get("nodes", []))}')

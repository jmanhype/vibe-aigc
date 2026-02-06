import urllib.request
import json

url = 'http://192.168.1.143:8188/object_info'
with urllib.request.urlopen(url, timeout=30) as r:
    info = json.loads(r.read())

# Check key LTX nodes
key_nodes = [
    'LTXAVTextEncoderLoader',  # How to load text encoder
    'LTXVBaseSampler',         # Main sampler
    'LTXVConditioning',        # Conditioning
    'CheckpointLoaderSimple',  # Standard loader
]

print('=== KEY LTX NODE SPECS ===')
for node in key_nodes:
    if node in info:
        print(f'\n{node}:')
        print(f'  Outputs: {info[node].get("output", [])}')
        inputs = info[node].get('input', {}).get('required', {})
        for name, spec in inputs.items():
            if isinstance(spec[0], list):
                # Show a few options
                opts = spec[0][:3]
                print(f'  {name}: {opts}...' if len(spec[0]) > 3 else f'  {name}: {spec[0]}')
            else:
                print(f'  {name}: {spec[0]}')
        opt = info[node].get('input', {}).get('optional', {})
        if opt:
            print(f'  Optional: {list(opt.keys())}')

# Check for simpler API nodes
print('\n=== HIGH-LEVEL API NODES ===')
for node in ['LtxvApiTextToVideo', 'WanTextToVideoApi']:
    if node in info:
        print(f'\n{node}:')
        print(f'  Outputs: {info[node].get("output", [])}')
        inputs = info[node].get('input', {}).get('required', {})
        for name, spec in inputs.items():
            if isinstance(spec[0], list):
                print(f'  {name}: [options]')
            else:
                print(f'  {name}: {spec[0]}')

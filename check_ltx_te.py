import urllib.request
import json

url = 'http://192.168.1.143:8188/object_info'
with urllib.request.urlopen(url, timeout=30) as r:
    info = json.loads(r.read())

# Check what text encoders LTXAVTextEncoderLoader accepts
node = info.get('LTXAVTextEncoderLoader', {})
inputs = node.get('input', {}).get('required', {})

print('LTXAVTextEncoderLoader inputs:')
for name, spec in inputs.items():
    print(f'  {name}:')
    if isinstance(spec, list) and len(spec) > 0:
        opts = spec[0]
        if isinstance(opts, dict) and 'options' in opts:
            opts = opts['options']
        if isinstance(opts, list):
            for o in opts[:10]:
                print(f'    - {o}')
            if len(opts) > 10:
                print(f'    ... ({len(opts)} total)')
        else:
            print(f'    {opts}')

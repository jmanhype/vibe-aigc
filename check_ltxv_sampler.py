import urllib.request
import json

url = 'http://192.168.1.143:8188/object_info'
with urllib.request.urlopen(url, timeout=30) as r:
    info = json.loads(r.read())

for node in ['LTXVBaseSampler', 'BasicGuider', 'KSamplerSelect', 'RandomNoise']:
    if node not in info:
        continue
    n = info[node]
    print(f'{node}:')
    print(f'  Output: {n.get("output", [])}')
    print('  Required:')
    for name, spec in n.get('input', {}).get('required', {}).items():
        print(f'    {name}: {spec[0]}')
    print()

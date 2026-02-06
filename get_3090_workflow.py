"""Get the full workflow from 3090."""

import urllib.request
import json

url = 'http://192.168.1.143:8188'

try:
    with urllib.request.urlopen(f'{url}/claude-code/workflow', timeout=15) as r:
        data = json.loads(r.read().decode())
        
    workflow = data.get('workflow', {})
    workflow_api = data.get('workflow_api', {})
    
    print("=== 3090 Workflow Analysis ===\n")
    
    # Get all node types
    nodes = workflow.get('nodes', [])
    node_types = {}
    for node in nodes:
        ntype = node.get('type', 'Unknown')
        if ntype not in node_types:
            node_types[ntype] = []
        node_types[ntype].append(node.get('id'))
    
    print(f"Total Nodes: {len(nodes)}")
    print(f"Unique Types: {len(node_types)}")
    print("\nNode Types (count):")
    for ntype, ids in sorted(node_types.items(), key=lambda x: -len(x[1])):
        print(f"  {ntype}: {len(ids)}")
    
    # Look for key nodes
    print("\n=== Key Nodes ===")
    key_patterns = ['sampler', 'model', 'vae', 'clip', 'loader', 'wan', 'ltx', 'video']
    
    for node in nodes:
        ntype = node.get('type', '').lower()
        title = node.get('title', '').lower()
        
        if any(p in ntype or p in title for p in key_patterns):
            widgets = node.get('widgets_values', [])
            widget_str = str(widgets)[:100] if widgets else ''
            print(f"  {node.get('id')}: {node.get('type')}")
            if node.get('title'):
                print(f"      title: {node.get('title')}")
            if widget_str:
                print(f"      values: {widget_str}")
    
    # Save full workflow for analysis
    with open('3090_workflow_dump.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\nFull workflow saved to 3090_workflow_dump.json")
    
except Exception as e:
    print(f"Error: {e}")

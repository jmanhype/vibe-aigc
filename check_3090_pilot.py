"""Check Comfy-Pilot on 3090."""

import urllib.request
import json

url = 'http://192.168.1.143:8188'

print("=== Comfy-Pilot Status on 3090 ===\n")

# 1. MCP Status
try:
    with urllib.request.urlopen(f'{url}/claude-code/mcp-status', timeout=10) as r:
        data = json.loads(r.read().decode())
        print(f"MCP Status: {data}")
except Exception as e:
    print(f"MCP Status Error: {e}")

print()

# 2. Current Workflow
try:
    with urllib.request.urlopen(f'{url}/claude-code/workflow', timeout=10) as r:
        data = json.loads(r.read().decode())
        workflow = data.get('workflow', {})
        nodes = workflow.get('nodes', [])
        
        if nodes:
            print(f"Current Workflow: {len(nodes)} nodes")
            print("\nNodes:")
            for node in nodes[:15]:
                nid = node.get('id')
                ntype = node.get('type')
                title = node.get('title', '')
                print(f"  {nid}: {ntype}" + (f" ({title})" if title else ""))
            if len(nodes) > 15:
                print(f"  ... and {len(nodes) - 15} more")
                
            # Show links/connections
            links = workflow.get('links', [])
            if links:
                print(f"\nConnections: {len(links)}")
        else:
            print("No workflow loaded in browser")
            print("(Someone needs to open ComfyUI in a browser on the 3090)")
except Exception as e:
    print(f"Workflow Error: {e}")

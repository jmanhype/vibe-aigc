import json

with open('3090_workflow_dump.json') as f:
    data = json.load(f)

workflow = data.get('workflow', {})
nodes = workflow.get('nodes', [])
links = workflow.get('links', [])

# Build link map
link_map = {}  # link_id -> (from_node, from_slot, to_node, to_slot, type)
for link in links:
    if len(link) >= 6:
        link_id, from_node, from_slot, to_node, to_slot, link_type = link[:6]
        link_map[link_id] = (from_node, from_slot, to_node, to_slot, link_type)

print("=== Sampling/Conditioning Nodes ===\n")

# Look for key node types
key_types = ['sampler', 'condition', 'guider', 'cfg', 'encode', 'noise', 'scheduler', 'sigmas']

for node in nodes:
    ntype = node.get('type', '')
    title = node.get('title', '')
    
    if any(x in ntype.lower() for x in key_types):
        print(f"{node.get('id')}: {ntype}" + (f" ({title})" if title else ""))
        
        # Show widget values
        if node.get('widgets_values'):
            vals = node.get('widgets_values')
            print(f"   values: {str(vals)[:100]}")
        
        # Show inputs with their connections
        inputs = node.get('inputs', [])
        for inp in inputs:
            link_id = inp.get('link')
            inp_name = inp.get('name', '?')
            if link_id and link_id in link_map:
                from_node, from_slot, _, _, _ = link_map[link_id]
                print(f"   input {inp_name} <- node {from_node} slot {from_slot}")

print("\n=== Custom Group Nodes ===\n")

# Look for the UUID-named nodes (custom group nodes)
for node in nodes:
    ntype = node.get('type', '')
    if '-' in ntype and len(ntype) > 30:  # UUID-like
        print(f"{node.get('id')}: {ntype[:20]}... (custom group)")
        if node.get('widgets_values'):
            print(f"   values: {str(node.get('widgets_values'))[:200]}")

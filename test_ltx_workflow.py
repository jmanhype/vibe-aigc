"""Test the corrected LTX-2 workflow pattern."""

import asyncio
import aiohttp
from vibe_aigc.workflow_executor import WorkflowExecutor, Capability
from vibe_aigc.workflow_strategies import VideoRequest

async def test():
    executor = WorkflowExecutor('http://192.168.1.143:8188')
    await executor.discover()
    
    strategy = executor.select_strategy(Capability.TEXT_TO_VIDEO, 'quality')
    print(f'Strategy: {strategy.name}')
    print(f'Models:')
    for role, model in executor.selected_models.items():
        print(f'  {role}: {model.filename}')
    
    request = VideoRequest(
        prompt='cyberpunk samurai walking through neon tokyo rain, cinematic',
        frames=25,
        width=512,
        height=320
    )
    
    workflow = strategy.build_workflow(request, executor.selected_models)
    
    print(f'\nWorkflow ({len(workflow)} nodes):')
    for nid, node in sorted(workflow.items(), key=lambda x: int(x[0])):
        print(f'  {nid}: {node["class_type"]}')
    
    # Try to queue it
    print('\nSubmitting to ComfyUI...')
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://192.168.1.143:8188/prompt',
            json={'prompt': workflow}
        ) as resp:
            result = await resp.json()
            
            if 'error' in result:
                print(f'\nVALIDATION ERROR: {result["error"]["message"]}')
                if 'node_errors' in result:
                    for nid, err in result['node_errors'].items():
                        print(f'  Node {nid}: {err["errors"][0]["message"]}')
                        if 'details' in err['errors'][0]:
                            print(f'    Details: {err["errors"][0]["details"][:200]}')
            else:
                print(f'\nVALIDATED OK!')
                print(f'Prompt ID: {result["prompt_id"]}')
                print('Workflow is valid and queued.')
                print('\nWaiting for generation (interrupt with Ctrl+C)...')
                
                prompt_id = result['prompt_id']
                for i in range(180):
                    await asyncio.sleep(2)
                    async with session.get(f'http://192.168.1.143:8188/history/{prompt_id}') as resp:
                        hist = await resp.json()
                        if prompt_id in hist:
                            status = hist[prompt_id].get('status', {})
                            if status.get('completed'):
                                outputs = hist[prompt_id].get('outputs', {})
                                for nid, out in outputs.items():
                                    if 'images' in out:
                                        fn = out['images'][0]['filename']
                                        print(f'\nSUCCESS!')
                                        print(f'Output: http://192.168.1.143:8188/view?filename={fn}')
                                        return
                            if status.get('status_str') == 'error':
                                msgs = status.get('messages', [])
                                for m in msgs:
                                    if m[0] == 'execution_error':
                                        print(f'\nEXECUTION ERROR: {m[1].get("exception_message", "Unknown")[:500]}')
                                return
                    if i % 15 == 0 and i > 0:
                        print(f'  ... {i*2}s')

if __name__ == '__main__':
    asyncio.run(test())

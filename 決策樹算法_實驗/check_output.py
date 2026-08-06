import json

nb = json.load(open('c:/Users/User/Desktop/機器學習/決策樹算法_實驗/test_stepbystep_output.ipynb', encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    exec_count = c.get('execution_count')
    outputs = c.get('outputs', [])
    print(f"Cell {i}: execution_count={exec_count}")
    for o in outputs:
        if 'text' in o:
            print(f"  Output: {''.join(o['text']).strip()}")
        elif 'ename' in o:
            print(f"  Error: {o['ename']}: {o['evalue']}")
        else:
            print(f"  Output type: {o.get('output_type', '?')}")

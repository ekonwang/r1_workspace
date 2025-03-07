import json
import sys
from utils import load_jsonl

def analyze_results(jsonl_path):
    # Load the jsonl file
    results = load_jsonl(jsonl_path)
    
    # Calculate metrics
    tot_acc = {'bon': 0, 'reward': 0}
    tot_eval = len(results)
    
    for item in results:
        tot_acc['bon'] += item['bon_reward']
        tot_acc['reward'] += item['reward']
    
    # Format the result string
    result = f'Evaluating: BoN@3 {tot_acc["bon"] / tot_eval * 100:.2f} ({tot_acc["bon"]:d}/{tot_eval:d}) | Pass@1 {tot_acc["reward"] / tot_eval * 100:.2f} ({tot_acc["reward"]:d}/{tot_eval:d})'
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_jsonl_file>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    result = analyze_results(jsonl_path)
    print(result)
    with open(jsonl_path.replace('.jsonl', '.log'), 'w') as f:
        f.write(result)

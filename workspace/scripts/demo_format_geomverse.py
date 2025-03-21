import utils

DATA = '.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/data.jsonl'
OUTPUT_DATA = '.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/grpo_conversations.jsonl'
jsonl_data = utils.load_jsonl(DATA)
QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\n\n\n"\
        "Here is the tikz code for the geometry problem:```\n{tikz}\n```"

def format_data(data):
    tikz_code = data['tikz']
    question = data['question']
    user_prompt = QUESTION_TEMPLATE.format(Question=question, tikz=tikz_code)

    answer = data['label']
    conversations = [
        {
            'role': 'user',
            'content': user_prompt
        }
    ]
    return {
        'conversations': conversations,
        'answer': answer
    }

new_data = []
for data in utils.mk_pbar(jsonl_data):
    formatted_data = format_data(data)
    new_data.append(formatted_data)

utils.save_jsonl(new_data, OUTPUT_DATA)

    
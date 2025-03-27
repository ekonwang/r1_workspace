import utils
import argparse

DATA = '.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/data.jsonl'
OUTPUT_DATA = '.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/grpo_conversations.jsonl'
QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\n\n\n"\
        "Here is the tikz code for the geometry problem:```\n{tikz}\n```"
EDIT_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\n\n\n"\
        "Here is the tikz code for the geometry problem:```\n{tikz}\n```"\
        "Optionally, consider to edit the tikz code to construct auxiliary lines in the thinking process, which should be marked with <auxiliary> </auxiliary> tags."

def format_data(data, edit=False):
    tikz_code = data['tikz']
    question = data['question']
    if edit:
        user_prompt = EDIT_TEMPLATE.format(Question=question, tikz=tikz_code)
    else:
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
        'prompt': user_prompt,
        'answer': answer,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DATA)
    parser.add_argument('--output', type=str, default=OUTPUT_DATA)
    parser.add_argument('--edit', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    new_data = []
    jsonl_data = utils.load_jsonl(args.data)
    for data in utils.mk_pbar(jsonl_data):
        formatted_data = format_data(data, edit=args.edit)
        new_data.append(formatted_data)

    utils.save_jsonl(new_data, args.output)

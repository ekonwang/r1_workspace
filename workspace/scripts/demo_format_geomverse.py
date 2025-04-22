import utils
import argparse
import re

DATA = '.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/data.jsonl'
OUTPUT_DATA = '.temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/grpo_conversations.jsonl'
QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\n\n\n"\
        "Here is the tikz code for the geometry problem:```\n{tikz}\n```"
EDIT_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\n\n\n"\
        "Here is the tikz code for the geometry problem:```\n{tikz}\n```"\
        "Optionally, consider to edit the tikz code to construct auxiliary lines in the thinking process, which should be marked with <auxiliary> </auxiliary> tags."


def _mask_coordinates(tikz_code):
    """
    将 TikZ 代码中的所有具体坐标数值替换为 [MASK_VALUE]。

    参数:
        tikz_code (str): 输入的 TikZ 代码字符串。

    返回:
        str: 修改后的 TikZ 代码字符串。
    """
    # 匹配坐标的正则表达式
    coordinate_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'

    # 匹配 radius 的正则表达式
    radius_pattern = r'radius\s*=\s*(-?\d+\.?\d*)'

    # 替换函数，将匹配到的坐标或 radius 替换为 [MASK_VALUE]
    def replace_with_mask(match):
        return "([MASK_VALUE], [MASK_VALUE])" if match.group(1) else f"radius=[MASK_VALUE]"

    # 使用 re.sub 替换所有匹配的坐标
    masked_code = re.sub(coordinate_pattern, "([MASK_VALUE], [MASK_VALUE])", tikz_code)

    # 使用 re.sub 替换所有匹配的 radius
    masked_code = re.sub(radius_pattern, "radius=[MASK_VALUE]", masked_code)

    return masked_code


def format_data(data, edit=False):
    tikz_code = data['tikz']
    question = data['question']
    tikz_code = _mask_coordinates(tikz_code)
    if edit:
        user_prompt = EDIT_TEMPLATE.format(Question=question, tikz=tikz_code)
    else:
        user_prompt = QUESTION_TEMPLATE.format(Question=question, tikz=tikz_code)

    answer = data['label']
    cot = data['cot']
    full_response = f'<think> {cot} </think>\n<answer> {answer} </answer>'

    conversations = [
        {
            'role': 'user',
            'content': user_prompt
        },
        {
            'role': 'assistant',
            'content': full_response,
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

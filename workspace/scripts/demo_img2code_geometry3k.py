import base64
import requests
import sys
import os
from openai import OpenAI
import json
import shutil

from utils_chat import chat_gpt4o, Parser
from utils_execution import CodeExecutor

from PIL import Image
import sys 
import os
from utils_geometry import *
from utils import load_jsonl, mk_pbar, save_jsonl, print_error
from datasets import load_dataset
from utils_inference import VoteModel, ShuffleVoteModel

DEBUG_MODE = False

CODE_FORMAT = """
You code need to incorporate the following sections:

1. Import the necessary packages

The code need to import the following packages:
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon, Rectangle, Wedge
from matplotlib.path import Path
import matplotlib.patches as patches

# import tools
from utils_geometry import *
```

2. Point Definition

The points are defined like this:
```python
points = {
    'A': (.., ..),
    'B': (.., ..),
    'C': (.., ..),
    ...
}
```

3. Draw the figure

You need to draw the figure according to the points and the geometry diagram. There are tools for you to draw the figure, which has been imported already. You need to call the tools to draw the figure.

You can call the tools like this:
```python
draw_angle_sector(ax, points['A'], points['B'], points['C'], label='xx°')
draw_right_angle(ax, points['A'], points['B'], points['C'])
draw_lines(ax, points['A'], points['B'])
draw_pentagon(ax, [points['A'], points['B'], points['C'], points['D'], points['E']])
...
```

4. Annotate the points

(1) You need to annotate the points in the figure like this, not use for or while loop. 
(2) Also you should only annotate the points that are defined in the tikz code.
```python
ax.text(points['A'][0], points['A'][1] - 1, 'A', fontsize=12, ha='center')
ax.text(points['B'][0], points['B'][1] - 1, 'B', fontsize=12, ha='center')
ax.text(points['C'][0], points['C'][1] - 1, 'C', fontsize=12, ha='center')
...
```

5. Figure Format and Figure Show

You need to incorporate the following format:

```python
# Set figure format
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.show()
```

"""





def convert_jpeg_to_png(input_image_path, target_dir=None):
    if not (input_image_path.endswith('.jpeg') or input_image_path.endswith('.jpg')):
        return input_image_path
    jpeg_image = Image.open(input_image_path)

    target_dir = os.path.dirname(input_image_path) if target_dir is None else target_dir
    image_name = os.path.basename(input_image_path)
    # Save as PNG
    output_image_path = os.path.join(target_dir, image_name.replace('.jpeg', '.png'))
    jpeg_image.save(output_image_path)
    return output_image_path


def __exec_prompt(prompt, parser, executor, max_num_retry=3, temperature=0.8):
    content, messages = chat_gpt4o(prompt, temperature=temperature)
    if DEBUG_MODE:
        print_error('[DEBUG] ' + content.replace('\n', '\\n'))

    for i in range(max_num_retry):
        result = parser.parse(content)
        if not result['status']:
            error_msg = result['message']
        else:
            exec_result = executor.execute(result['content'])
            if exec_result[0] == 0:
                return {
                    "img_str": exec_result[1],
                    "code": result['content']
                }
            else:
                error_msg = exec_result[1]
        
        # ERROR Branch
        prompt = f"""
        OBSERVATION: Parsing or Execution error. Error message:
        {error_msg}
        Please fix the error and generate the fixed code.
        """
        content, messages = chat_gpt4o(prompt, messages, temperature=temperature)
        if DEBUG_MODE:
            print_error('[DEBUG] ' + content.replace('\n', '\\n') + '\n\n')
    
    return None


def process_data(image_path, point_positions = None, max_num_retry=3):
    # try to convert 

    _refine_already = False
    pcode = None

    working_dir = os.path.dirname(image_path)
    image_path = convert_jpeg_to_png(image_path)
    parser = Parser()
    executor = CodeExecutor(working_dir)
    image_url = f"<img src='{image_path}'>"

    with open(f'{os.path.dirname(__file__)}/utils_geometry.py', 'r') as f:
        tool_code = f.read()
    tool_code = [c for c in tool_code.split('\n') if not (c.startswith('import ') or c.startswith('from '))]
    tool_code = '\n'.join(tool_code)

    TOOLS = f"""
```
Here are the tools for you to draw the figure, you can only call the following tools to draw the figure.

Please strictly follow the description of the tool when you call it.

{tool_code}
```
"""

    INIT_PROMPT = f"""
Your task is to generate the corresponding python code with matplotlib for the given image and its tikz code.
## Requirements ##

** Here are code format and figure format requirements: **
{CODE_FORMAT}

** Here are some tools for you to Draw the figure (step 3): **
{TOOLS}

** INPUT **: ## geometry diagram ##
{image_url}

<point_positions>

Please generate the Python code according to the requirements strictly and the ## geometry diagram ##.
"""
    
    if point_positions:
        POINT_PROMPT = f"""
** INPUT **: ## point positions ##
{point_positions}
"""
    else:
        POINT_PROMPT = ""
    INIT_PROMPT = INIT_PROMPT.replace('<point_positions>', POINT_PROMPT)

    REFINE_PROMPT = """
Here is a code of the geometry diagram for you to refine. Let's analyze the geometry diagram, find the possible errors in the code and adjust the python code.

Here is the code:
{code}

Here is the geometry diagram:
{image_url}

Here is the generated geometry:
{generated_geometry}

Please first analyze the difference between the target geometry and the generated geometry deeply, then generate the refined Python code. 
"""

    VOTE_PROMPT = """
Here is the # Target Geometry #
{target_geometry}

Here are some generated geometries, your task is to choose the most similar one to the target geometry.

{generated_geometries}

You must first analyze the target geometry and the generated geometries, then choose the index of the most similar geometry from the list, in the <index> </index> tags.
"""


    code_list = []
    img_str_list = []
    SAMPLE_SIZE = 3

    for idx in mk_pbar(range(SAMPLE_SIZE), desc="Generating"):
        result = __exec_prompt(INIT_PROMPT, parser, executor, max_num_retry=3, temperature=0.8)
        if result:
            code_list.append(result['code'])
            img_str_list.append(result['img_str'])

            refine_prompt = REFINE_PROMPT.format(code=result['code'], image_url=image_url, generated_geometry=result['img_str'])
            result = __exec_prompt(refine_prompt, parser, executor, max_num_retry=3, temperature=0.8)
            if result:
                code_list.append(result['code'])
                img_str_list.append(result['img_str'])
            else:
                print_error(f"Failed to refine code for {idx}")
                continue
        
        else:
            print_error(f"Failed to generate code for {idx}")
            continue
    
    generated_geometries = ""
    for idx, img_str in enumerate(img_str_list):
        generated_geometries += f"""
{idx}. <img src='{img_str}'>
"""
    # vote_prompt = VOTE_PROMPT.format(target_geometry=image_url, generated_geometries=generated_geometries)
    # vote_model = VoteModel(vote_prompt=vote_prompt)
    
    vote_model = ShuffleVoteModel(vote_prompt=VOTE_PROMPT)
    vote_result = vote_model.vote(vote_times=5, inputs={
        'img_str_list': img_str_list,
        'target_geometry': image_url
    })

    # select the code with the most similar geometry
    pcode = code_list[vote_result]
    img_str = img_str_list[vote_result] 

    # optimal img path
    img_path = img_str.split("'", 1)[1].split("'", 1)[0]
    # 复制为 selected_image.png
    shutil.copy(img_path, os.path.join(working_dir, f'selected_image_{vote_result}.png'))

    return pcode


def __check_done(task_path):
    # 如果 task_path 下有selected_image*.png，则认为已经处理过
    if os.path.isdir(task_path):
        for file in os.listdir(task_path):
            if file.startswith('selected_image'):
                return True
    return False


def __get_already_processed_idxs(processed_dataset_dict):
    already_processed_idx = []
    for idx, data in processed_dataset_dict.items():
        if data['code']:
            already_processed_idx.append(idx)
    return already_processed_idx


if __name__ == "__main__":
    dataset = load_jsonl('.temp/datasets/intergpt_geometry3k/conversations_map.jsonl')
    OUTPUT_DIR = '.temp/datasets/gps-geometry3k'
    RESULT_FILE = os.path.join(OUTPUT_DIR, 'processed_dataset.json')
    RESULT_JSONL = os.path.join(OUTPUT_DIR, 'processed_dataset.jsonl')
    MAX_PROCESS = 800


    # processed_dataset = load_jsonl(dataset)

    processed_dataset_dict = {idx: d for idx, d in enumerate(dataset)}
    if os.path.exists(RESULT_FILE):
        processed_dataset_dict = json.load(open(RESULT_FILE))
    processed_dataset = list(processed_dataset_dict.values())


    for idx, _ in enumerate(mk_pbar(range(len(processed_dataset_dict)), desc="Processing Whole Dataset")):
        # data = dataset[idx]
        buffer = sys.stdout
        data = processed_dataset_dict[idx].copy()

        task_path = os.path.join(OUTPUT_DIR, f'{idx:04d}')
        if __check_done(task_path):
            print_error(f"Already processed {idx}")
            continue
        shutil.rmtree(task_path, ignore_errors=True)
        os.makedirs(task_path, exist_ok=True)
        try:
            image_path = data['subdir_path'] + '/img_diagram.png'
            point_positions = data['point_positions']
            # question = data['question']
            # answer = data['final_answer'][0]
            # image = data['images'][0]
            # choices = data['choices']
            # assert len(choices) == 4, f"choices should be 4, but got {len(choices)}"

            # right_form_answer = ['A', 'B', 'C', 'D'][choices.index(str(answer))]

            # image_path = os.path.join(task_path, 'image.png')
            # import pdb; pdb.set_trace()
            # save image into task_path + '/image.png'
            # image.save(image_path)
            
            pcode = process_data(image_path, point_positions)
            sys.stdout = buffer
            # if pcode:
            #     print(pcode.replace('\n', '\\n') + '\n\n' + '=' * 50 + '\n\n')
            # else:
            #     continue
            new_data = {
                **data,
                'code': pcode
            }
            processed_dataset_dict[idx] = new_data
            processed_dataset = list(processed_dataset_dict.values())
            
        except Exception as e:
            if DEBUG_MODE:
                raise e
            else:
                sys.stdout = buffer
                shutil.rmtree(task_path, ignore_errors=True)
                print(e)
                print(f"Failed to generate code for {idx}")
                if 'key' in str(e) or 'BdbQuit' in str(e):
                    raise e
            continue

        json.dump(processed_dataset_dict, open(RESULT_FILE, 'w'), indent=4, ensure_ascii=False)
        save_jsonl(processed_dataset, RESULT_JSONL)

        if len(__get_already_processed_idxs(processed_dataset_dict)) >= MAX_PROCESS:
            print_error(f"Already processed {len(__get_already_processed_idxs(processed_dataset_dict))} / {len(processed_dataset_dict)}")
            break
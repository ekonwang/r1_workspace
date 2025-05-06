import os
import json
import os
import sys
import re
from datasets import load_dataset
from datetime import datetime
from utils import load_jsonl, save_jsonl, print_error, mk_len_pbar
from time import sleep
from rich.progress import track, Progress
from utils import llm_config, chat_vlm
import argparse

import torch
from typing import List, Dict, Union, Optional, Any, Callable
from vllm import LLM, SamplingParams
from PIL import Image
import base64
from io import BytesIO
from utils_data import load_custom_dataset, load_geometry3k_dataset, load_mmlu_dataset
from utils_data import *
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import code_to_image, mk_pbar

# --- v5: append the mathvista and olympiadbench benchmarks --- #
# --- v6: change the maxlen params for pass@1 --- #
# --- v7: run 3 times and then average --- #

MAXLEN = 4096

class VLMEval:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = MAXLEN * 2,
        dtype: str = "bfloat16",
        vl_model: bool = True,
        **kwargs
    ):
        """
        Initialize a VLM evaluator with vllm backend.
        
        Args:
            model_name: Name of the Qwen VL model to load
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length for the model
            dtype: Data type for model weights (float16, bfloat16, or float32)
            **kwargs: Additional arguments to pass to vllm.LLM
        """
        self.model_name = model_name
        
        
        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=MAXLEN
        )
        self.vl_model = vl_model

        if self.vl_model:
            self.processor = AutoProcessor.from_pretrained(model_name)
            # Initialize vllm model with tensor parallelism
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                limit_mm_per_prompt={"image": 10, "video": 10},
                max_model_len=max_model_len,
                dtype=dtype,
                **kwargs
            )
        else:
            # Initialize vllm model with tensor parallelism
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                **kwargs
            )
            self.processor = AutoTokenizer.from_pretrained(model_name)

    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert an image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Process the message list to handle image inputs.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Processed messages ready for the model
        """
        processed_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, str):
                # Simple text message
                processed_messages.append({"role": role, "content": content})
            else:
                # Mixed content (text and images)
                processed_content = []
                
                for item in content:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        # Handle image - could be a path, base64 string, or PIL Image
                        if "image_path" in item:
                            image_data = self._encode_image_to_base64(item["image_path"])
                            processed_content.append({
                                "type": "image", 
                                "image_url": f"data:image/jpeg;base64,{image_data}",
                                "min_pixels": 224 * 224,
                                "max_pixels": 1280 * 28 * 28,
                            })
                        elif "image" in item and isinstance(item["image"], Image.Image):
                            # Handle PIL Image
                            buffered = BytesIO()
                            item["image"].save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            processed_content.append({
                                "type": "image", 
                                "image_url": f"data:image/jpeg;base64,{img_str}",
                                "min_pixels": 224 * 224,
                                "max_pixels": 1280 * 28 * 28,
                            })
                        else:
                            raise ValueError('No image detected')
                
                processed_messages.append({"role": role, "content": processed_content})
    
        vllm_input = self.processor.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True,
        )
        if self.vl_model:
            image_inputs, video_inputs, video_kwargs = process_vision_info(processed_messages, return_video_kwargs=True)
            mm_data = {}
            mm_data["image"] = image_inputs
            llm_inputs = {
                "prompt": vllm_input,
                "multi_modal_data": mm_data,
                # # FPS will be returned in video_kwargs
                # "mm_processor_kwargs": video_kwargs,
            }
            return llm_inputs
        else:
            return vllm_input

    

    def chat_vlm(
        self, 
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None
    ) -> tuple:
        """
        Generate a response for the given messages.
        
        Args:
            messages: List of message dictionaries with role and content
            sampling_params: Optional sampling parameters for generation
            
        Returns:
            Tuple of (response_text, full_message_history)
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        

        prompts = [self._prepare_messages(prompt) for prompt in prompts]

        # Generate response using vllm
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
        )
        
        # Extract the generated text
        # response_text = outputs[0].outputs[0].text
        response_texts = [output.outputs[0].text for output in outputs]
        
        return response_texts


# Initialize the VLM evaluator
vlm_evaluator = None


def _generate_model_name(model_name: str):
    model_name = model_name.split("/")[-2:]
    return '-'.join(model_name)


def _eval_olympiadbench_(example: dict):
    NUMERICAL_QUESTION = "{problem}  Output the thinking process in <think> </think> and final answer (the number) in <answer> </answer> tags."\
        "Here is the diagram for the geometry problem."
    
    def make_conversation_image(example):
        problem = example["question"]
        prompt = NUMERICAL_QUESTION.format(problem=problem)
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt  
                        },
                        {
                            "type": "image",
                            "image": code_to_image(example["code"])
                        }
                    ]
                },
            ],
        }

    def cal_reward(content, sol):
        reward = 0.0
        # Try symbolic verification first
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip() # if in <answer> </answer> tags, extract the answer, else just strip.
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            if float(verify(parse(student_answer, extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()]), parse(sol))) > 0:
                # verify success
                print_error(f"Verify success: {student_answer} == {sol}")
                reward = 1.0

            if ground_truth == student_answer:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        return reward
    
    example['prompt'] = make_conversation_image(example)['prompt']
    return example, cal_reward
        

def _eval_mathvista_(example: dict):
    MULTI_CHOICE_QUESTION = "{problem}  Output the thinking process in <think> </think> and final answer (the option letter) in <answer> </answer> tags."\
        "Here is the diagram for the geometry problem."
    
    NUMERICAL_QUESTION = "{problem}  Output the thinking process in <think> </think> and final answer (the number) in <answer> </answer> tags."\
        "Here is the diagram for the geometry problem."

    def make_conversation_image(example):

        if example["question_type"] == "multi_choice":
            problem = f"""
{example["question"]}

A. {example["choices"][0]}
B. {example["choices"][1]}
C. {example["choices"][2]}
D. {example["choices"][3]}
            """
            prompt = MULTI_CHOICE_QUESTION.format(problem=problem)
        else:
            problem = example["question"]
            prompt = NUMERICAL_QUESTION.format(problem=problem)

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "image": code_to_image(example["code"])
                        }
                    ]
                },
            ],
        }

    def cal_reward(content, sol):
        reward = 0.0
        # Try symbolic verification first
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            # TODO: replace content.strip() with '' (unfinished response)
            student_answer = content_match.group(1).strip() if content_match else content.strip() 

            if ground_truth == student_answer:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        return reward

    example['prompt'] = make_conversation_image(example)['prompt']
    # example['solution'] = example['answer'] # solution is already in the example

    return example, cal_reward


def _eval_geometry3k_(example: dict):
    QUESTION_TEMPLATE = "{problem}  Output the thinking process in <think> </think> and final answer (the option letter) in <answer> </answer> tags."\
        "Here is the diagram for the geometry problem."

    def make_conversation_image(example):
        problem = f"""
{example["problem"]}

A. {example["choices"][0]}
B. {example["choices"][1]}
C. {example["choices"][2]}
D. {example["choices"][3]}
"""
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(problem=problem)
                        },
                        {
                            "type": "image",
                            "image": example['image']
                        }
                    ]
                },
            ],
        }

    def cal_reward(content, sol):
        reward = 0.0
        # Try symbolic verification first
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            if ground_truth == student_answer:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        return reward

    example['prompt'] = make_conversation_image(example)['prompt']
    example['solution'] = example['answer']

    return example, cal_reward

    # BON evaluation
    # bon_replies = []
    # bon_reward = 0.0
    # for i in range(3):
    #     sampling_params = SamplingParams(
    #         temperature=0.8,
    #         top_p=1.0,
    #         max_tokens=1024
    #     )
    #     reply, _ = vlm_evaluator.chat_vlm(example['prompt'], sampling_params)
    #     bon_replies.append(reply)
    #     bon_reward = cal_reward(reply, example['answer'])
    #     if bon_reward == 1.0:
    #         break

    # reply, _ = vlm_evaluator.chat_vlm(example['prompt'])
    # reward = cal_reward(reply, example['answer'])

    # return {'prompt': example["prompt"], "bon_replies": bon_replies, "bon_reward": int(bon_reward), "reward": int(reward), 'reply': reply, 'solution': example['answer']}


def _eval_mmlu_(example: dict):
    QUESTION_TEMPLATE = "{problem}  Output the thinking process in <think> </think> and final answer (the option letter) in <answer> </answer> tags."

    def make_conversation_image(example):
        problem = f"""
{example["question"]}

A. {example["choices"][0]}
B. {example["choices"][1]}
C. {example["choices"][2]}
D. {example["choices"][3]}
"""

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": QUESTION_TEMPLATE.format(problem=problem)
                },
            ],
        }

    def cal_reward(content, sol):
        reward = 0.0
        # Try symbolic verification first
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            if ground_truth == student_answer:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        return reward

    example['prompt'] = make_conversation_image(example)['prompt']
    answer_idx = int(example['answer'])
    solution = f"<answer> {['A', 'B', 'C', 'D'][answer_idx]} </answer>"
    example['solution'] = solution

    return example, cal_reward


def _eval_geomverse_(example: dict):
    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."\
        "Here is the diagram for the geometry problem."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=example["problem"])
                        },
                        {
                            "type": "image",
                            "image": 
                        }
                    ]
                },
            ],
        }

    def cal_reward(content, sol):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                # if student_answer == ground_truth:
                #     reward = 1.0
                if abs(float(student_answer) - float(ground_truth)) < 2e-2 * abs(float(ground_truth)):
                    # relax the tolerance
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        return reward

    example['prompt'] = make_conversation_image(example)['prompt']
    return example, cal_reward

    # # BON evaluation
    # bon_replies = []
    # bon_reward = 0.0
    # for i in range(3):
    #     sampling_params = SamplingParams(
    #         temperature=0.8,
    #         top_p=1.0,
    #         max_tokens=1024
    #     )
    #     reply, _ = vlm_evaluator.chat_vlm(example['prompt'], sampling_params)
    #     bon_replies.append(reply)
    #     bon_reward = cal_reward(reply, example['solution'])
    #     if bon_reward == 1.0:
    #         break

    # reply, _ = vlm_evaluator.chat_vlm(example['prompt'])
    # reward = cal_reward(reply, example['solution'])

    # return {'prompt': example["prompt"], "bon_replies": bon_replies, "bon_reward": int(bon_reward), "reward": int(reward), 'reply': reply, 'solution': example['solution']}

def _eval_aime_(example: dict):
    """
    1. prompt
    2. solution
    """
    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": QUESTION_TEMPLATE.format(Question=example["Question"])
                },
            ],
        }

    def cal_reward(content, sol):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                # if student_answer == ground_truth:
                #     reward = 1.0
                if abs(float(student_answer) - float(ground_truth)) < 1e-2 * abs(float(ground_truth)):
                    # relax the tolerance
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        return reward

    example['prompt'] = make_conversation_image(example)['prompt']
    example['solution'] = f'<answer> {example["Answer"]} </answer>'
    return example, cal_reward


def eval_dataset(dataset, output_path, verbose: bool = False, eval_func: Callable = _eval_geomverse_, bon_num: int = 3):
    tot_acc = {'bon': 0, 'reward': 0}
    tot_eval = len(dataset)
    all_eval_data = []

    bon_prompts = []
    prompts = []
    processed_examples = []
    for element in mk_pbar(dataset, description='Preprocessing'):
        example, cal_reward = eval_func(element)
        processed_examples.append(example)
        prompts.append(example['prompt'])
        for i in range(bon_num):
            bon_prompts.append(example['prompt'])
    
    bon_sampling_params = SamplingParams(
        temperature=0.8,
        top_p=1.0,
        max_tokens=MAXLEN
    )
    
    assert len(bon_prompts) == len(dataset) * bon_num
    bon_replies = vlm_evaluator.chat_vlm(bon_prompts, bon_sampling_params)
    replies = vlm_evaluator.chat_vlm(prompts)
    for i in range(len(dataset)):
        for j in range(bon_num):
            if cal_reward(bon_replies[i * bon_num + j], processed_examples[i]['solution']) == 1.0:
                tot_acc['bon'] += 1
                break
        if cal_reward(replies[i], processed_examples[i]['solution']) == 1.0:
            tot_acc['reward'] += 1
        
        all_eval_data.append({
            'prompt': example["prompt"],
            'solution': processed_examples[i]['solution'],
            "bon_replies": bon_replies[i * bon_num: (i + 1) * bon_num],
            "reply": replies[i],
        })

    save_jsonl(all_eval_data, output_path)
    results = f'Evaluating: BoN@{bon_num} {tot_acc["bon"] / tot_eval * 100:.2f} ({tot_acc["bon"]:d}/{tot_eval:d}) | Pass@1 {tot_acc["reward"] / tot_eval * 100:.2f} ({tot_acc["reward"]:d}/{tot_eval:d})'
    result_dict = {
        f"BoN@{bon_num}": tot_acc["bon"] / tot_eval * 100,
        f"Pass@1": tot_acc["reward"] / tot_eval * 100,
        'text': results,
    }
    # print_error(results)
    print(json.dumps(result_dict, indent=4, ensure_ascii=False))
    return result_dict


def debug_test():
    vlm_evaluator = VLMEval(
        model_name='.temp/models/Qwen_Qwen2.5-VL-3B-Instruct',
        tensor_parallel_size=torch.cuda.device_count(),
        # tensor_parallel_size=2,
        gpu_memory_utilization=0.9
    )

    def pack_image_path(image_path=".temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/TRAIN_MIX_1/images/1.jpeg"):
        test_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the image in detail."
                    },
                    {
                        "type": "image",
                        "image_path": image_path
                    }
                ]
            }
        ]
        return test_input

    def pack_image_obj(image_path=".temp/datasets/GeomVerse/TRAIN/TRAIN_MIX/TRAIN_MIX_1/images/1.jpeg"):
        image = Image.open(image_path)
        test_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe the image in detail."
                    },
                    {
                        "type": "image",
                        "image": image
                    }
                ]
            }
        ]
        return test_input
    
    # results = vlm_evaluator.chat_vlm([pack_image_path()])
    # print(results[0])
    results = vlm_evaluator.chat_vlm([pack_image_obj()])
    print(results[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default='.temp/models/Qwen_Qwen2.5-3B-Instruct')
    parser.add_argument("--output_path", type=str, default='.temp/outputs_v7_vl')
    parser.add_argument("--dataset_path", type=str, default='Geomverse-D2')
    parser.add_argument("--verbose", action='store_true', help='output verbose info for debug.')
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()



def main():
    global vlm_evaluator
    args = parse_args()
    # Qwen-2.5-Instruct evaluation
    MODEL_PATH = args.model_path  # Use the correct path for Qwen-2.5-Instruct
    DATASET_CONFIGS = {
        "Geomverse-D2": {
            "output_path": "IN-GeomVerse/D2",
            "load_path": ".temp/datasets/GeomVerse/TEST/D2/data.jsonl",
        },
        "InterGPS-Geometry3K": {
            "output_path": "IN-Geometry3K",
            "load_path": ".temp/datasets/intergpt_geometry3k",
        },
        "MMLU": {
            "output_path": "OOD-MMLU",
            "load_path": "cais/mmlu",
        },
        "AIME": {
            "output_path": "OOD-AIME",
            "load_path": ".temp/datasets/di-zhang-fdu_AIME_1983_2024",
        },
        "MathVista": {
            "output_path": "OOD-MathVista",
            "load_path": ".temp/datasets/filter-gps-mathvista-geometry-v2/gps-mathvista-geometry-v2.jsonl",
        },
        'OlympiadBench': {
            "output_path": "OOD-OlympiadBench",
            "load_path": ".temp/datasets/filter-gps-olympiad-bench/gps-olympiad-bench.jsonl",
        }
    }

    OUTPUT_PATH = f'{args.output_path}/{DATASET_CONFIGS[args.dataset_path]["output_path"]}/{_generate_model_name(MODEL_PATH)}.jsonl'
    DONE_LOG = OUTPUT_PATH.replace('.jsonl', '.log')
    DONE_JSON = OUTPUT_PATH.replace('.jsonl', '.json')
    if os.path.exists(DONE_LOG):
        print_error(f'{DONE_LOG}, skip inference')
        exit()

    if args.dataset_path == "Geomverse-D2":
        dataset = load_custom_dataset(DATASET_CONFIGS[args.dataset_path]["load_path"], train_split_ratio=1, sample_size=None)
    elif args.dataset_path == "InterGPS-Geometry3K":
        dataset = load_geometry3k_dataset(DATASET_CONFIGS[args.dataset_path]["load_path"], sample_size=None)['test']
    elif args.dataset_path == "MMLU":
        dataset = load_mmlu_dataset(DATASET_CONFIGS[args.dataset_path]["load_path"], sample_size=798)['test']
    elif args.dataset_path == "AIME":
        dataset = load_aime_dataset(DATASET_CONFIGS[args.dataset_path]["load_path"], sample_size=None)['train'] # 933 examples
    elif args.dataset_path == "MathVista":
        dataset = load_jsonl_dataset(DATASET_CONFIGS[args.dataset_path]["load_path"], sample_size=None) # testmini 177 examples, unmask
    elif args.dataset_path == "OlympiadBench":
        dataset = load_jsonl_dataset(DATASET_CONFIGS[args.dataset_path]["load_path"], sample_size=None) # olympiadbench 100 geo problems with single solution
    
    vlm_evaluator = VLMEval(
        model_name=MODEL_PATH,
        tensor_parallel_size=torch.cuda.device_count(),
        # tensor_parallel_size=2,
        gpu_memory_utilization=0.9
    )
    print_error(MODEL_PATH)
    results = {'average': {}, 'max': {}, 'min': {}}

    for epoch in range(args.epochs):
        epoch_output_path = OUTPUT_PATH.replace('.jsonl', f'_{epoch}.jsonl')
        if args.dataset_path == "Geomverse-D2":
            result = eval_dataset(dataset, epoch_output_path, True, _eval_geomverse_)
        elif args.dataset_path == "InterGPS-Geometry3K":
            result = eval_dataset(dataset, epoch_output_path, True, _eval_geometry3k_)
        elif args.dataset_path == "MMLU":
            result = eval_dataset(dataset, epoch_output_path, True, _eval_mmlu_)
        elif args.dataset_path == "AIME":
            result = eval_dataset(dataset, epoch_output_path, True, _eval_aime_)
        elif args.dataset_path == "MathVista":
            result = eval_dataset(dataset, epoch_output_path, True, _eval_mathvista_)
        elif args.dataset_path == "OlympiadBench":
            result = eval_dataset(dataset, epoch_output_path, True, _eval_olympiadbench_)
        
        results[f'epoch_{epoch}'] = result
    
    for epoch, result in results.items():
        for key, value in result.items():
            if not isinstance(value, str):
                if key not in results['average']:
                    results['average'][key] = 0
                    results['max'][key] = 0
                    results['min'][key] = 1e9
                results['average'][key] += value
                results['max'][key] = max(results['max'][key], value)
                results['min'][key] = min(results['min'][key], value)
    
    for key, value in results['average'].items():
        results['average'][key] /= args.epochs
    
    print_error(json.dumps(results, indent=4, ensure_ascii=False))

    with open(DONE_LOG, 'w') as f:
        f.write(f'{MODEL_PATH}\n')
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
    
    with open(DONE_JSON, 'w') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))

    del vlm_evaluator

if __name__ == "__main__":
    main()
    # debug_test()
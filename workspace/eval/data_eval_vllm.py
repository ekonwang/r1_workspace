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

import torch
from typing import List, Dict, Union, Optional, Any
from vllm import LLM, SamplingParams
from PIL import Image
import base64
from io import BytesIO
from utils_data import load_custom_dataset
from math_verify import parse, verify

from transformers import AutoProcessor


class VLMEval:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
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
        
        # Initialize vllm model with tensor parallelism
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            **kwargs
        )
        
        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
    
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
                                "image_url": f"data:image/jpeg;base64,{image_data}"
                            })
                        elif "image_url" in item:
                            processed_content.append({"type": "image", "image_url": item["image_url"]})
                        elif "image" in item and isinstance(item["image"], Image.Image):
                            # Handle PIL Image
                            buffered = BytesIO()
                            item["image"].save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            processed_content.append({
                                "type": "image", 
                                "image_url": f"data:image/jpeg;base64,{img_str}"
                            })
                
                processed_messages.append({"role": role, "content": processed_content})

            vllm_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return vllm_prompt

    
    def chat_vlm(
        self, 
        messages: List[Dict],
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
        
        # Process messages to handle images
        processed_prompt = self._prepare_messages(messages)
        print(processed_prompt)
        
        # Generate response using vllm
        outputs = self.llm.generate(
            processed_prompt,
            sampling_params=sampling_params,
            prompt_token_ids=None,
        )
        
        # Extract the generated text
        response_text = outputs[0].outputs[0].text
        
        # Update message history
        full_message_history = messages + [{"role": "assistant", "content": response_text}]
        
        return response_text, full_message_history


# Initialize the VLM evaluator
vlm_evaluator = None


def _generate_model_name(model_name: str):
    model_name = model_name.split("/")[-2:]
    return '-'.join(model_name)

# reply, messages = chat_gpt4o("Could you please give me a list of all the countries in the world?")
# print(reply)
def eval_question(messages: List[Dict]):
    reply, _ = vlm_evaluator.chat_vlm(messages)
    return reply


def _eval_geomverse_(example: dict):
    QUESTION_TEMPLATE = "<image> {Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
            # "prompt": [
            #     {
            #         "multi_modal_data": {"image": example["image"]},
            #         "prompt": QUESTION_TEMPLATE.format(Question=example["problem"]),
            #     },
            # ],
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
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
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

    # BON evaluation
    bon_replies = []
    bon_reward = 0.0
    for i in range(3):
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=1.0,
            max_tokens=1024
        )
        reply, _ = vlm_evaluator.chat_vlm(example['prompt'], sampling_params)
        bon_replies.append(reply)
        bon_reward = cal_reward(reply, example['solution'])
        if bon_reward == 1.0:
            break

    reply, _ = vlm_evaluator.chat_vlm(example['prompt'])
    reward = cal_reward(reply, example['solution'])

    return {'prompt': example["problem"], "bon_replies": bon_replies, "bon_reward": int(bon_reward), "reward": int(reward), 'reply': reply, 'solution': example['solution']}


def eval_dataset(dataset, output_path, verbose: bool = False):
    tot_acc = {'bon': 0, 'reward': 0}
    tot_eval = 0
    all_eval_data = []

    # pbar = mk_len_pbar(len(dataset), desc='Evaluating')
    with Progress() as progress:
        task_id = progress.add_task("[red]Processing...", total=len(dataset))

        for element_id in range(len(dataset)):
            element = dataset[element_id]

            try:
                eval_dict = _eval_geomverse_(element)
            except Exception as err:
                raise err
                if 'keyboard' in str(err).lower():
                    raise err
                print_error(err)
            else:
                tot_acc['bon'] += eval_dict['bon_reward']
                tot_acc['reward'] += eval_dict['reward']
                tot_eval += 1
                if verbose:
                    print(f"\n" * 3)
                    print("=" * 40)
                    print(f"Question: {eval_dict['prompt']}")
                    print(">" * 20)
                    print(f"Answer: {eval_dict['reply']}")
                    print(">" * 20)
                    print(f"Ground Truth: {eval_dict['solution']}")
                
                all_eval_data.append(eval_dict)
                save_jsonl(all_eval_data, output_path, use_tqdm=False)
            
            # pbar.update(1)
            # pbar.set_description(f'Evaluating: {tot_acc / tot_eval * 100:.2f} ({tot_acc:6d}/{tot_eval:6d})')
            progress.update(task_id, advance=1)
            progress.update(task_id, description=f'Evaluating: BoN@3 {tot_acc["bon"] / tot_eval * 100:.2f} ({tot_acc["bon"]:d}/{tot_eval:d}) | Pass@1 {tot_acc["reward"] / tot_eval * 100:.2f} ({tot_acc["reward"]:d}/{tot_eval:d})')

        return None


def clean_up():
    # Ensure proper cleanup of vLLM resources
    if 'vlm_evaluator' in locals() and vlm_evaluator is not None:
        # Explicitly delete the LLM instance to release resources
        if hasattr(vlm_evaluator, 'llm'):
            del vlm_evaluator.llm
        del vlm_evaluator
    
    # Force garbage collection to clean up any remaining references
    import gc
    gc.collect()

# MODEL_PATH = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/models/Qwen_Qwen2-VL-2B-Instruct'
# vlm_evaluator = VLMEval(
#     model_name=MODEL_PATH,
#     tensor_parallel_size=4,  # Adjust based on available GPUs
#     gpu_memory_utilization=0.9
# )
# dataset = load_custom_dataset('.temp/datasets/GeomVerse/TEST/D1/data.jsonl', train_split_ratio=1)
# eval_dataset(dataset, f'.temp/outputs/GeomVerse/D1/{_generate_model_name(MODEL_PATH)}.jsonl', True)


MODEL_PATH = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/workspace/train/outputs/geo_v1/checkpoint-200'
vlm_evaluator = VLMEval(
    model_name=MODEL_PATH,
    tensor_parallel_size=4,  # Adjust based on available GPUs
    gpu_memory_utilization=0.9
)
dataset = load_custom_dataset('.temp/datasets/GeomVerse/TEST/D1/data.jsonl', train_split_ratio=1)
eval_dataset(dataset, f'.temp/outputs/GeomVerse/D1/{_generate_model_name(MODEL_PATH)}.jsonl', True)

clean_up()

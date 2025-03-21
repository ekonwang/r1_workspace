import torch
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import is_peft_available

from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import generate_model_card

from datetime import datetime


if __name__ == "__main__":

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.\n\n\n"\
        "Here is the tikz code for the geometry problem:```\n{tikz}\n```"
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": QUESTION_TEMPLATE.format(Question=example["problem"], tikz=example["geometry"])
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": example["completion"]
                },
            ],
        }

    tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/r1_workspace/.temp/models/Qwen_Qwen2.5-3B-Instruct")
    example = make_conversation_image({'problem': 'What is the area of the circle?', 'geometry': 'The circle has a radius of 5 cm.', 'completion': 'The area of the circle is 78.54 cm².'})
    results = maybe_apply_chat_template(example, tokenizer)
    print(results)

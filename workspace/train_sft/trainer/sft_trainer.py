# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

if is_peft_available():
    from peft import PeftConfig, get_peft_model


class Qwen2VLSFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) of multimodal models like Qwen2-VL and Aria.
    
    Example:
    
    ```python
    from datasets import load_dataset
    from trl import Qwen2VLSFTTrainer
    
    dataset = load_dataset("your_dataset", split="train")
    
    trainer = Qwen2VLSFTTrainer(
        model="Qwen/Qwen2-VL-7B",
        train_dataset=dataset,
        args=TrainingArguments(output_dir="./output"),
    )
    
    trainer.train()
    ```
    
    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:
            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co
            - A `PreTrainedModel` object
        args (`TrainingArguments`, *optional*):
            The training arguments to use.
        train_dataset (`Union[Dataset, IterableDataset]`, *optional*):
            The dataset to use for training. Must include columns "prompt" and "completion".
        eval_dataset (`Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]`, *optional*):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase`, *optional*):
            The processing class to use. If None, will be loaded from the model's name.
        callbacks (`List[TrainerCallback]`, *optional*):
            A list of callbacks to customize the training loop.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*):
            A tuple containing the optimizer and scheduler to use.
        peft_config (`PeftConfig`, *optional*):
            The PEFT configuration to use.
        max_pixels (`int`, *optional*):
            Maximum number of pixels for image processing.
        min_pixels (`int`, *optional*):
            Minimum number of pixels for image processing.
        attn_implementation (`str`, *optional*):
            The attention implementation to use.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: TrainingArguments = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = TrainingArguments(f"{model_name}-SFT")

        # Model
        model_init_kwargs = {"attn_implementation": attn_implementation}
        if isinstance(model, str):
            model_id = model
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Processing class
        if processing_class is None:
            # Qwen VL series model
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            # language model
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the paper
        
        # Initialize the data collator
        def data_collator(features):
            return features
        
        # Initialize the trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,  # Add this line to use our simple data collator
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    def _set_signature_columns_if_needed(self):
        # Override to ensure we keep the necessary columns
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "completion", "image"]
            # Add any other columns that might be in the dataset
            if self.train_dataset is not None:
                self._signature_columns.extend(
                    [k for k in self.train_dataset.column_names if k not in self._signature_columns]
                )

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for the model.
        """
        # Process the inputs based on whether they contain images or not
        if "image" in inputs and inputs["image"] is not None and any(inputs["image"]):
            # Multimodal inputs with images
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            completions_text = [maybe_apply_chat_template(example, self.processing_class)["completion"] for example in inputs]
            
            # Combine prompts and completions
            texts = [p + c for p, c in zip(prompts_text, completions_text)]
            
            # Process with images
            prompt_inputs = self.processing_class(
                text=texts,
                images=inputs["image"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_length if hasattr(self.args, "max_length") else None,
                return_attention_mask=True,
                add_special_tokens=True,
            )
        else:
            # Text-only inputs
            completions_text = [maybe_apply_chat_template(example, self.processing_class)["completion"] for example in inputs]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            
            # Combine prompts and completions
            texts = [p + c for p, c in zip(prompts_text, completions_text)]
            
            # Process text only
            prompt_inputs = self.processing_class(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_length if hasattr(self.args, "max_length") else None,
                return_attention_mask=True,
                add_special_tokens=True,
            )
        
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        device = self.accelerator.device
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)

        if self.max_prompt_length is not None:
            max_context_length = self.max_prompt_length + self.max_completion_length
            prompt_ids = prompt_ids[:, -max_context_length :]
            prompt_mask = prompt_mask[:, -max_context_length :]
        
        # Decode the generated completions
        context_texts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)

        # Log prompt length if DEBUG_MODE and LOG_LENGTHS are enabled
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            prompt_lengths = prompt_mask.sum(dim=1).tolist()
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Prompt Lengths -------------\n")
                for i, length in enumerate(prompt_lengths):
                    f.write(f"Context {i}: Length = {length}\n")
                    f.write(f"Context texts: {context_texts[i]}\n")

        return {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "labels": prompt_ids,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss.
        """
        # Prepare the inputs
        # prepared_inputs = self._prepare_inputs(inputs) # 在 inner loop 中会自动调用，这里不需要调这个 function
        
        # Forward pass
        outputs = model(**inputs) # 需要传入 Labels，否则 Loss 不会自动计算
        
        # Get the loss
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.
        
        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id if hasattr(self, "hub_model_id") else None,
            dataset_name=dataset_name,
            tags=tags,
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

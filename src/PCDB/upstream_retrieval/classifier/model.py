"""Llama 3.3 70B wrapper for yes/no generation and for the P(yes) confidence score.

Archived code; only the cluster-specific cache directory was removed (set HF_HOME instead).
"""
import os

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class llamaRunner:

    def __init__(self,
                 original_model_name,
                 temperature,
                 max_new_tokens,
                 top_p,
                 access_token,
                 trained_model_path=None,
                 use_quantization=False,
                 use_4bit=False,
                 use_8bit=False,
                 torch_dtype=torch.bfloat16):
        print("Running model : ", original_model_name)
        print("Available GPUs :", torch.cuda.device_count())
        print("Torch data type : ", torch_dtype)
        print("Use quantization (T/F) : ", use_quantization)
        print("Use 4 bit: ", use_4bit)
        print("Use 8 bit: ", use_8bit)

        quantization_config = None
        if use_quantization:
            if not (use_4bit or use_8bit):
                raise ValueError("Must set use_4bit ot use_8bit to enable quantization.")

            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype="bfloat16"
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(original_model_name,
                                                          token=access_token,
                                                          quantization_config=quantization_config,
                                                          torch_dtype=torch_dtype,
                                                          device_map="auto",
                                                          )

        self.tokenizer = AutoTokenizer.from_pretrained(original_model_name,
                                                       token=access_token,
                                                       max_seq_length=4096,
                                                       padding_side="left",
                                                       )
        if trained_model_path:
            print("Loading trained model at : ", trained_model_path)
            self.model = PeftModel.from_pretrained(self.model, trained_model_path)

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def generate_batch(self, batch_prompts):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(batch_prompts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=4096)
        inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                          eos_token_id=self.terminators,
                                          pad_token_id=self.tokenizer.eos_token_id,
                                          max_new_tokens=self.max_new_tokens,
                                          top_p=self.top_p,
                                          temperature=self.temperature)

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [output[len(prompt):] for prompt, output in zip(batch_prompts, decoded_outputs)]

    def generate_batch_with_probabilities(self, batch_prompts, only_logits=False):
        """Confidence score: softmax over the next-token logits of 'yes' and 'no' only."""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(batch_prompts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=4096)
        inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

        yes_id = self.tokenizer.encode('yes', add_special_tokens=False)[0]
        no_id = self.tokenizer.encode('no', add_special_tokens=False)[0]
        yes_logit = logits[:, yes_id]
        no_logit = logits[:, no_id]

        if only_logits:
            return yes_logit.cpu().tolist(), no_logit.cpu().tolist()

        stacked_logits = torch.stack([no_logit, yes_logit], dim=-1)
        probabilities = F.softmax(stacked_logits, dim=1)
        no_probabilities = probabilities[:, 0].cpu().tolist()
        yes_probabilities = probabilities[:, 1].cpu().tolist()
        return yes_probabilities, no_probabilities

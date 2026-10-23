"""Run the zero-shot or fine-tuned Llama 3.3 70B classifier on a TSV of articles (TITLE, ABSTRACT).

Archived code. Kept only the modes used for the reported results; the token is read
from the HF_TOKEN environment variable.
"""
import argparse
import gc
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bc_dataset import BCDataset
from .model import llamaRunner

ORIGINAL_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
TEMPERATURE = 0.001
MAX_NEW_TOKENS = 1
TOP_P = 0.5
ACCESS_TOKEN = os.environ.get("HF_TOKEN")
SAVE_RESULTS_FOLDER = "outputs/pcdb/upstream_retrieval/classifier/predictions"


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_file_path', type=str, required=True)
    parser.add_argument('--action', type=str, required=True,
                        help="run_baseline_fp16 | run_ft_model | run_model_for_probabilities")
    parser.add_argument('--ft_model_path', type=str, default=None)
    parser.add_argument('--save_results_folder', type=str, default=SAVE_RESULTS_FOLDER)
    parser.add_argument('--save_file_name', type=str, required=True)
    return parser.parse_args()


def save_dataframe(df, f_name, save_results_folder=SAVE_RESULTS_FOLDER):
    os.makedirs(save_results_folder, exist_ok=True)
    save_path = os.path.join(save_results_folder, f_name + ".tsv")
    print("Saving file : ", save_path)
    df.to_csv(save_path, sep="\t", index=False)
    return save_path


def run_model_for_probabilities(data,
                                trained_model_path=None,
                                batch_size=4,
                                save_file_name="bc_prob_test",
                                save_results_folder=SAVE_RESULTS_FOLDER,
                                only_logits=False):
    """Add yes_probability / no_probability columns (4-bit model, as used for all 1.6M articles)."""
    model = llamaRunner(original_model_name=ORIGINAL_MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        top_p=TOP_P,
                        access_token=ACCESS_TOKEN,
                        trained_model_path=trained_model_path,
                        use_quantization=True,
                        use_4bit=True)

    input_dataloader = DataLoader(BCDataset(data), batch_size=batch_size, shuffle=False)
    outputs = {'yes': [], 'no': []}
    total = len(input_dataloader)
    for i, batch_prompts in tqdm(enumerate(input_dataloader), total=total):
        yes_prob, no_prob = model.generate_batch_with_probabilities(batch_prompts,
                                                                    only_logits=only_logits)
        outputs['yes'].extend(yes_prob)
        outputs['no'].extend(no_prob)

        # Checkpoint long corpus runs so a killed job can resume.
        if i > 0 and i % 1000 == 0:
            save_data = data[:(i + 1) * batch_size].copy()
            save_data['yes_probability'] = outputs['yes']
            save_data['no_probability'] = outputs['no']
            print(f"Processed {i} out of {total} batches. Save to file.")
            save_dataframe(save_data, save_file_name, save_results_folder)

    torch.cuda.empty_cache()
    gc.collect()

    data['yes_probability'] = outputs['yes']
    data['no_probability'] = outputs['no']
    return save_dataframe(data, save_file_name, save_results_folder)


def run_model(model, data, batch_size, save_file_name,
              save_results_folder=SAVE_RESULTS_FOLDER):
    """Add the generated one-token answer ('yes'/'no') as an `answer` column."""
    input_dataloader = DataLoader(BCDataset(data), batch_size=batch_size, shuffle=False)
    outputs = []
    for batch_prompts in tqdm(input_dataloader, total=len(input_dataloader)):
        outputs.extend(model.generate_batch(batch_prompts))

    torch.cuda.empty_cache()
    gc.collect()
    data["answer"] = outputs
    return save_dataframe(data, save_file_name, save_results_folder)


def run_baseline_fp16(data, batch_size=1,
                      save_results_folder=SAVE_RESULTS_FOLDER,
                      save_file_name="baseline_fp16_test_set"):
    """Zero-shot baseline: the untuned model in bf16."""
    model = llamaRunner(original_model_name=ORIGINAL_MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        top_p=TOP_P,
                        access_token=ACCESS_TOKEN)
    return run_model(model, data, batch_size, save_file_name,
                     save_results_folder=save_results_folder)


def run_ft_model(data,
                 trained_model_path,
                 batch_size=1,
                 save_results_folder=SAVE_RESULTS_FOLDER,
                 save_file_name="ft_test_set"):
    """Fine-tuned model: 4-bit base model + LoRA adapter."""
    print("Running fine tuned model, model path : ", trained_model_path)
    model = llamaRunner(original_model_name=ORIGINAL_MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        top_p=TOP_P,
                        access_token=ACCESS_TOKEN,
                        trained_model_path=trained_model_path,
                        use_quantization=True,
                        use_4bit=True)
    return run_model(model, data, batch_size, save_file_name,
                     save_results_folder=save_results_folder)


def main():
    args = parse_argument()
    df = pd.read_csv(args.data_file_path, sep="\t")
    print("Run model on data : ", args.data_file_path)
    if args.action == "run_baseline_fp16":
        run_baseline_fp16(df, save_results_folder=args.save_results_folder,
                          save_file_name=args.save_file_name)
    elif args.action == 'run_ft_model':
        run_ft_model(df, args.ft_model_path, save_results_folder=args.save_results_folder,
                     save_file_name=args.save_file_name)
    elif args.action == 'run_model_for_probabilities':
        run_model_for_probabilities(df,
                                    trained_model_path=args.ft_model_path,
                                    save_results_folder=args.save_results_folder,
                                    save_file_name=args.save_file_name)
    else:
        raise ValueError(f"Unknown action: {args.action}")


if __name__ == '__main__':
    main()

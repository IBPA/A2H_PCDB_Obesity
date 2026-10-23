"""QLoRA fine-tuning of Llama 3.3 70B with torchtune (4 x A100 80GB in the original runs).

Archived code. The token is read from HF_TOKEN; errors from `tune` now stop the run
instead of being printed and skipped.
"""
import argparse
import os
import subprocess
from glob import glob

os.environ["MKL_THREADING_LAYER"] = "GNU"
FINE_TUNE_LOG_PATH = "outputs/pcdb/upstream_retrieval/classifier/logs"
RECIPE_FOLDER = "outputs/pcdb/upstream_retrieval/classifier/recipes"


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--action', type=str, required=True, help="grid_search_fine_tune | fine_tune_model")
    parser.add_argument('--fold_n', type=str)
    parser.add_argument('--recipe_config', type=str)
    return parser.parse_args()


def log_subprocess(command, stdout_logfile, stderr_logfile, message):
    with open(stdout_logfile, "w") as stdout_log, open(stderr_logfile, "w") as stderr_log:
        stdout_log.write(message)
        stderr_log.write(message)
        process = subprocess.run(command, stdout=stdout_log, stderr=stderr_log, text=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def download_llama_model(log_folder_path=FINE_TUNE_LOG_PATH):
    """Download base weights to torchtune's default location (/tmp/Llama-3.3-70B-Instruct)."""
    os.makedirs(log_folder_path, exist_ok=True)
    download_command = [
        "tune", "download", "meta-llama/Llama-3.3-70B-Instruct",
        "--ignore-patterns", "original/consolidated*",
        "--hf-token", os.environ["HF_TOKEN"],
    ]
    print("Starting model download...")
    log_subprocess(download_command,
                   os.path.join(log_folder_path, "download_stdout.log"),
                   os.path.join(log_folder_path, "download_stderr.log"),
                   "Download weights\n")


def run_finetune_command(recipe_path, log_folder_path):
    os.makedirs(log_folder_path, exist_ok=True)
    finetune_command = [
        "tune", "run", "--nproc_per_node", "4",
        "lora_finetune_distributed", "--config", recipe_path,
    ]
    print("Fine tune : ", recipe_path)
    log_subprocess(finetune_command,
                   os.path.join(log_folder_path, "finetune_stdout.log"),
                   os.path.join(log_folder_path, "finetune_stderr.log"),
                   f"fine tuning for {recipe_path}\n")


def grid_search_fine_tune(fold_n):
    all_recipes = sorted(glob(os.path.join(RECIPE_FOLDER, f"fold_{fold_n}", "*.yaml")))
    print("Working on fold : ", fold_n)
    download_llama_model()
    for curr_recipe in all_recipes:
        param_set_str = os.path.basename(curr_recipe)[:-5]
        run_finetune_command(curr_recipe, os.path.join(FINE_TUNE_LOG_PATH, f"fold_{fold_n}", param_set_str))


def fine_tune_model(recipe_config):
    download_llama_model()
    run_finetune_command(recipe_config, os.path.join(FINE_TUNE_LOG_PATH, "final_model"))


if __name__ == "__main__":
    args = parse_argument()
    if args.action == "grid_search_fine_tune":
        grid_search_fine_tune(args.fold_n)
    elif args.action == "fine_tune_model":
        fine_tune_model(args.recipe_config)
    else:
        raise ValueError(f"Unknown action: {args.action}")

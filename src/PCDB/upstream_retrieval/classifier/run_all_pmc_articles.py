"""Score one chunk of retrieved articles with the final fine-tuned classifier (P(yes) per article).

Archived code; cluster paths are now arguments. Articles with P(yes) >= 0.9 form PCDB
(134,565 of 1,616,149; the lowest kept score is 0.9047).
"""
import argparse
import os

import pandas as pd

from .run_model import run_model_for_probabilities


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--chunk_num', type=str, required=True)
    parser.add_argument('--chunk_dir', type=str, default='outputs/pcdb/upstream_retrieval/retrieval/chunks')
    parser.add_argument('--save_results_folder', type=str, default='outputs/pcdb/upstream_retrieval/classifier/chunk_scores')
    parser.add_argument('--trained_model_path', type=str,
                        default='outputs/pcdb/upstream_retrieval/classifier/model_weights/final/epoch_2')
    return parser.parse_args()


def main(batch_size=4):
    args = parse_argument()
    chunk_name = f"chunk_{args.chunk_num}"
    target_chunk = os.path.join(args.chunk_dir, f"{chunk_name}.tsv")
    if not os.path.exists(target_chunk):
        raise FileNotFoundError(f"Chunk {target_chunk} not found.")

    data = pd.read_csv(target_chunk, sep="\t")
    processed_path = os.path.join(args.save_results_folder, f"{chunk_name}.tsv")
    # Resume a partially scored chunk (jobs hit the cluster time limit).
    if os.path.exists(processed_path):
        processed_data = pd.read_csv(processed_path, sep="\t")
        data = data[~data["pmcid"].isin(processed_data["pmcid"])]
        print("Remaining rows for processing: ", len(data))
        if len(data) == 0:
            return
        chunk_name = f"{chunk_name}_remaining"

    run_model_for_probabilities(data,
                                trained_model_path=args.trained_model_path,
                                batch_size=batch_size,
                                save_file_name=chunk_name,
                                save_results_folder=args.save_results_folder)


if __name__ == '__main__':
    main()

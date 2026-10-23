"""Stage 1c: attach title + abstract to each open-access article and split into chunks for the classifier.

Chunking follows the archived prepare_inference_data.py (50,000 articles per chunk).
The original code that dropped articles without an abstract was not preserved; the
filter below reproduces its effect (1,669,186 -> 1,616,149 articles; every dropped
article we sampled had a title but no abstract in BioC-PMC).
"""
import argparse
import os

import pandas as pd

from .get_pmc_text import get_pmc_text

CHUNK_SIZE = 50000


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--open_access_results', type=str,
                        default='outputs/pcdb/upstream_retrieval/retrieval/bioc_pmc_oa_query_uid_results.tsv')
    parser.add_argument('--bioc_pmc_dir', type=str, required=True)
    parser.add_argument('--chunk_dir', type=str, default='outputs/pcdb/upstream_retrieval/retrieval/chunks')
    return parser.parse_args()


def main():
    args = parse_argument()
    os.makedirs(args.chunk_dir, exist_ok=True)
    articles = pd.read_csv(args.open_access_results, sep="\t", dtype=str, keep_default_na=False)
    articles = articles.drop(columns=["queries"])

    for index, start in enumerate(range(0, len(articles), CHUNK_SIZE)):
        chunk = articles[start:start + CHUNK_SIZE].copy()
        for section_type in ["TITLE", "ABSTRACT"]:
            chunk = get_pmc_text(section_type, chunk, args.bioc_pmc_dir)
        chunk = chunk[(chunk["TITLE"] != "") & (chunk["ABSTRACT"] != "")]
        chunk.to_csv(os.path.join(args.chunk_dir, f"chunk_{index + 1}.tsv"), sep="\t", index=False)


if __name__ == '__main__':
    main()

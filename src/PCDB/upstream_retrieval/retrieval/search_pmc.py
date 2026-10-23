"""Stage 1: search PubMed Central with one query per MeSH disease.

Archived code (run 14 Nov 2024 with NCBI EDirect). Cleaned of credentials and
unused helpers only; the search logic is unchanged.

Requires EDirect (esearch/efetch) on PATH. Set NCBI_API_KEY to raise the rate limit.
"""
import argparse
import os
import subprocess
import time
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DEFAULT_SAVE_EVERY = 100
QUERY_TEMPLATE = '{disease_terms} AND (animal) AND ((rat) OR (rabbit) OR (monkey) OR (dog) OR (mouse) OR (pig) OR (hamster))'


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Search PMC for each MeSH disease query.')

    parser.add_argument(
        '--query',
        type=str,
        default='data/PCDB/upstream_retrieval/retrieval/mesh_disease_queries.tsv',
        help='TSV with columns disease_name and query.',
    )

    parser.add_argument(
        '--save_every',
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help=f'Save to results once every {DEFAULT_SAVE_EVERY} queries.',
    )

    parser.add_argument(
        '--query_uid_results_folderpath',
        type=str,
        default='outputs/pcdb/upstream_retrieval/retrieval/',
        help='Output directory',
    )

    parser.add_argument(
        '--email',
        type=str,
        required=True,
        help='Email passed to NCBI E-utilities.',
    )

    return parser.parse_args()


def check_data(folder_path):
    """Return the index of the last query already saved, so a rerun resumes after it."""
    existing_data_files = glob(os.path.join(folder_path, "query_uid_results_*.tsv"))
    top = -1
    for f_path in existing_data_files:
        upper_end = f_path[:-4].split("query_uid_results_")[-1]
        top = max([top, int(upper_end)])

    return top


def main():
    args = parse_argument()

    Path(args.query_uid_results_folderpath).mkdir(exist_ok=True, parents=True)

    up_to = check_data(args.query_uid_results_folderpath)
    #####################################################
    # search queries and save the query keyword and IDs #
    #####################################################
    print('\nSearching queries and saving the UIDs...')

    queries_df = pd.read_csv(args.query, sep="\t", dtype=str)
    print(queries_df)
    pbar = tqdm(queries_df.iterrows(), total=len(queries_df))

    ids, queries, keywords = [], [], []

    for idx, row in pbar:
        q = row["query"]
        k = row["disease_name"]
        pbar.set_description(f'Processing {k}')

        # skip this query if it was already searched before
        if idx <= up_to:
            print(f"skip {k}")
            continue

        db = "pmc"
        try:
            esearch_args = ('esearch', '-db', db, '-query', QUERY_TEMPLATE.format(disease_terms=q), '-email', args.email)
            esearch = subprocess.Popen(esearch_args, stdout=subprocess.PIPE)
            efetch_args = ('efetch', '-format', 'uid', '-email', args.email)
            result = subprocess.check_output(efetch_args, stdin=esearch.stdout, text=True)
            esearch.wait()
        except Exception as e:
            # Failed queries were logged and re-run afterwards (see process_search_results.py --op select_missing_keywords).
            print(f'Exception occured while processing {q}!')
            print(e)
            continue

        new_ids = [f'PMC{id}' if db == 'pmc' else id for id in result.rstrip().split('\n')]
        ids += new_ids
        queries += [q] * len(new_ids)
        keywords += [k] * len(new_ids)

        if (idx > 0 and idx % args.save_every == 0) or (idx == len(queries_df) - 1):
            df = pd.DataFrame({f'{db}id': ids, 'queries': queries, "keywords": keywords})
            filepath = os.path.join(args.query_uid_results_folderpath,
                                    f"query_uid_results_{idx}.tsv")
            df.to_csv(filepath, sep='\t', index=False)
            ids, queries, keywords = [], [], []

        time.sleep(0.33)


if __name__ == '__main__':
    main()

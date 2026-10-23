"""Stage 1b: merge the raw search hits, de-duplicate articles, keep PMC open-access articles.

Archived code. Paths are now arguments; the open-access ID set is built from the
BioC-PMC file names, as the original notebook (create_bioc_pmc_id_set.ipynb) did.

BioC-PMC is NCBI's BioC-format dump of the PMC open-access subset
(https://ftp.ncbi.nlm.nih.gov/pub/wilbur/BioC-PMC/), one <PMCID>.xml file per article.
"""
import argparse
import os
from glob import glob

import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

KEYWORD_COLUMN_NAME = "disease_name"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--op',
        type=str,
        required=True,
        help="Choose among: select_missing_keywords, select_pmc_articles"
    )
    parser.add_argument('--search_results_dir', type=str, default='outputs/pcdb/upstream_retrieval/retrieval')
    parser.add_argument('--query', type=str, default='data/PCDB/upstream_retrieval/retrieval/mesh_disease_queries.tsv')
    parser.add_argument('--bioc_pmc_dir', type=str, help='Local copy of BioC-PMC (needed for select_pmc_articles).')

    return parser.parse_args()


def select_missing_keywords(complete_df, query_file_path, save_dir):
    """Write the queries that returned no saved results, so they can be searched again."""
    searched_keywords = complete_df["keywords"].unique()

    with ProgressBar():
        searched_keywords = searched_keywords.compute()
        print("Number of searched keywords: ", len(searched_keywords))

    queries_df = pd.read_csv(query_file_path, sep="\t")
    all_keywords = queries_df[KEYWORD_COLUMN_NAME].tolist()
    missing_keywords = set(all_keywords).difference(set(searched_keywords))
    print(f"Found missing keywords {missing_keywords} \n")
    missing_keywords_df = queries_df[queries_df[KEYWORD_COLUMN_NAME].isin(missing_keywords)]
    missing_keywords_df.to_csv(os.path.join(save_dir, "missing_search_keywords.tsv"), sep="\t", index=False)


def load_open_access_pmcids(bioc_pmc_dir):
    print("load BioC PMC ids (open access)...")
    return {os.path.basename(f).replace(".xml", "") for f in glob(os.path.join(bioc_pmc_dir, "*.xml"))}


def select_pmc_articles(complete_df, bioc_pmc_id_set, save_dir):
    """Keep one row per article (the first query that matched it) for open-access articles only.

    Counts from the original run (14 Nov 2024): 222,757,668 hits -> 2,559,425 unique
    articles -> 1,669,186 in the open-access subset.
    """
    print("Select PMC OA articles...")
    with ProgressBar():
        complete_df = complete_df[complete_df["pmcid"] != ""]
        print("Search results # of pmc articles : ", len(complete_df))
        complete_df = complete_df.drop_duplicates(subset=['pmcid'])
        print("After dropping duplicate articles, # of pmc articles : ", len(complete_df))
        complete_df = complete_df[complete_df["pmcid"].isin(bioc_pmc_id_set)]
        print("Number of articles in PMC OA: ", len(complete_df))
        result_df = complete_df.compute()
        save_path = os.path.join(save_dir, "bioc_pmc_oa_query_uid_results.tsv")
        print("Saving PMC OA article results to: ", save_path)
        result_df.to_csv(save_path, index=False, sep="\t")


def main():
    args = parse_argument()
    all_query_uid_results_paths = glob(os.path.join(args.search_results_dir, "query_uid_results_*.tsv"))
    print(f"Found files : {all_query_uid_results_paths}")
    dask_dfs = [dd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
                for path in all_query_uid_results_paths]
    all_results_df = dd.concat(dask_dfs)
    if args.op == "select_pmc_articles":
        select_pmc_articles(all_results_df, load_open_access_pmcids(args.bioc_pmc_dir), args.search_results_dir)
    elif args.op == "select_missing_keywords":
        select_missing_keywords(all_results_df, args.query, args.search_results_dir)
    else:
        raise ValueError(f"Unknown op: {args.op}")


if __name__ == "__main__":
    main()

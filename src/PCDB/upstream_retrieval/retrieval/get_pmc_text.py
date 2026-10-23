"""Read section text (TITLE, ABSTRACT, METHODS, RESULTS, ...) of a PMC article from BioC-PMC.

Archived code. Used for the classifier input (title + abstract) and for the
entity-extraction inputs (abstract, Results and Methods sections).
The BioC-PMC files carry an .xml suffix but hold BioC JSON.
"""
import argparse
import json
import os
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

ALLOWED_SECTIONS = [
    'INTRO',
    'METHODS',
    'CONCL',
    'TITLE',
    'ABSTRACT',
    'RESULTS',
    'DISCUSS',
]

ALLOWED_TYPES = [
    'abstract',
    'paragraph',
    'front'
]


def parse_argument():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--section_type', type=str, required=True)
    parser.add_argument('--data_file_path', type=str, required=True)
    parser.add_argument('--save_file_path', type=str, required=True)
    parser.add_argument('--bioc_pmc_dir', type=str, required=True)
    return parser.parse_args()


def load_passages(pmcid, bioc_pmc_dir):
    file_path = os.path.join(bioc_pmc_dir, pmcid + ".xml")
    if not os.path.exists(file_path):
        print(f"can not find file for pmcid: {pmcid}, return no passages")
        return []
    with open(file_path, "r") as f:
        documents = json.load(f)['documents']
    assert len(documents) == 1
    return documents[0]['passages']


def normalize_text(section_text):
    section_text = [unicodedata.normalize('NFKC', x) for x in section_text]
    return [re.sub(r'[​ ‌\xa0]', ' ', text) for text in section_text]


def get_section_text_by_pmcid(pmcid, target_section_type, bioc_pmc_dir, return_list=False):
    """Join all passages of one section type; 'all' keeps every allowed section."""
    section_text = []
    for passage in load_passages(pmcid, bioc_pmc_dir):
        infons = passage.get('infons', {})
        if 'section_type' not in infons or 'type' not in infons:
            continue
        if infons['type'] not in ALLOWED_TYPES:
            continue
        if target_section_type == 'all':
            if infons['section_type'] not in ALLOWED_SECTIONS:
                continue
        elif infons['section_type'] != target_section_type:
            continue
        section_text.append(passage['text'])

    section_text = normalize_text(section_text)
    if return_list:
        return section_text
    return '\n'.join(section_text).strip()


def get_pmc_text(section_type, data, bioc_pmc_dir):
    print("Retrieving section : ", section_type)
    tqdm.pandas()
    data[section_type] = data["pmcid"].progress_apply(
        lambda x: get_section_text_by_pmcid(x, section_type, bioc_pmc_dir))

    empty_rows_n = (data[section_type] == '').sum()
    if empty_rows_n > 0:
        print(f"Empty text for {section_type} : {empty_rows_n} rows")
    return data


if __name__ == '__main__':
    args = parse_argument()
    if args.section_type not in ALLOWED_SECTIONS + ["all"]:
        raise ValueError(f"{args.section_type} is not a correct section name.")
    data = pd.read_csv(args.data_file_path, sep="\t", keep_default_na=False, dtype=str)
    data = get_pmc_text(args.section_type, data, args.bioc_pmc_dir)
    data.to_csv(args.save_file_path, sep="\t", index=False)

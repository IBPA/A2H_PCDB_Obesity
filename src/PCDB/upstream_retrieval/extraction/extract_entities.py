"""Stage 3: extract diseases, drugs and animal models from PCDB articles with Mistral Large.

Reference implementation. PCDB itself was extracted with the same model, prompts, system
message and decoding settings through the Mistral AI batch API; this script runs them on
the open weights with Hugging Face transformers instead. Outputs can differ slightly from
the released preclinical_database_raw.tsv (sampling at temperature 0.1, hardware).

Model access: mistralai/Mistral-Large-Instruct-2411 is gated (Mistral Research License);
accept the terms on Hugging Face and set HF_TOKEN. The 123B model needs ~250 GB of GPU
memory in bf16 (e.g. 4 x A100 80GB), or about half that with --load_in_4bit.

Example (first 10 PCDB articles):
    python -m src.PCDB.upstream_retrieval.extraction.extract_entities --bioc_pmc_dir /path/to/BioC-PMC --limit 10
"""
import argparse
import json
import os
import re
from dataclasses import dataclass

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..retrieval.get_pmc_text import get_section_text_by_pmcid
from .prompts import (
    ANIMAL_EXTRACTION_PROMPT,
    DISEASE_EXTRACTION_PROMPT,
    DRUG_EXTRACTION_PROMPT,
    DRUG_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)

MODEL_NAME = "mistralai/Mistral-Large-Instruct-2411"

# Settings recorded in the original batch requests; the disease step's were not recorded, so it decodes greedily.
DECODING = {
    "disease": {"temperature": 0.0, "max_new_tokens": 1024},
    "drug_summary": {"temperature": 0.1, "max_new_tokens": 1024},
    "drug": {"temperature": 0.0, "max_new_tokens": 526},
    "animal": {"temperature": 0.1, "max_new_tokens": 1024},
}


@dataclass
class Generator:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def main() -> int:
    args = parse_arguments()
    articles = load_pcdb_articles(args.articles_path, args.limit)
    generator = load_generator(args.model_name, args.load_in_4bit)
    records = [extract_article_entities(article, generator, args.bioc_pmc_dir)
               for article in tqdm(articles.to_dict("records"))]
    write_raw_table(records, args.output_path)
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PCDB entities with Mistral Large.")
    parser.add_argument("--bioc_pmc_dir", required=True, help="Local copy of BioC-PMC.")
    parser.add_argument("--articles_path", default="data/PCDB/upstream_retrieval/retrieval/retrieved_articles.tsv.gz")
    parser.add_argument("--output_path", default="outputs/pcdb/upstream_retrieval/extraction/preclinical_database_raw.tsv")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N articles.")
    return parser.parse_args()


def load_pcdb_articles(articles_path: str, limit: int | None) -> pd.DataFrame:
    articles = pd.read_csv(articles_path, sep="\t", dtype={"pmcid": str})
    articles = articles[articles["in_pcdb"]].reset_index(drop=True)
    return articles.head(limit) if limit else articles


def load_generator(model_name: str, load_in_4bit: bool) -> Generator:
    token = os.environ.get("HF_TOKEN")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 token=token,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=quantization_config,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    return Generator(model, tokenizer)


def extract_article_entities(article: dict, generator: Generator, bioc_pmc_dir: str) -> dict:
    """Return one raw-table row: disease, interventions (as reported), drugs (single compounds), animals."""
    pmcid = article["pmcid"]
    sections = {name: get_section_text_by_pmcid(pmcid, name, bioc_pmc_dir)
                for name in ["TITLE", "ABSTRACT", "RESULTS", "METHODS"]}
    interventions = extract_interventions(sections["RESULTS"], generator)
    return {
        "pmcid": pmcid,
        "link": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}",
        "probability": article["yes_probability"],
        "TITLE": sections["TITLE"],
        "disease": extract_diseases(sections["TITLE"], sections["ABSTRACT"], generator),
        "interventions": interventions,
        "drugs": split_drug_combinations(interventions),
        "animals": extract_animals(sections["METHODS"], generator),
    }


def extract_diseases(title: str, abstract: str, generator: Generator) -> list[str]:
    response = generate(DISEASE_EXTRACTION_PROMPT % (title, abstract), generator, **DECODING["disease"])
    return first_list_in_json(response)


def extract_interventions(results_text: str, generator: Generator) -> list[str]:
    """Two steps, because extraction straight from a long Results section missed drugs."""
    if not results_text:
        return []
    summary = generate(DRUG_SUMMARY_PROMPT % results_text, generator, **DECODING["drug_summary"])
    response = generate(DRUG_EXTRACTION_PROMPT % summary, generator, **DECODING["drug"])
    return first_list_in_json(response)


def extract_animals(methods_text: str, generator: Generator) -> list[dict]:
    if not methods_text:
        return []
    response = generate(ANIMAL_EXTRACTION_PROMPT % methods_text, generator, **DECODING["animal"])
    return first_list_in_json(response)


def split_drug_combinations(interventions: list[str]) -> list[str]:
    return sorted({drug.strip() for intervention in interventions for drug in intervention.split(" + ")})


def generate(prompt: str, generator: Generator, temperature: float, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    inputs = generator.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to(generator.model.device)
    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    with torch.no_grad():
        output_ids = generator.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                              pad_token_id=generator.tokenizer.eos_token_id, **sampling)
    prompt_length = inputs["input_ids"].shape[1]
    return generator.tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)


def first_list_in_json(response: str) -> list:
    """Parse the model's JSON answer (possibly in a ```json fence) and return its list.

    Answers are either {"diseases": [...]} / {"drugs": [...]} or a bare list of animal dicts.
    Unparseable answers become an empty list, as one malformed answer should not stop a
    134k-article run; they are reported so they can be inspected.
    """
    text = re.sub(r"^```(?:json)?|```$", "", response.strip()).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"Unparseable model answer, stored as empty: {response[:200]!r}")
        return []
    if isinstance(parsed, dict):
        return next((value for value in parsed.values() if isinstance(value, list)), [])
    return parsed


def write_raw_table(records: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(records)} articles to {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())

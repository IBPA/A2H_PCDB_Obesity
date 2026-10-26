# Relation extraction

PCDB records which disease, drug and animal entities a publication *mentions*.
It does not record which of them took part in the same experiment. A paper
naming three diseases and two drugs yields six disease–drug pairs in the
database, and most of those pairs were never tested together.

This code asks a language model, once per publication, which of the relations
PCDB asserts are actually supported by an in vivo experiment in that paper. It
reproduces the `pcdb_relations_all.jsonl` shipped in the PCDB release.

## What you need

- A PCDB release directory — the one holding `pcdb_complete.tsv`,
  `disease_entities.tsv`, `drug_entities.tsv`, `animal_entities.tsv`.
- Full text for the publications, as files named by accession:
  `PMC1234567.xml` (JATS) or `PMC1234567.txt` (BioC plain text), in any
  directory layout. The PMC Open Access bulk subsets are the usual source.
  Papers with no file are kept with `status: no_text` and an empty prompt, so
  partial coverage is fine — the output stays a complete image of the database.
- One or more GPUs able to serve the model. A 24 GB card is enough.

```
pip install -r requirements.txt
uv pip install --pre vllm --torch-backend=auto     # see requirements.txt
```

## Running it

Everything is a command-line argument; nothing is read from a fixed location.
Run the scripts from inside this directory, since they import each other as
top-level modules.

**1. Build one prompt per publication.**

```
python build_input.py \
    --pcdb-dir /path/to/preclinical-db-v1.1 \
    --text-dir /path/to/pmc_oa \
    --outdir full_input/
```

Pass `--text-dir` more than once to span several sources; JATS wins over BioC
for the same accession. Writes `input_000.jsonl …` plus `index.csv`, which has
every field except the prompt so the set can be inspected without reading
gigabytes. Add `--limit N` for a trial run.

The whole set is roughly 4 GB, which is why it is sharded: shards let
extraction run in parallel, one per GPU.

**2. Serve the model.** Once per GPU, with a distinct `GPU_ID` and `PORT`:

```
GPU_ID=0 PORT=8000 ./serve.sh
```

**3. Extract.** One client per server, pointed at the shards you gave it:

```
python run_extraction.py \
    --shards full_input/input_000.jsonl full_input/input_001.jsonl \
    --out labels_gpu0.jsonl \
    --base-url http://127.0.0.1:8000/v1
```

Every finished publication is appended and flushed immediately, and a restart
skips what is already on disk — a sweep this long *will* be interrupted.

## Output

One JSONL record per publication, carrying a verdict per asserted relation:

```json
{"pmcid": "PMC10000367", "status": "ok",
 "labels": [{"row_id": 12, "label": "no", "reason": "entity_not_in_paper"},
            {"row_id": 13, "label": "yes", "reason": ""}]}
```

`label` is `yes`, `no` or `unclear`. `reason` is set on `no` and names the
ground for rejection — `entity_not_in_paper`, `induction_agent` (the drug
caused the disease rather than treating it), `in_vitro_only`,
`different_experiment`, `disease_not_modeled`, `background_mention`,
`imaging_or_reagent`, `normalization_error`.

## The context window is the lever

Prompt and completion share one window. At `MAXLEN=20480` the longest prompts
left too little room for the model to reason and answer, and verdicts came
back missing with `finish: stop` and no error. The fix is a larger `MAXLEN`,
not a larger `--max-tokens`. Publications whose roster is so long that it
would dwarf the paper are marked `oversized` and left for a separate pass at a
bigger window.

## The prompt is pinned

`prompt.py` holds the instruction text that produced the published labels,
with its sha256 checked on import. Editing the text without updating
`TASK_SHA256` raises immediately, because labels from an edited prompt are not
comparable with the published ones.

## Validation

On a held-out sample of 120 relations in 120 publications, adjudicated by a
human against a second independent annotator, the published labels score
accuracy 0.900, precision 0.933, recall 0.875 and Cohen's kappa 0.800.
`score_relation_validation.py` computes these from the sample shipped as
`data/PCDB/pcdb_relations/relation_validation_sample.csv`.

Of the 519,902 asserted relations, 15.63% are rejected: 81,245 `no` and 652
`unclear`. That is the rate as labelled. The sample is stratified by the
model's label, 60 per label, so it measures the model's conditional error
rates and not the database's label mix — correcting the database-wide rate
means reweighting those two rates, which is not done here.

## Files

| file | role |
|---|---|
| `prompt.py` | the instruction text, hash-pinned, and prompt assembly |
| `pcdb_entities.py` | PCDB catalogs, entity resolution, article text |
| `build_input.py` | one prompt per publication, sharded |
| `serve.sh` | vLLM server, pinned configuration |
| `run_extraction.py` | resumable extraction driver |
| `vllm_client.py` | stdlib HTTP client and reply parsing |
| `verify_completeness.py` | fails unless every relation PCDB asserts is labelled |
| `build_release_relations.py` | predictions to the release's `pcdb_relations_all.{jsonl,csv}` |
| `relation_statistics.py` | every reported statistic, derived from the release |
| `score_relation_validation.py` | accuracy, kappa and the conditional error rates |
| `requirements.txt` | pinned dependencies |

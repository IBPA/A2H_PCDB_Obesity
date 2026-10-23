# PCDB upstream pipeline

The steps that produce `preclinical_database_raw.tsv`, the input to `scripts/create_PCDB.sh`.

```console
bash scripts/run_pcdb_upstream.sh retrieval      # Stage 1
bash scripts/run_pcdb_upstream.sh classifier     # Stage 2 (4 x A100 80GB)
bash scripts/run_pcdb_upstream.sh score_corpus   # Stage 2, applied to all retrieved articles
bash scripts/run_pcdb_upstream.sh extraction     # Stage 3
```

Extra dependencies: `src/PCDB/upstream_retrieval/requirements.txt`.

## Stage counts

| Stage | Articles |
|---|---|
| PMC search (5,006 MeSH disease queries), unique articles | 2,559,425 |
| In the PMC open-access subset | 1,669,186 |
| With title and abstract (classifier input) | 1,616,149 |
| Classifier confidence ≥ 0.9 (PCDB publications) | 134,565 |

## Stage 1: literature retrieval

- **Database:** PubMed Central, searched with NCBI EDirect on 14 November 2024 (no date limit).
- **Queries:** `data/PCDB/upstream_retrieval/retrieval/mesh_disease_queries.tsv`. Each query combines a MeSH disease name and its entry terms with:

  ```
  {disease terms} AND (animal) AND ((rat) OR (rabbit) OR (monkey) OR (dog) OR (mouse) OR (pig) OR (hamster))
  ```

- **Retrieved articles:** `data/PCDB/upstream_retrieval/retrieval/retrieved_articles.tsv.gz`, one row per open-access article, with the classifier input and PCDB membership flags.

## Stage 2: article classifier

- **Model:** `meta-llama/Llama-3.3-70B-Instruct`, QLoRA fine-tuned with torchtune.
- **Prompt:** `src/PCDB/upstream_retrieval/classifier/prompts.py`.
- **Confidence score:** the softmax over the `yes` and `no` next-token logits.
- **Labeled data:** 525 articles (244 positive), `data/PCDB/upstream_retrieval/classifier/labeled_dataset.tsv`.
- **Splits:** `split_assignments.tsv`. 132 articles are held out for testing; the remaining 393 are used for 5-fold cross-validation.
- **Hyperparameter search:** learning rate {2e-5, 5e-5} × batch size {2, 4} × epochs {3, 5}.
- **Final model:** the best setting (lr 2e-5, batch size 4, 3 epochs), retrained on all 393 articles.
- **Predictions:** `outputs/pcdb/upstream_retrieval/classifier/`.
  - `grid_search_cv_results.tsv` and `grid_search_validation_predictions.tsv`: cross-validation.
  - `test_predictions.tsv`: test set.

| Test set (n = 132) | Precision | Recall | F1 |
|---|---|---|---|
| Confidence ≥ 0.5 | 0.85 | 0.98 | 0.91 |
| Confidence ≥ 0.9 (PCDB threshold) | 0.98 | 0.84 | 0.90 |

## Stage 3: entity extraction

- **Model:** Mistral Large Instruct 2411. PCDB was extracted through the Mistral AI API (`mistral-large-2411`).
- **Code:** `src/PCDB/upstream_retrieval/extraction/extract_entities.py` runs the same prompts and settings on the open weights (`mistralai/Mistral-Large-Instruct-2411`) with Hugging Face transformers.
- **Prompts:** `src/PCDB/upstream_retrieval/extraction/prompts.py`.
- **Inputs:**
  - Diseases are extracted from the title and abstract.
  - Drugs are extracted from the Results section, by summarizing the experiments first and then extracting the drugs from that summary.
  - Animal models are extracted from the Methods section.
- **Output:** `preclinical_database_raw.tsv` (see `data/PCDB/download.sh`; SHA-256 `2244dc6298e6db3cb13bf6ea91bdb40d9439d8d21a13a4106e6c56824bccf85b`).

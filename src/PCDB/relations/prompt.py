"""The relation-extraction prompt, and the assembly of one prompt per paper.

TASK is the instruction text that produced the published labels. It is pinned
by sha256 and checked on import, because a silent edit here would invalidate
every number downstream while leaving the code working.

A prompt is three parts: this task text, the article, and the roster of
candidate relations PCDB asserts for that article. Each candidate line carries
the mention its canonical name was extracted from - without it a model cannot
match 'DOX' in the text to 'Doxorubicin', nor notice when the database's name
and the paper's wording disagree.
"""

import hashlib
from pathlib import Path

import pcdb_entities

RULE = "=" * 70

TASK_SHA256 = "057a13407b3866adf32859212211bdd11890932770db996bf054cc77fc42924f"

TASK = """You are auditing a biomedical database. For each candidate relation below, decide whether those entities took part in the SAME in vivo experiment in the paper provided.

The database recorded which entities the paper mentions, not which were tested together. A paper naming several diseases and drugs therefore implies combinations that may never have been tested; your job is to say which are real.

Each entity is given as the database's own canonical name, followed where they differ by the mention it was extracted from, like:
  disease='Breast Neoplasms' (extracted as: breast cancer)
  drug='Sirolimus' (extracted as: rapamycin)
FIRST confirm the entity is actually in the paper, searching broadly before concluding it is missing. Papers seldom use the canonical name, so try the canonical name, the extracted mention, and any abbreviation the paper defines - a paper introduces one on first use, as "N-methyl-D-aspartate (NMDA)", then uses only "NMDA" after that. Expect the same thing to be written differently: spelling and nomenclature variants (ciclosporin, cyclosporine and cyclosporin A are one drug), a salt or ester of the same drug, systematic versus trivial chemical names for one compound, singular vs plural, or a formal term covering the paper's clinical phrase ("Liver Neoplasms" for "hepatocellular carcinoma"). None of that is an error - if the paper discusses what the canonical name denotes, the entity is present, however it is spelled.

The database's extraction and normalization are still fallible, and when they are wrong the relation is wrong whatever the experiment did. Reject it in two cases:
- the entity is absent under any name you can think of - the extractor recorded something the paper never studied -> reason "entity_not_in_paper"
- the canonical name denotes a genuinely different thing from what the paper describes, because the mention was mis-normalized. A wrongly expanded abbreviation is the usual cause: a paper writing "MC" for Micrandilactone C may have been recorded as "Mecobalamin" (vitamin B12), a different compound sharing those initials -> reason "normalization_error"

The question is whether the paper supports the entity, not whether a relation can be made to work. If it does not, answer "no". But do not reach for these two reasons merely because the paper's wording differs from the canonical name - that difference is normal, and is why the mention is given to you.

Label each row_id:
- "yes": the text shows these entities together in one in vivo experiment
- "no": the relation is not supported - either they are not in the same in vivo experiment, or the entity itself is wrong
- "unclear": the text genuinely does not settle it

Rules:
- In vivo only. Cell-culture and ex vivo experiments do not count.
- A positive control is "yes": a benchmark drug given to animals WAS tested against that disease in that species.
- Combination therapy is "yes" for each component drug.
- A drug that induced the disease it is paired with is "no" (streptozotocin is not a diabetes treatment). Judge this per pair, not per drug: the same drug can induce one condition and treat another in the same paper, and is then "no" with the first and "yes" with the second.
- A disease-drug pair needs BOTH of these to be "yes". (i) The drug was given to the animals as a candidate treatment - not as the agent that induced the disease, not as an assay reagent, not merely mentioned. A repurposed or first-in-class compound counts: what matters is the role it plays in this paper, not whether it is an approved therapy. (ii) The disease is present in the animal, either as a modeled phenotype (a tumour in that tissue, raised blood glucose, a motor deficit) or as its defining molecular lesion. For a disease-animal pair only (ii) applies; for a drug-animal pair, only (i).
- So a disease named only as clinical background, only as the drug's intended indication, or only as a comorbidity measured alongside, while the animals model something else, is "no".
- A drug-free experiment can still make a true disease-animal pair: an untreated disease-model arm genuinely links that disease to that species.

When the label is "no", give a reason from exactly this set:
  different_experiment      both are real here, but in separate experiments
  in_vitro_only             appears only in cell culture, enzyme assay, or ex vivo work
  induction_agent           the compound created the disease model rather than treating it
  imaging_or_reagent        used as a tracer, stain, vehicle, or tool compound that sets up a measurement - not as a treatment
  background_mention        named only as prior work or comparison
  disease_not_modeled       the disease was discussed or measured, not modeled in those animals
  entity_not_in_paper       the entity is absent from the paper - an extraction error
  normalization_error       the paper describes a different entity than the canonical name

Reply with JSON only, exactly one entry per row_id. Include "reason" when the label is "no":
{"labels": [{"row_id": 1, "label": "yes"}, {"row_id": 2, "label": "no", "reason": "entity_not_in_paper"}]}"""



def _verify_task() -> None:
    actual = hashlib.sha256(TASK.encode()).hexdigest()
    if actual != TASK_SHA256:
        raise RuntimeError(
            "TASK text does not match TASK_SHA256: the prompt has been edited.\n"
            f"  expected {TASK_SHA256}\n  actual   {actual}\n"
            "Labels produced with an edited prompt are not comparable with the "
            "published results. Update TASK_SHA256 deliberately if that is intended.")


_verify_task()


def build_prompt(pmcid: str, relations: list[dict], article: str,
                 entities: dict) -> str:
    mentions = mention_map(entities)
    candidates = "\n".join(describe(r, mentions) for r in relations)
    if entities["unmapped"]:
        candidates += ("\n\nAlso mentioned in this paper but NOT under audit "
                       "(no database entry; ignore): "
                       + ", ".join(entities["unmapped"]))
    return (f"{TASK}\n\n{RULE}\nPAPER\n{RULE}\n\n{article_only(article)}\n\n"
            f"{RULE}\nCANDIDATE RELATIONS ({len(relations)})\n{RULE}\n{candidates}")


def describe(relation: dict, mentions: dict) -> str:
    """One candidate line, carrying the mention each canonical name came from."""
    parts = []
    for kind in ("disease", "drug", "animal"):
        if not relation[kind]:
            continue
        said = mentions.get((kind, relation[f"{kind}_id"]), [])
        extracted = [m for m in said
                     if m.strip().lower().rstrip("s") != relation[kind].lower().rstrip("s")]
        suffix = (f" (extracted as: {', '.join(dict.fromkeys(extracted))})"
                  if extracted else "")
        parts.append(f"{kind}={relation[kind]!r}{suffix}")
    return (f"row_id {relation['row_id']}  [{relation['relation_type']}]  "
            + ", ".join(parts))


def mention_map(entities: dict) -> dict:
    """(kind, pcdb_id) -> the mentions PCDB resolved to it."""
    mapping = {}
    for pcdb_id, said in entities["diseases"].items():
        mapping[("disease", pcdb_id)] = said
    for pcdb_id, said in entities["drugs"].items():
        mapping[("drug", pcdb_id)] = said
    species = entities.get("animal_species_by_id") or {}
    for pcdb_id in entities["animals"]:
        said = species.get(pcdb_id) or entities["animal_species"]
        mapping[("animal", pcdb_id)] = [said]
    return mapping


def article_only(article: str) -> str:
    """Drop anything before the title; the prompt supplies its own preamble."""
    marker = "TITLE:"
    return article[article.index(marker):] if marker in article else article


def article_text(path: Path) -> str:
    """BioC-derived .txt files are already in build_article_text's layout, so
    they are read verbatim; .xml goes through the JATS parser."""
    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")
    return pcdb_entities.build_article_text(str(path))

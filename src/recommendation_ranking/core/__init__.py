"""Internals of the ranking pipeline, split by concern.

Pipeline scripts should import from `ranking_core`, which re-exports the public
names from here. Import a submodule directly only when editing the internals.

  config     — column names, paths, fixed vocabularies
  features   — raw dataset -> model frame; the fold-safe one-hot encoder
  evaluation — chosen-arm scoring and drug-level roll-ups
  models     — the three candidate regressors and their search spaces
  selection  — family choice and hyperparameter tuning, both fold-bounded
  labels     — human-readable preclinical design labels
"""

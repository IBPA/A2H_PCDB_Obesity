"""Sensitivity of the preclinical outcome to control adjustment.

Compares the unadjusted treatment-arm change against the control-adjusted
preclinical outcome, at the arm level and at the drug level. No model fitting.

Arm-level inputs come from `data/obesity/preclinical_arms_with_components.csv`,
reading its stored `treatment_change_pct` and `control_change_pct` directly.
Those columns already apply the per-unit convention, and their difference is
the published `preclinical_outcome`.

A negative control-adjusted outcome means either that treated animals lost
weight or that they gained less than controls. The attenuated-gain section
counts which, over all arms and over the arms entering each mixed-effects
model, since that distinction bears on how those models read.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "data" / "obesity" / "preclinical_arms_with_components.csv"
PAIRED_SOURCE = REPO_ROOT / "data" / "obesity" / "obesity_a2h_dataset.csv"
TABLE_OUT = Path(__file__).parent / "control_adjustment_sensitivity.csv"

# Rank 1 = most negative median (greatest weight loss) under each measure.
RANK_ASCENDING = True
SHIFT_THRESHOLD = 2
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0

# Disease-model classification and the genetic-strain exclusions, transcribed from
# src/obesity_LMER/*.R so the subsets here are the ones those models actually fit.
DIO_PATTERN = r"DIO|diet|high-fat|HFD"
GENETIC_PATTERN = r"ob/ob|db/db|genetic|leptin"
LMER_SUBSETS = {
    "liraglutide_dio_mice": ("liraglutide", "mice", (144, 170, 263)),
    "liraglutide_dio_rats": ("liraglutide", "rats", (144, 170, 263)),
    "semaglutide_dio_mice": ("semaglutide", "mice", (22,)),
}


def main() -> int:
    arms = load_arms(SOURCE)
    arm_correlation = spearman_with_ci(
        arms["treatment_change_pct"], arms["control_adjusted_outcome_pct"]
    )
    arm_bootstrap_ci = bootstrap_spearman_ci(
        arms["treatment_change_pct"].to_numpy(),
        arms["control_adjusted_outcome_pct"].to_numpy(),
    )
    per_drug = summarize_per_drug(arms)
    rank_correlations = correlate_drug_rankings(per_drug)
    attenuated_gain = count_attenuated_gain(arms, lmer_arm_ids(PAIRED_SOURCE))

    table = assemble_table(
        arm_correlation,
        arm_bootstrap_ci,
        control_term_distribution(arms),
        per_drug,
        rank_correlations,
        attenuated_gain,
    )
    table.to_csv(TABLE_OUT, index=False)
    print(f"wrote {TABLE_OUT}")
    return 0


def load_arms(path: Path) -> pd.DataFrame:
    """Read both arm changes as stored; the unit branch is already applied in them."""
    raw = pd.read_csv(path)

    arms = raw[["preclinical_arm_id", "intervention"]].copy()
    arms["treatment_change_pct"] = raw["treatment_change_pct"]
    arms["control_arm_change_pct"] = raw["control_change_pct"]
    arms["control_adjusted_outcome_pct"] = (
        raw["treatment_change_pct"] - raw["control_change_pct"]
    )

    if arms[["treatment_change_pct", "control_adjusted_outcome_pct"]].isna().any().any():
        raise ValueError("measures could not be computed for every arm")
    return arms


def spearman_with_ci(x: pd.Series, y: pd.Series, confidence: float = 0.95) -> dict:
    """Fisher z-transform CI on the Spearman rho (SE = 1/sqrt(n - 3))."""
    result = stats.spearmanr(x, y)
    n = len(x)
    z_critical = stats.norm.ppf(0.5 + confidence / 2)
    half_width = z_critical / np.sqrt(n - 3)
    low, high = np.tanh(np.arctanh(result.statistic) + np.array([-1, 1]) * half_width)
    return {
        "rho": float(result.statistic),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(result.pvalue),
        "n": n,
    }


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, confidence: float = 0.95
) -> dict:
    """Percentile CI from resampling arms, as a check on the Fisher z interval."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_DRAWS, len(x)))
    draws = np.array([stats.spearmanr(x[i], y[i]).statistic for i in indices])
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(draws[~np.isnan(draws)], [tail, 100 - tail])
    return {"ci_low": float(low), "ci_high": float(high), "n": len(x)}


def control_term_distribution(arms: pd.DataFrame) -> dict[str, float]:
    """Spread of the quantity control adjustment subtracts.

    This is what drives the arm-level disagreement, so it is reported across all
    arms rather than only as the per-drug medians below.
    """
    control = arms["control_arm_change_pct"]
    return {
        "median": float(control.median()),
        "mean": float(control.mean()),
        "sd": float(control.std()),
        "min": float(control.min()),
        "max": float(control.max()),
    }


def summarize_per_drug(arms: pd.DataFrame) -> pd.DataFrame:
    per_drug = (
        arms.groupby("intervention")
        .agg(
            n_arms=("preclinical_arm_id", "size"),
            median_treatment_change_pct=("treatment_change_pct", "median"),
            median_control_adjusted_outcome_pct=(
                "control_adjusted_outcome_pct",
                "median",
            ),
            median_control_arm_change_pct=("control_arm_change_pct", "median"),
        )
        .reset_index()
        .rename(columns={"intervention": "drug"})
    )

    per_drug["rank_treatment_change"] = per_drug["median_treatment_change_pct"].rank(
        ascending=RANK_ASCENDING, method="average"
    )
    per_drug["rank_control_adjusted"] = per_drug[
        "median_control_adjusted_outcome_pct"
    ].rank(ascending=RANK_ASCENDING, method="average")
    per_drug["rank_shift"] = (
        per_drug["rank_control_adjusted"] - per_drug["rank_treatment_change"]
    )
    per_drug["abs_rank_shift"] = per_drug["rank_shift"].abs()
    per_drug["rank_shift_gt2"] = per_drug["abs_rank_shift"] > SHIFT_THRESHOLD

    return per_drug.sort_values("rank_control_adjusted").reset_index(drop=True)


def lmer_arm_ids(path: Path) -> dict[str, list[int]]:
    """Arm ids entering each mixed-effects model, mirroring src/obesity_LMER/*.R.

    An arm qualifies as diet-induced only if it matches the DIO pattern and not
    the genetic one — the R scripts classify genetic first, so a strain matching
    both is Genetic — and the named arms are dropped as genetic backgrounds
    misclassified as DIO in the source annotation.
    """
    paired = pd.read_csv(path)
    disease_model = paired["preclinical_disease_model"].fillna("")
    eligible = (
        disease_model.str.contains(DIO_PATTERN, case=False)
        & ~disease_model.str.contains(GENETIC_PATTERN, case=False)
        & paired["translation_outcome"].notna()
        & paired["preclinical_dosage_duration(days)"].notna()
    )
    return {
        label: sorted(
            paired.loc[
                eligible
                & (paired["intervention"].str.lower() == drug)
                & (paired["preclinical_animal_species"] == species)
                & ~paired["preclinical_arm_id"].isin(excluded),
                "preclinical_arm_id",
            ].unique()
        )
        for label, (drug, species, excluded) in LMER_SUBSETS.items()
    }


def count_attenuated_gain(
    arms: pd.DataFrame, subsets: dict[str, list[int]]
) -> pd.DataFrame:
    """Split the negative control-adjusted outcomes into weight loss and gain.

    An arm whose treatment-arm change is non-negative gained weight, so its
    negative adjusted outcome reflects gaining less than control rather than
    losing weight. Reported for all arms and for each mixed-effects subset.
    """
    scopes = {"all_preclinical_arms": arms["preclinical_arm_id"].tolist(), **subsets}
    rows = []
    for label, arm_ids in scopes.items():
        subset = arms[arms["preclinical_arm_id"].isin(arm_ids)]
        negative = subset[subset["control_adjusted_outcome_pct"] < 0]
        attenuated = negative[negative["treatment_change_pct"] >= 0]
        rows.append({
            "label": label,
            "n_arms": len(subset),
            "n_negative_adjusted": len(negative),
            "n_attenuated_gain": len(attenuated),
            "pct_attenuated_gain": (
                100 * len(attenuated) / len(negative) if len(negative) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def correlate_drug_rankings(per_drug: pd.DataFrame) -> dict[str, dict]:
    """Rank correlation over all drugs, and over drugs with >=3 arms only.

    Single-arm drugs have a median equal to one observation, so the restricted
    version shows whether they drive the agreement.
    """
    multi_arm = per_drug[per_drug["n_arms"] >= 3]
    return {
        "all_drugs": spearman_with_ci(
            per_drug["rank_treatment_change"], per_drug["rank_control_adjusted"]
        ),
        "drugs_with_3plus_arms": spearman_with_ci(
            multi_arm["rank_treatment_change"], multi_arm["rank_control_adjusted"]
        ),
    }


def assemble_table(
    arm_correlation: dict,
    arm_bootstrap_ci: dict,
    control_distribution: dict[str, float],
    per_drug: pd.DataFrame,
    rank_correlations: dict[str, dict],
    attenuated_gain: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "section": "arm_level_correlation",
            "label": "all_preclinical_arms",
            "n": arm_correlation["n"],
            "spearman_rho": arm_correlation["rho"],
            "ci_low": arm_correlation["ci_low"],
            "ci_high": arm_correlation["ci_high"],
            "p_value": arm_correlation["p_value"],
            "ci_method": "fisher_z",
        },
        {
            "section": "arm_level_correlation",
            "label": "all_preclinical_arms",
            "n": arm_bootstrap_ci["n"],
            "spearman_rho": arm_correlation["rho"],
            "ci_low": arm_bootstrap_ci["ci_low"],
            "ci_high": arm_bootstrap_ci["ci_high"],
            "p_value": np.nan,
            "ci_method": f"bootstrap_percentile_{BOOTSTRAP_DRAWS}",
        },
    ]
    rows += [
        {
            "section": "drug_rank_correlation",
            "label": label,
            "n": stats_dict["n"],
            "spearman_rho": stats_dict["rho"],
            "ci_low": stats_dict["ci_low"],
            "ci_high": stats_dict["ci_high"],
            "p_value": stats_dict["p_value"],
            "ci_method": "fisher_z",
        }
        for label, stats_dict in rank_correlations.items()
    ]

    rows += [
        {
            "section": "control_arm_change_distribution",
            "label": statistic,
            "n": len(per_drug) and arm_correlation["n"],
            "value": value,
        }
        for statistic, value in control_distribution.items()
    ]

    gain_rows = attenuated_gain.assign(section="attenuated_gain")
    gain_rows["n"] = gain_rows["n_arms"]

    drug_rows = per_drug.assign(section="per_drug").rename(columns={"drug": "label"})
    drug_rows["n"] = drug_rows["n_arms"]

    columns = [
        "section",
        "label",
        "n",
        "n_arms",
        "value",
        "n_negative_adjusted",
        "n_attenuated_gain",
        "pct_attenuated_gain",
        "median_treatment_change_pct",
        "median_control_adjusted_outcome_pct",
        "median_control_arm_change_pct",
        "rank_treatment_change",
        "rank_control_adjusted",
        "rank_shift",
        "abs_rank_shift",
        "rank_shift_gt2",
        "spearman_rho",
        "ci_low",
        "ci_high",
        "p_value",
        "ci_method",
    ]
    return pd.concat(
        [pd.DataFrame(rows), gain_rows, drug_rows], ignore_index=True
    )[columns].round(4)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

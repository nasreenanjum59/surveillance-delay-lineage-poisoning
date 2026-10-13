"""
Reproduce the Results section
=============================

Single self contained script that regenerates the simulation, sensitivity,
control, measured rate and analytical application results reported in the
Results section, together with the retrospective analysis of UK surveillance
data. It has no dependency on any other file in the project.

The retrospective analysis requires either a network connection, to query the
open LAPIS instance, or a previously downloaded weekly counts file. If neither
is available that section is skipped and the remainder still runs.

    python reproduce_results.py

Runtime is approximately three to five minutes. Output:

    results_report.txt          console report, also written to disk
    table_sim_main.csv          Table: simulation against the closed form
    table_sweep.csv             Table: delay across the recall range
    table_gi_sensitivity.csv    Table: sensitivity to the generation interval
    table_background.csv        Table: background misclassification control
    table_measured_rate.csv     Table: results at the measured growth rate
    table_delay_implied.csv     Table: delay implied by the poisoning experiments
    latex_tables.tex            the same tables as LaTeX fragments
    fig_delay_distribution.png  delay histograms under the measured attack
    fig_delay_versus_recall.png simulated delay against the analytic model
    table_retrospective.csv     Table: detection dates against UKHSA designations
    table_censoring_sensitivity.csv  Table: retrospective result under four censoring dates
    weekly_BQ_1_submission.csv  weekly counts by submission date (detection)
    weekly_BQ_1_collection.csv  weekly counts by collection date (growth estimation)

Every table corresponds to one subsection of the Results.

Author: Nasreen Anjum
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

# =====================================================================
# 1. Configuration
# =====================================================================

CONFIG = {
    # Epidemic
    "growth_rate": 0.231,        # reference scenario, doubling time 3 days
    "generation_mean": 3.59,     # realised household generation time, Omicron
    "generation_cv": 0.4,        # coefficient of variation
    "seed_cases": 5,
    "background_daily": 3000,
    "population": 10_000_000,
    "horizon_days": 120,

    # Observation
    "sequencing_fraction": 0.05,
    "reporting_delay": 7,

    # Detection rules
    "count_threshold": 5,
    "prevalence_threshold": 0.01,
    "prevalence_weeks": 2,
    "growth_min_sequences": 20,

    # Experiment
    "n_replicates": 1000,
    "sweep_replicates": 300,
    "control_replicates": 1000,
    "random_seed": 42,
}

# The measured growth advantage for BQ.1 in the United Kingdom
MEASURED_GROWTH_RATE = 0.0513
MEASURED_HORIZON = 300
MEASURED_THRESHOLD = 20
MEASURED_REPLICATES = 500

LINEAGES = ["B.1.1.7", "B.1.617.2", "BA.1", "BA.2", "BA.5", "BQ.1", "P.1"]
SOURCE_INDEX = 6        # P.1
DESTINATION_INDEX = 1   # B.1.617.2

# Row normalised Random Forest confusion matrices from the companion study.
CLEAN_MATRIX = np.array([
    [99.9,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1],
    [ 1.9, 94.0,  2.8,  1.4,  0.0,  0.0,  0.0],
    [ 0.4,  2.0, 96.9,  0.4,  0.4,  0.0,  0.0],
    [ 0.0,  0.0,  0.0, 99.9,  0.1,  0.0,  0.0],
    [ 0.0,  0.0,  0.2,  0.5, 99.2,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.8,  9.3, 88.9,  0.0],
    [ 7.4,  0.0,  0.0,  0.0,  0.0,  0.0, 92.6],
]) / 100.0

# Targeted relabelling of P.1 as B.1.617.2 at a 70% source lineage rate.
POISONED_MATRIX_70 = np.array([
    [99.9,  0.1,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 1.9, 94.9,  2.3,  0.9,  0.0,  0.0,  0.0],
    [ 0.4,  1.8, 97.1,  0.0,  0.7,  0.0,  0.0],
    [ 0.0,  0.0,  0.0, 99.8,  0.1,  0.0,  0.0],
    [ 0.0,  0.0,  0.2,  0.4, 99.3,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.8,  8.6, 89.6,  0.0],
    [ 6.3, 75.8,  0.0,  0.0,  0.0,  0.0, 17.9],
]) / 100.0

# Source lineage recall reported in the companion study, by mapping and rate.
POISONING_RESULTS = [
    ("P.1 -> B.1.617.2", 0,   0.9260, 98.62),
    ("P.1 -> B.1.617.2", 30,  0.7474, 98.44),
    ("P.1 -> B.1.617.2", 50,  0.4105, 97.98),
    ("P.1 -> B.1.617.2", 70,  0.1790, np.nan),
    ("P.1 -> B.1.617.2", 100, 0.0000, 97.32),
    ("BQ.1 -> BA.5",     0,   0.8890, 98.62),
    ("BQ.1 -> BA.5",     30,  0.5593, 97.27),
    ("BQ.1 -> BA.5",     50,  0.2724, 96.19),
    ("BQ.1 -> BA.5",     100, 0.0000, 95.03),
    ("BA.5 -> BA.2",     0,   0.9920, 98.62),
    ("BA.5 -> BA.2",     30,  0.8617, 94.05),
    ("BA.5 -> BA.2",     50,  0.3892, 77.03),
    ("BA.5 -> BA.2",     100, 0.0000, 62.95),
]

REPORT = io.StringIO()


def say(text: str = "") -> None:
    """Print to the console and accumulate the report."""
    print(text)
    REPORT.write(text + "\n")


# =====================================================================
# 2. Model components
# =====================================================================

def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Return the matrix with every row summing to exactly one."""
    return matrix / matrix.sum(axis=1, keepdims=True)


def build_poisoned_matrix(clean: np.ndarray, target_recall: float) -> np.ndarray:
    """
    Construct a poisoned confusion matrix for an arbitrary source recall.

    The source diagonal is set to the required recall and the removed mass is
    transferred to the destination column, reproducing the structure seen in
    the measured matrices. This is a modelling assumption, not an observation.
    """
    poisoned = clean.copy()
    clean_recall = clean[SOURCE_INDEX, SOURCE_INDEX]
    if target_recall > clean_recall:
        raise ValueError("Target recall cannot exceed the clean recall.")
    poisoned[SOURCE_INDEX, SOURCE_INDEX] = target_recall
    poisoned[SOURCE_INDEX, DESTINATION_INDEX] += clean_recall - target_recall
    return normalise_rows(poisoned)


def generation_interval_pmf(mean: float, cv: float, max_days: int = 40) -> np.ndarray:
    """
    Discretised gamma generation interval, parameterised by mean and CV.

    Probability mass is assigned to each whole day by integrating the
    continuous density over the interval centred on that day, from s - 0.5 to
    s + 0.5. Assigning the interval from s - 1 to s instead would displace the
    mean of the resulting discrete distribution upwards by approximately half a
    day. The support extends to 40 days so that truncation removes a negligible
    share of the distribution even at the highest dispersion examined in the
    sensitivity analysis.
    """
    sd = mean * cv
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    days = np.arange(1, max_days + 1)
    density = (gamma_dist.cdf(days + 0.5, a=shape, scale=scale)
               - gamma_dist.cdf(days - 0.5, a=shape, scale=scale))
    return density / density.sum()


def reproduction_number(growth_rate: float, pmf: np.ndarray) -> float:
    """
    Reproduction number consistent with a target growth rate.

    Obtained by inverting the Euler-Lotka relation for the discretised
    generation interval, so the simulation grows at the rate the analytic model
    assumes rather than at a rate chosen independently.
    """
    days = np.arange(1, len(pmf) + 1)
    return 1.0 / np.sum(pmf * np.exp(-growth_rate * days))


def simulate_incidence(rng, cfg: dict, pmf: np.ndarray, r_eff: float) -> np.ndarray:
    """Daily incidence of the emerging lineage, as a renewal branching process."""
    horizon = cfg["horizon_days"]
    incidence = np.zeros(horizon, dtype=np.int64)
    incidence[0] = cfg["seed_cases"]
    window = len(pmf)
    cumulative = incidence[0]

    for t in range(1, horizon):
        lookback = min(t, window)
        past = incidence[t - lookback:t][::-1]
        susceptible = max(0.0, 1.0 - cumulative / cfg["population"])
        force = r_eff * np.sum(past * pmf[:lookback]) * susceptible
        if force <= 0:
            continue
        incidence[t] = rng.poisson(min(force, 1e9))
        cumulative += incidence[t]

    return incidence


def sequence_submissions(rng, cfg: dict, incidence: np.ndarray) -> np.ndarray:
    """
    True lineage counts of genomes reaching the repository on each day.

    Kept separate from classification so that the clean and poisoned pipelines
    receive exactly the same submitted genomes within a replicate.
    """
    horizon = cfg["horizon_days"]
    n_lin = len(LINEAGES)
    submitted = np.zeros((horizon, n_lin), dtype=np.int64)

    sequenced_source = rng.binomial(
        np.minimum(incidence, np.iinfo(np.int32).max), cfg["sequencing_fraction"])

    resident = [i for i in range(n_lin) if i != SOURCE_INDEX]
    per_lineage = cfg["background_daily"] // (n_lin - 1)
    sequenced_background = rng.binomial(per_lineage, cfg["sequencing_fraction"],
                                        size=(horizon, len(resident)))

    for t in range(horizon):
        arrival = t + cfg["reporting_delay"]
        if arrival >= horizon:
            continue
        submitted[arrival, SOURCE_INDEX] += sequenced_source[t]
        for k, i in enumerate(resident):
            submitted[arrival, i] += sequenced_background[t, k]

    return submitted


def classify(rng, submitted: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Assign a label to every submitted genome.

    Poisoning enters the simulation here and nowhere else: substituting the
    poisoned matrix for the clean one is the entire intervention.
    """
    horizon, n_lin = submitted.shape
    labelled = np.zeros((horizon, n_lin), dtype=np.int64)
    for t in range(horizon):
        for i in range(n_lin):
            n = submitted[t, i]
            if n > 0:
                labelled[t] += rng.multinomial(n, matrix[i])
    return labelled


# =====================================================================
# 3. Detection rules
# =====================================================================

def detect_by_count(labelled: np.ndarray, cfg: dict):
    cumulative = np.cumsum(labelled[:, SOURCE_INDEX])
    hits = np.flatnonzero(cumulative >= cfg["count_threshold"])
    return int(hits[0]) if hits.size else None


def detect_by_prevalence(labelled: np.ndarray, cfg: dict):
    n_weeks = labelled.shape[0] // 7
    consecutive = 0
    for w in range(n_weeks):
        block = labelled[w * 7:(w + 1) * 7]
        total = block.sum()
        if total == 0:
            consecutive = 0
            continue
        share = block[:, SOURCE_INDEX].sum() / total
        if share >= cfg["prevalence_threshold"]:
            consecutive += 1
            if consecutive >= cfg["prevalence_weeks"]:
                return (w + 1) * 7 - 1
        else:
            consecutive = 0
    return None


def slope_significant(weeks, successes, totals) -> bool:
    """Binomial logistic regression by IRLS; is the slope significantly > 0?"""
    x = np.column_stack([np.ones_like(weeks, dtype=float), weeks.astype(float)])
    beta = np.zeros(2)
    for _ in range(50):
        eta = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = totals * p * (1 - p)
        if np.all(w < 1e-10):
            return False
        z = eta + (successes - totals * p) / np.maximum(w, 1e-10)
        wx = x * w[:, None]
        try:
            new = np.linalg.solve(x.T @ wx, wx.T @ z)
        except np.linalg.LinAlgError:
            return False
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    eta = np.clip(x @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = totals * p * (1 - p)
    try:
        cov = np.linalg.inv(x.T @ (x * w[:, None]))
    except np.linalg.LinAlgError:
        return False
    return (beta[1] - 1.96 * np.sqrt(max(cov[1, 1], 1e-12))) > 0


def detect_by_growth(labelled: np.ndarray, cfg: dict):
    n_weeks = labelled.shape[0] // 7
    weekly_source = np.array([labelled[w * 7:(w + 1) * 7, SOURCE_INDEX].sum()
                              for w in range(n_weeks)])
    weekly_total = np.array([labelled[w * 7:(w + 1) * 7].sum()
                             for w in range(n_weeks)])
    for w in range(3, n_weeks):
        if weekly_source[:w + 1].sum() < cfg["growth_min_sequences"]:
            continue
        if slope_significant(np.arange(w + 1), weekly_source[:w + 1],
                             weekly_total[:w + 1]):
            return (w + 1) * 7 - 1
    return None


RULES = {"count": detect_by_count,
         "prevalence": detect_by_prevalence,
         "growth": detect_by_growth}


# =====================================================================
# 4. Experiment
# =====================================================================

def analytic_delay(clean_recall, poisoned_recall, growth_rate) -> float:
    return math.log(clean_recall / poisoned_recall) / growth_rate


def run_experiment(cfg: dict, poisoned_matrix: np.ndarray,
                   n_replicates: int) -> pd.DataFrame:
    """
    Run the paired experiment.

    Each replicate generates one epidemic and one set of submitted genomes,
    classified twice. Common random numbers remove sequencing noise from the
    comparison and isolate the effect of the classifier.
    """
    rng = np.random.default_rng(cfg["random_seed"])
    pmf = generation_interval_pmf(cfg["generation_mean"], cfg["generation_cv"])
    r_eff = reproduction_number(cfg["growth_rate"], pmf)

    clean = normalise_rows(CLEAN_MATRIX)
    poisoned = normalise_rows(poisoned_matrix)

    records = []
    for rep in range(n_replicates):
        incidence = simulate_incidence(rng, cfg, pmf, r_eff)
        cumulative = np.cumsum(incidence)
        submitted = sequence_submissions(rng, cfg, incidence)
        labelled_clean = classify(rng, submitted, clean)
        labelled_pois = classify(rng, submitted, poisoned)

        row = {"replicate": rep}
        for name, rule in RULES.items():
            t_clean = rule(labelled_clean, cfg)
            t_pois = rule(labelled_pois, cfg)
            if t_clean is None:
                row[f"{name}_delay"] = np.nan
                row[f"{name}_size_ratio"] = np.nan
                row[f"{name}_extra"] = np.nan
            elif t_pois is None:
                row[f"{name}_delay"] = np.inf
                row[f"{name}_size_ratio"] = np.inf
                row[f"{name}_extra"] = np.nan
            else:
                row[f"{name}_delay"] = t_pois - t_clean
                size_clean = max(cumulative[t_clean], 1)
                row[f"{name}_size_ratio"] = cumulative[t_pois] / size_clean
                row[f"{name}_extra"] = int(cumulative[t_pois] - size_clean)
            row[f"{name}_clean_day"] = t_clean

        t_ref = row["count_clean_day"]
        if t_ref is not None:
            dest_clean = labelled_clean[:t_ref + 1, DESTINATION_INDEX].sum()
            dest_pois = labelled_pois[:t_ref + 1, DESTINATION_INDEX].sum()
            row["destination_inflation"] = (dest_pois / dest_clean) if dest_clean else np.nan

        records.append(row)

    return pd.DataFrame(records)


def summarise(results: pd.DataFrame, predicted: float) -> pd.DataFrame:
    rows = []
    for name in RULES:
        delays = results[f"{name}_delay"]
        finite = delays[np.isfinite(delays)]
        ratios = results[f"{name}_size_ratio"]
        ratios = ratios[np.isfinite(ratios)]
        extra = results[f"{name}_extra"].dropna()
        rows.append({
            "rule": name,
            "analytic_days": round(predicted, 2),
            "median_days": round(float(finite.median()), 1) if len(finite) else np.nan,
            "iqr_low": round(float(finite.quantile(0.25)), 1) if len(finite) else np.nan,
            "iqr_high": round(float(finite.quantile(0.75)), 1) if len(finite) else np.nan,
            "never_detected": round(float(np.mean(np.isinf(delays))), 4),
            "size_ratio": round(float(ratios.median()), 2) if len(ratios) else np.nan,
            "extra_infections": int(extra.median()) if len(extra) else np.nan,
        })
    return pd.DataFrame(rows)


def background_control(cfg: dict, n_replicates: int) -> dict:
    """How often each rule fires when the emerging lineage is absent entirely."""
    control = dict(cfg)
    control["seed_cases"] = 0
    rng = np.random.default_rng(cfg["random_seed"] + 1)
    pmf = generation_interval_pmf(cfg["generation_mean"], cfg["generation_cv"])
    r_eff = reproduction_number(cfg["growth_rate"], pmf)
    clean = normalise_rows(CLEAN_MATRIX)

    fired = {name: [] for name in RULES}
    for _ in range(n_replicates):
        incidence = simulate_incidence(rng, control, pmf, r_eff)
        submitted = sequence_submissions(rng, control, incidence)
        labelled = classify(rng, submitted, clean)
        for name, rule in RULES.items():
            fired[name].append(rule(labelled, control))

    out = {}
    for name, days in fired.items():
        hits = [d for d in days if d is not None]
        out[name] = {"rate": len(hits) / n_replicates,
                     "median_day": float(np.median(hits)) if hits else None}
    return out


# =====================================================================
# 5. Analyses, one per Results subsection
# =====================================================================

def section_derived_reproduction_number(cfg: dict) -> None:
    say("5.3.1  DERIVED REPRODUCTION NUMBER AND REALISED GROWTH")
    say("-" * 70)
    pmf = generation_interval_pmf(cfg["generation_mean"], cfg["generation_cv"])
    r_eff = reproduction_number(cfg["growth_rate"], pmf)
    target = math.log(2) / cfg["growth_rate"]

    rng = np.random.default_rng(cfg["random_seed"])
    doublings = []
    for _ in range(200):
        incidence = simulate_incidence(rng, cfg, pmf, r_eff)
        a, b = incidence[20], incidence[45]
        if a > 0 and b > 0:
            doublings.append(25 * math.log(2) / math.log(b / a))
    realised = float(np.median(doublings))

    say(f"  generation interval : gamma, mean {cfg['generation_mean']} d, "
        f"CV {cfg['generation_cv']}")
    say(f"  growth rate         : {cfg['growth_rate']} per day "
        f"(target doubling {target:.2f} d)")
    say(f"  derived R_eff       : {r_eff:.3f}")
    say(f"  realised doubling   : {realised:.2f} d "
        f"(deviation {abs(realised - target) / target * 100:.1f}%)")
    say()


def section_main_table(cfg: dict) -> pd.DataFrame:
    say("5.3.2  AGREEMENT BETWEEN SIMULATION AND CLOSED FORM")
    say("-" * 70)
    clean_recall = normalise_rows(CLEAN_MATRIX)[SOURCE_INDEX, SOURCE_INDEX]
    pois_recall = normalise_rows(POISONED_MATRIX_70)[SOURCE_INDEX, SOURCE_INDEX]
    predicted = analytic_delay(clean_recall, pois_recall, cfg["growth_rate"])

    results = run_experiment(cfg, POISONED_MATRIX_70, cfg["n_replicates"])
    table = summarise(results, predicted)

    say(f"  clean recall {clean_recall*100:.2f}%, poisoned recall "
        f"{pois_recall*100:.2f}%, ratio {clean_recall/pois_recall:.3f}")
    say(f"  predicted size ratio (eq. 6): {clean_recall/pois_recall:.2f}")
    say(table.to_string(index=False))
    inflation = results["destination_inflation"].dropna()
    say(f"  destination inflation median: {inflation.median():.4f}")
    say()
    table.to_csv("table_sim_main.csv", index=False)
    return table


def section_sweep(cfg: dict) -> pd.DataFrame:
    say("5.3.4  DELAY ACROSS THE RANGE OF RECALL REDUCTION")
    say("-" * 70)
    clean = normalise_rows(CLEAN_MATRIX)
    clean_recall = clean[SOURCE_INDEX, SOURCE_INDEX]
    rows = []
    for recall in [0.7474, 0.60, 0.4105, 0.30, 0.179, 0.10, 0.05, 0.02, 0.005]:
        matrix = build_poisoned_matrix(clean, recall)
        results = run_experiment(cfg, matrix, cfg["sweep_replicates"])
        delays = results["count_delay"]
        finite = delays[np.isfinite(delays)]
        rows.append({
            "source_recall": recall,
            "reduction_factor": round(clean_recall / recall, 2),
            "analytic_days": round(analytic_delay(clean_recall, recall,
                                                  cfg["growth_rate"]), 2),
            "median_days": float(finite.median()),
            "iqr_low": float(finite.quantile(0.25)),
            "iqr_high": float(finite.quantile(0.75)),
            "never_detected": round(float(np.mean(np.isinf(delays))), 4),
        })
    table = pd.DataFrame(rows)
    say(table.to_string(index=False))
    say()
    table.to_csv("table_sweep.csv", index=False)
    return table


def section_gi_sensitivity(cfg: dict) -> pd.DataFrame:
    say("5.3.5  SENSITIVITY TO THE GENERATION INTERVAL")
    say("-" * 70)
    clean_recall = normalise_rows(CLEAN_MATRIX)[SOURCE_INDEX, SOURCE_INDEX]
    pois_recall = normalise_rows(POISONED_MATRIX_70)[SOURCE_INDEX, SOURCE_INDEX]
    predicted = analytic_delay(clean_recall, pois_recall, cfg["growth_rate"])

    rows = []
    for mean in [3.59, 6.84]:
        for cv in [0.2, 0.4, 0.6, 0.8, 1.0]:
            local = dict(cfg)
            local.update(generation_mean=mean, generation_cv=cv)
            pmf = generation_interval_pmf(mean, cv)
            r_eff = reproduction_number(local["growth_rate"], pmf)
            results = run_experiment(local, POISONED_MATRIX_70,
                                     cfg["sweep_replicates"])
            delays = results["count_delay"]
            finite = delays[np.isfinite(delays)]
            ratios = results["count_size_ratio"]
            ratios = ratios[np.isfinite(ratios)]
            rows.append({
                "gi_mean": mean, "cv": cv,
                "gamma_shape": round(1 / cv ** 2, 1),
                "r_eff": round(r_eff, 2),
                "analytic_days": round(predicted, 2),
                "median_days": float(finite.median()),
                "iqr_low": float(finite.quantile(0.25)),
                "iqr_high": float(finite.quantile(0.75)),
                "size_ratio": round(float(ratios.median()), 2),
            })
    table = pd.DataFrame(rows)
    say(table.to_string(index=False))
    say(f"  analytic prediction is {predicted:.2f} days in every row")
    say()
    table.to_csv("table_gi_sensitivity.csv", index=False)
    return table


def section_background(cfg: dict) -> pd.DataFrame:
    say("5.3.6  CONTROL FOR BACKGROUND MISCLASSIFICATION")
    say("-" * 70)
    rows = []
    for horizon, threshold in [(120, 5), (300, 5), (300, 20)]:
        local = dict(cfg)
        local.update(horizon_days=horizon, count_threshold=threshold)
        control = background_control(local, cfg["control_replicates"])
        rows.append({
            "horizon_days": horizon,
            "count_threshold": threshold,
            "count_rate": control["count"]["rate"],
            "count_median_day": control["count"]["median_day"],
            "prevalence_rate": control["prevalence"]["rate"],
            "growth_rate_rule": control["growth"]["rate"],
        })
    table = pd.DataFrame(rows)
    say(table.to_string(index=False))
    say()
    table.to_csv("table_background.csv", index=False)
    return table


def section_measured_rate(cfg: dict) -> pd.DataFrame:
    say("5.4  RESULTS AT THE MEASURED GROWTH RATE")
    say("-" * 70)
    local = dict(cfg)
    local.update(growth_rate=MEASURED_GROWTH_RATE,
                 horizon_days=MEASURED_HORIZON,
                 count_threshold=MEASURED_THRESHOLD)

    clean_recall = normalise_rows(CLEAN_MATRIX)[SOURCE_INDEX, SOURCE_INDEX]
    pois_recall = normalise_rows(POISONED_MATRIX_70)[SOURCE_INDEX, SOURCE_INDEX]
    predicted = analytic_delay(clean_recall, pois_recall, MEASURED_GROWTH_RATE)

    pmf = generation_interval_pmf(local["generation_mean"], local["generation_cv"])
    r_eff = reproduction_number(MEASURED_GROWTH_RATE, pmf)

    results = run_experiment(local, POISONED_MATRIX_70, MEASURED_REPLICATES)
    table = summarise(results, predicted)

    say(f"  growth rate {MEASURED_GROWTH_RATE} per day "
        f"(doubling {math.log(2)/MEASURED_GROWTH_RATE:.2f} d), "
        f"derived R_eff {r_eff:.3f}")
    say(f"  horizon {MEASURED_HORIZON} d, count threshold "
        f"{MEASURED_THRESHOLD} (selected against the control)")
    say(table.to_string(index=False))
    inflation = results["destination_inflation"].dropna()
    say(f"  destination inflation median: {inflation.median():.4f}")
    say()
    table.to_csv("table_measured_rate.csv", index=False)
    return table


def section_implied_delays() -> pd.DataFrame:
    say("5.5  DELAY IMPLIED BY THE MEASURED POISONING EXPERIMENTS")
    say("-" * 70)
    rows = []
    baseline = {}
    for mapping, rate, recall, accuracy in POISONING_RESULTS:
        if rate == 0:
            baseline[mapping] = recall
        clean_recall = baseline[mapping]
        if recall <= 0:
            delay_td = np.inf
            size = np.inf
            d3 = d135 = np.nan
        else:
            delay_td = math.log2(clean_recall / recall)
            size = clean_recall / recall
            d3 = delay_td * 3.0
            d135 = delay_td * (math.log(2) / MEASURED_GROWTH_RATE)
        rows.append({
            "mapping": mapping, "poisoning_rate": rate,
            "recall_pct": round(recall * 100, 2),
            "accuracy_pct": accuracy,
            "delay_doubling_times": (round(delay_td, 2)
                                     if np.isfinite(delay_td) else "inf"),
            "delay_days_Td3": round(d3, 1) if np.isfinite(d3) else "",
            "delay_days_Td13_5": round(d135, 1) if np.isfinite(d135) else "",
            "size_factor": round(size, 2) if np.isfinite(size) else "inf",
        })
    table = pd.DataFrame(rows)
    say(table.to_string(index=False))
    say()
    table.to_csv("table_delay_implied.csv", index=False)
    return table



# =====================================================================
# 5b. Retrospective analysis of UK surveillance data
# =====================================================================

LAPIS_ENDPOINT = "https://lapis.cov-spectrum.org/open/v2/sample/aggregated"

# Three distinct dates in the history of the BQ.1 label. They are not
# interchangeable, and only the last of them determines when a standard
# lineage assignment workflow could apply the label.
#
#   2022-09-02  BQ.1 added to cov-lineages/pango-designation in commit b995eb75
#   2022-10-03  published in the pango-designation v1.15 release
#   2022-10-10  BQ alias available in pangolin-data v1.15.1, the data package
#               downloaded by pangolin, verified absent at v1.14
#
# Censoring uses the assignable date. The v1.15 release date is retained as a
# sensitivity case and gives identical detection dates.
BQ1_DESIGNATION_COMMIT = "2022-09-02"
BQ1_PANGO_RELEASE = "2022-10-03"
BQ1_ASSIGNABLE_DATE = "2022-10-10"
UKHSA_DATE = "2022-10-28"

RETRO = {
    "country": "United Kingdom",
    "target": "BQ.1",
    "assignable_date": BQ1_ASSIGNABLE_DATE,
    "ukhsa_designation": UKHSA_DATE,     # V-22OCT-01
    "date_from": "2022-07-01",
    "date_to": "2023-02-28",
    "growth_window": ("2022-08-22", "2022-11-21"),
    "weekly_file": "weekly_BQ_1_submission.csv",
}

CENSOR_DATES = {
    "None": None,
    "Designation commit, 2 Sep": BQ1_DESIGNATION_COMMIT,
    "Pango v1.15, 3 Oct": BQ1_PANGO_RELEASE,
    "pangolin-data v1.15.1, 10 Oct": BQ1_ASSIGNABLE_DATE,
}


def _lapis(params: dict) -> list:
    import json
    import urllib.parse
    import urllib.request
    url = LAPIS_ENDPOINT + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "surveillance-delay-repro/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise ValueError("Unexpected response shape from the API.")


def fetch_weekly(index_by: str, lineage: str | None) -> pd.DataFrame:
    """
    Weekly counts for one country, indexed by collection or submission date.

    Detection is assessed on submission dates because a surveillance system can
    only act on data it has received. Growth is estimated on collection dates
    because the growth of a lineage is a property of transmission.
    """
    field = "date" if index_by == "collection" else "dateSubmitted"
    prefix = "date" if index_by == "collection" else "dateSubmitted"
    params = {"country": RETRO["country"], "fields": field,
              f"{prefix}From": RETRO["date_from"],
              f"{prefix}To": RETRO["date_to"]}
    if lineage:
        params["pangoLineage"] = f"{lineage}*"

    frame = pd.DataFrame(_lapis(params)).rename(columns={field: "date"})
    if frame.empty:
        return pd.DataFrame(columns=["week", "count"])
    frame["date"] = pd.to_datetime(frame["date"], format="mixed", errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["week"] = frame["date"].dt.to_period("W").dt.start_time
    return frame.groupby("week")["count"].sum().reset_index()


def build_retro_series(index_by: str) -> pd.DataFrame:
    totals = fetch_weekly(index_by, None).rename(columns={"count": "total_count"})
    target = fetch_weekly(index_by, RETRO["target"]).rename(
        columns={"count": "target_count"})
    weekly = (totals.merge(target, on="week", how="left")
                    .fillna({"target_count": 0}))
    weekly["target_count"] = weekly["target_count"].astype(int)
    weekly = weekly.sort_values("week").reset_index(drop=True)
    weekly["share"] = weekly["target_count"] / weekly["total_count"].replace(0, np.nan)
    return weekly


def apply_rules_weekly(weekly: pd.DataFrame, cfg: dict) -> dict:
    """The three detection rules applied to an observed weekly series."""
    out = {}
    cumulative = weekly["target_count"].cumsum().to_numpy()
    hit = np.flatnonzero(cumulative >= cfg["count_threshold"])
    out["count"] = weekly["week"].iloc[hit[0]] if hit.size else None

    consecutive = 0
    out["prevalence"] = None
    for idx, value in enumerate(weekly["share"].fillna(0.0)):
        if value >= cfg["prevalence_threshold"]:
            consecutive += 1
            if consecutive >= cfg["prevalence_weeks"]:
                out["prevalence"] = weekly["week"].iloc[idx]
                break
        else:
            consecutive = 0

    out["growth"] = None
    successes = weekly["target_count"].to_numpy()
    totals = weekly["total_count"].to_numpy()
    for end in range(3, len(weekly)):
        if successes[:end + 1].sum() < cfg["growth_min_sequences"]:
            continue
        if slope_significant(np.arange(end + 1), successes[:end + 1],
                             totals[:end + 1]):
            out["growth"] = weekly["week"].iloc[end]
            break
    return out


def fit_growth_advantage(weekly: pd.DataFrame) -> tuple:
    """
    Logistic growth advantage of the target lineage, by binomial GLM.

    The share is modelled rather than the absolute count because weekly
    sequencing volume varies substantially, so counts partly measure sequencing
    capacity rather than the epidemic.
    """
    start, end = RETRO["growth_window"]
    window = weekly[(weekly["week"] >= start) & (weekly["week"] <= end)].copy()
    if len(window) < 4:
        return None, None, None
    window["t"] = (window["week"] - window["week"].min()).dt.days

    x = np.column_stack([np.ones(len(window)), window["t"].to_numpy(float)])
    successes = window["target_count"].to_numpy(float)
    totals = window["total_count"].to_numpy(float)
    beta = np.zeros(2)
    for _ in range(100):
        eta = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = totals * p * (1 - p)
        z = eta + (successes - totals * p) / np.maximum(w, 1e-10)
        wx = x * w[:, None]
        new = np.linalg.solve(x.T @ wx, wx.T @ z)
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    eta = np.clip(x @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = totals * p * (1 - p)
    cov = np.linalg.inv(x.T @ (x * w[:, None]))
    se = np.sqrt(cov[1, 1])
    return float(beta[1]), float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)


def section_retrospective(cfg: dict):
    """Results 5.1 and 5.2: the retrospective analysis of UK surveillance data."""
    say("5.1 / 5.2  RETROSPECTIVE ANALYSIS OF UK SURVEILLANCE DATA")
    say("-" * 70)

    import os
    try:
        if os.path.exists(RETRO["weekly_file"]):
            submission = pd.read_csv(RETRO["weekly_file"], parse_dates=["week"])
            submission["share"] = (submission["target_count"]
                                   / submission["total_count"].replace(0, np.nan))
            say(f"  using cached {RETRO['weekly_file']}")
            collection = None
        else:
            say("  querying the open LAPIS instance ...")
            submission = build_retro_series("submission")
            submission.to_csv(RETRO["weekly_file"], index=False)
            collection = build_retro_series("collection")
            collection.to_csv("weekly_BQ_1_collection.csv", index=False)
    except Exception as exc:
        say(f"  SKIPPED: {exc}")
        say("  Run with a network connection, or place a weekly counts file "
            f"named {RETRO['weekly_file']} alongside this script.")
        say()
        return None

    ukhsa = pd.to_datetime(RETRO["ukhsa_designation"])
    assignable = pd.to_datetime(RETRO["assignable_date"])
    rows = []

    for label, series in [("submission, uncensored", submission),
                          ("submission, censored", submission[submission["week"] >= assignable])]:
        detections = apply_rules_weekly(series.reset_index(drop=True), cfg)
        for rule, week in detections.items():
            rows.append({"series": label, "rule": rule,
                         "detection_week": week.date() if week is not None else None,
                         "error_days": (week - ukhsa).days if week is not None else None})

    if collection is not None:
        detections = apply_rules_weekly(collection, cfg)
        for rule, week in detections.items():
            rows.append({"series": "collection, uncensored", "rule": rule,
                         "detection_week": week.date() if week is not None else None,
                         "error_days": (week - ukhsa).days if week is not None else None})

    table = pd.DataFrame(rows)
    say(table.to_string(index=False))

    removed = int(submission.loc[submission["week"] < assignable,
                                 "target_count"].sum())
    by_cutoff = int(submission.loc[submission["week"] <= "2022-10-24",
                                   "target_count"].sum())
    say(f"  BQ.1 records excluded by censoring at {assignable.date()}: {removed}")
    say(f"  cumulative target by the UKHSA data cut-off week: {by_cutoff:,} "
        f"(UKHSA cited 717, from a different repository)")

    # Sensitivity of the detection dates to the censoring specification. The
    # two operationally defensible October dates are compared against the
    # earlier repository commit and against no censoring at all.
    say()
    say("  censoring sensitivity")
    sens_rows = []
    for label, date in CENSOR_DATES.items():
        if date is None:
            series = submission.copy()
            excluded = 0
        else:
            cut = pd.to_datetime(date)
            series = submission[submission["week"] >= cut].reset_index(drop=True)
            excluded = int(submission.loc[submission["week"] < cut,
                                          "target_count"].sum())
        found = apply_rules_weekly(series, cfg)
        row = {"censoring": label, "date": date, "bq1_records_excluded": excluded}
        for rule, week in found.items():
            row[f"{rule}_error_days"] = ((week - ukhsa).days
                                         if week is not None else np.nan)
        sens_rows.append(row)
    sensitivity = pd.DataFrame(sens_rows)
    say(sensitivity.to_string(index=False))
    sensitivity.to_csv("table_censoring_sensitivity.csv", index=False)
    say()

    # The growth advantage must be fitted on collection dates. Growth is a
    # property of transmission and is therefore indexed by when infections
    # occurred, whereas the submission series is shaped by repository batching
    # and would give a biased estimate.
    if collection is None:
        collection_file = "weekly_BQ_1_collection.csv"
        if os.path.exists(collection_file):
            collection = pd.read_csv(collection_file, parse_dates=["week"])
        else:
            say("  growth rate NOT estimated: the collection date series is "
                "required and was not available. Fitting on submission dates "
                "would give a biased estimate because submissions arrive in "
                "batches.")
            say()
            table.to_csv("table_retrospective.csv", index=False)
            return table

    slope, low, high = fit_growth_advantage(collection)
    if slope is not None:
        say(f"  logistic growth advantage s = {slope:.4f} per day "
            f"(95% CI {low:.4f} to {high:.4f}), fitted on collection dates")
        say(f"  doubling time of the odds = {math.log(2)/slope:.2f} days")
    say()
    table.to_csv("table_retrospective.csv", index=False)
    return table


# =====================================================================
# 6. Figures
# =====================================================================

def make_figures(cfg: dict, sweep: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean_recall = normalise_rows(CLEAN_MATRIX)[SOURCE_INDEX, SOURCE_INDEX]
    pois_recall = normalise_rows(POISONED_MATRIX_70)[SOURCE_INDEX, SOURCE_INDEX]
    predicted = analytic_delay(clean_recall, pois_recall, cfg["growth_rate"])

    results = run_experiment(cfg, POISONED_MATRIX_70, cfg["n_replicates"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, rule in zip(axes, RULES):
        values = results[f"{rule}_delay"]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=range(int(values.min()), int(values.max()) + 2),
                color="#4477aa", edgecolor="white", align="left")
        ax.axvline(predicted, color="#cc3311", linestyle="--", linewidth=2,
                   label=f"analytic ({predicted:.1f} d)")
        ax.axvline(values.median(), color="black", linestyle=":", linewidth=2,
                   label=f"simulated median ({values.median():.1f} d)")
        ax.set_title(f"{rule} rule")
        ax.set_xlabel("Detection delay (days)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Replicates")
    fig.suptitle("Detection delay under targeted poisoning of P.1 at a 70% rate")
    fig.tight_layout()
    fig.savefig("fig_delay_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["reduction_factor"], sweep["analytic_days"],
            color="#cc3311", linewidth=2, label="Analytic model")
    ax.errorbar(sweep["reduction_factor"], sweep["median_days"],
                yerr=[sweep["median_days"] - sweep["iqr_low"],
                      sweep["iqr_high"] - sweep["median_days"]],
                fmt="o", color="#4477aa", capsize=3,
                label="Simulation (median and IQR)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Recall reduction factor  $r_{clean}/r_{poisoned}$")
    ax.set_ylabel("Detection delay (days)")
    ax.set_title(f"Delay against recall reduction "
                 f"(doubling time {math.log(2)/cfg['growth_rate']:.1f} days)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_delay_versus_recall.png", dpi=300)
    plt.close(fig)

    say("Figures written: fig_delay_distribution.png, fig_delay_versus_recall.png")
    say()


def write_latex(tables: dict) -> None:
    """Write every table as a LaTeX fragment for pasting into the manuscript."""
    with open("latex_tables.tex", "w", encoding="utf-8") as handle:
        for name, table in tables.items():
            handle.write(f"% ---- {name} ----\n")
            handle.write(table.to_latex(index=False, escape=True,
                                        float_format="%.2f"))
            handle.write("\n\n")
    say("LaTeX fragments written: latex_tables.tex")


# =====================================================================
# 7. Entry point
# =====================================================================

def main() -> None:
    cfg = dict(CONFIG)

    say("=" * 70)
    say("REPRODUCTION OF THE RESULTS SECTION")
    say("=" * 70)
    say(f"replicates {cfg['n_replicates']} baseline, {cfg['sweep_replicates']} "
        f"per sweep point, {cfg['control_replicates']} per control")
    say(f"random seed {cfg['random_seed']}")
    say()

    tables = {}
    retro = section_retrospective(cfg)
    if retro is not None:
        tables["retrospective"] = retro

    section_derived_reproduction_number(cfg)
    tables["sim_main"] = section_main_table(cfg)
    tables["sweep"] = section_sweep(cfg)
    tables["gi_sensitivity"] = section_gi_sensitivity(cfg)
    tables["background"] = section_background(cfg)
    tables["measured_rate"] = section_measured_rate(cfg)
    tables["delay_implied"] = section_implied_delays()

    make_figures(cfg, tables["sweep"])
    write_latex(tables)

    with open("results_report.txt", "w", encoding="utf-8") as handle:
        handle.write(REPORT.getvalue())
    print("\nFull report written to results_report.txt")


if __name__ == "__main__":
    main()

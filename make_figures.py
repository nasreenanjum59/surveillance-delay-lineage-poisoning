"""
Generate the Results figures
============================

Produces five publication figures. Figures 1 to 4 are drawn entirely from the
cached surveillance data and from simulation output. Figure 5 is a vector
schematic of the experimental design and contains no numerical content.

    python make_figures.py

Requires weekly_BQ_1_collection.csv, weekly_BQ_1_submission.csv and
reproduce_results.py in the same directory.

Author: Nasreen Anjum
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import reproduce_results as R

# ---------------------------------------------------------------------
# Shared visual language
# ---------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

BLUE = "#33639B"
RED = "#C0392B"
GREEN = "#1F8A70"
ORANGE = "#D98324"
GREY = "#6B7280"
LIGHT = "#D8E1EA"

# First date on which the BQ.1 label could be applied by a standard lineage
# assignment workflow: the BQ alias entered pangolin-data at v1.15.1, released
# 10 October 2022, and is absent from v1.14. This is distinct from the
# designation commit of 2 September and the pango-designation v1.15 release of
# 3 October, both of which precede operational availability.
ASSIGNABLE_DATE = pd.Timestamp("2022-10-10")
UKHSA_DATE = pd.Timestamp("2022-10-28")


# ---------------------------------------------------------------------
# Figure 1. Retrospective surveillance timeline
# ---------------------------------------------------------------------

def figure_one() -> None:
    collection = pd.read_csv("weekly_BQ_1_collection.csv", parse_dates=["week"])
    submission = pd.read_csv("weekly_BQ_1_submission.csv", parse_dates=["week"])
    for frame in (collection, submission):
        frame["share"] = frame["target_count"] / frame["total_count"].replace(0, np.nan)

    cfg = dict(R.CONFIG)
    detect_collection = R.apply_rules_weekly(collection, cfg)
    censored = submission[submission["week"] >= ASSIGNABLE_DATE].reset_index(drop=True)
    detect_submission = R.apply_rules_weekly(censored, cfg)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    rule_colour = {"count": RED, "prevalence": ORANGE, "growth": GREEN}

    panels = [
        (axes[0], collection, detect_collection, "A",
         "Indexed by collection date", False),
        (axes[1], submission, detect_submission, "B",
         "Indexed by submission date", True),
    ]

    for ax, frame, detections, letter, title, censor in panels:
        bars = ax.bar(frame["week"], frame["total_count"], width=5.5,
                      color=LIGHT, edgecolor="none", zorder=1,
                      label="all sequences per week")
        ax.set_ylabel("Sequences per week")
        ax.set_ylim(0, frame["total_count"].max() * 1.25)

        twin = ax.twinx()
        twin.spines["top"].set_visible(False)
        twin.grid(False)
        twin.plot(frame["week"], frame["share"] * 100, color=BLUE,
                  marker="o", markersize=3.2, linewidth=1.6, zorder=3,
                  label="BQ.1 share")
        twin.set_ylabel("BQ.1 share of sequences (%)", color=BLUE)
        twin.tick_params(axis="y", colors=BLUE)
        twin.set_ylim(0, 70)

        if censor:
            ax.axvspan(frame["week"].min(), ASSIGNABLE_DATE, color="#000000",
                       alpha=0.055, zorder=0)
            ax.text(frame["week"].min() + pd.Timedelta(days=4),
                    frame["total_count"].max() * 1.13,
                    "label not yet\nassignable", fontsize=7.2, color=GREY,
                    va="top")

        ax.axvline(ASSIGNABLE_DATE, color=GREY, linestyle=(0, (4, 2)),
                   linewidth=1.2, zorder=2)
        ax.axvline(UKHSA_DATE, color="black", linewidth=1.8, zorder=2)

        seen: dict = {}
        for rule, week in detections.items():
            if week is None:
                continue
            twin.axvline(week, color=rule_colour[rule], linestyle=":",
                         linewidth=1.6, zorder=2)
            # Offset markers that coincide so overlapping rules stay visible
            nudge = seen.get(week, 0)
            seen[week] = nudge + 1
            twin.plot([week + pd.Timedelta(days=3.2 * nudge)], [66 - 4.5 * nudge],
                      marker="v", markersize=5,
                      color=rule_colour[rule], zorder=4)

        ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold")

    axes[1].set_xlabel("Week")

    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", markersize=3.5,
                   linewidth=1.6, label="BQ.1 share of sequences"),
        plt.Rectangle((0, 0), 1, 1, color=LIGHT, label="all sequences per week"),
        plt.Line2D([], [], color="black", linewidth=1.8,
                   label="UKHSA designation, 28 Oct 2022"),
        plt.Line2D([], [], color=GREY, linestyle=(0, (4, 2)), linewidth=1.2,
                   label="BQ.1 assignable in pangolin-data, 10 Oct 2022"),
        plt.Line2D([], [], color=RED, linestyle=":", linewidth=1.6,
                   label="count rule"),
        plt.Line2D([], [], color=ORANGE, linestyle=":", linewidth=1.6,
                   label="prevalence rule"),
        plt.Line2D([], [], color=GREEN, linestyle=":", linewidth=1.6,
                   label="growth rule"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    fig.autofmt_xdate(rotation=30)
    fig.savefig("figure1_retrospective_timeline.png")
    fig.savefig("figure1_retrospective_timeline.pdf")
    plt.close(fig)
    print("Figure 1 written")
    return detect_collection, detect_submission


# ---------------------------------------------------------------------
# Figure 2. Analytical benchmark against stochastic simulation
# ---------------------------------------------------------------------

def figure_two() -> None:
    sweep = pd.read_csv("table_sweep.csv")
    clean_recall = R.normalise_rows(R.CLEAN_MATRIX)[R.SOURCE_INDEX, R.SOURCE_INDEX]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    grid = np.logspace(np.log10(sweep["reduction_factor"].min()),
                       np.log10(sweep["reduction_factor"].max()), 300)
    ax.plot(grid, np.log(grid) / R.CONFIG["growth_rate"], color=RED,
            linewidth=2.0, zorder=2, label="Analytical benchmark")

    lower = sweep["median_days"] - sweep["iqr_low"]
    upper = sweep["iqr_high"] - sweep["median_days"]
    ax.errorbar(sweep["reduction_factor"], sweep["median_days"],
                yerr=[lower, upper], fmt="o", color=BLUE, markersize=5,
                capsize=3, linewidth=1.3, zorder=3,
                label="Simulation, median and IQR")

    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Recall reduction factor  $r_{\mathrm{clean}}\,/\,r_{\mathrm{poisoned}}$")
    ax.set_ylabel("Detection delay (days)")
    ax.set_xlim(1.1, 260)
    ax.set_ylim(-1.5, 27)

    secondary = ax.secondary_yaxis(
        "right",
        functions=(lambda d: d / (math.log(2) / R.CONFIG["growth_rate"]),
                   lambda t: t * (math.log(2) / R.CONFIG["growth_rate"])))
    secondary.set_ylabel("Delay (lineage doubling times)")
    secondary.spines["top"].set_visible(False)

    ax.annotate("70% poisoning of P.1\nrecall 92.6% to 17.9%",
                xy=(clean_recall / 0.179, 7.0),
                xytext=(2.4, 15.5), fontsize=7.8, color=GREY,
                arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    ax.legend(loc="upper left")
    fig.savefig("figure2_benchmark_versus_simulation.png")
    fig.savefig("figure2_benchmark_versus_simulation.pdf")
    plt.close(fig)
    print("Figure 2 written")


# ---------------------------------------------------------------------
# Figure 3. Distribution of detection delay
# ---------------------------------------------------------------------

def figure_three(replicates: pd.DataFrame, benchmark: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2), sharey=True)
    titles = {"count": "Count rule", "prevalence": "Prevalence rule",
              "growth": "Growth rule"}

    for ax, rule in zip(axes, ["count", "prevalence", "growth"]):
        values = replicates[f"{rule}_delay"]
        values = values[np.isfinite(values)]
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
        ax.hist(values, bins=bins, color=LIGHT, edgecolor=BLUE,
                linewidth=0.8, zorder=2)
        ax.axvline(benchmark, color=RED, linewidth=1.8, zorder=3)
        ax.axvline(values.median(), color="black", linestyle=(0, (4, 2)),
                   linewidth=1.4, zorder=3)
        ax.set_title(titles[rule], loc="left", fontweight="bold")
        ax.set_xlabel("Detection delay (days)")
        ax.text(0.97, 0.94,
                f"median {values.median():.0f} d\nIQR {values.quantile(.25):.0f}"
                f"\u2013{values.quantile(.75):.0f} d",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
                color=GREY)

    axes[0].set_ylabel("Replicates")
    handles = [
        plt.Line2D([], [], color=RED, linewidth=1.8,
                   label=f"analytical benchmark, {benchmark:.2f} d"),
        plt.Line2D([], [], color="black", linestyle=(0, (4, 2)), linewidth=1.4,
                   label="simulated median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.09))
    fig.savefig("figure3_delay_distribution.png")
    fig.savefig("figure3_delay_distribution.pdf")
    plt.close(fig)
    print("Figure 3 written")


# ---------------------------------------------------------------------
# Figure 4. Reference against measured growth scenario
# ---------------------------------------------------------------------

def figure_four(reference: pd.DataFrame, measured: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    rules = ["count", "prevalence", "growth"]
    labels = ["Count", "Prevalence", "Growth"]
    positions = np.arange(len(rules))

    ax = axes[0]
    for offset, table, colour, name, benchmark in [
        (-0.16, reference, BLUE, "Reference, $T_d$ = 3 d", 7.11),
        (0.16, measured, ORANGE, "Measured, $T_d$ = 13.5 d", 32.04),
    ]:
        medians = [table.loc[table["rule"] == r, "median_days"].iloc[0] for r in rules]
        lows = [table.loc[table["rule"] == r, "iqr_low"].iloc[0] for r in rules]
        highs = [table.loc[table["rule"] == r, "iqr_high"].iloc[0] for r in rules]
        ax.errorbar(positions + offset, medians,
                    yerr=[np.array(medians) - np.array(lows),
                          np.array(highs) - np.array(medians)],
                    fmt="o", color=colour, markersize=6, capsize=3,
                    linewidth=1.4, label=name, zorder=3)
        ax.axhline(benchmark, color=colour, linestyle=(0, (4, 2)),
                   linewidth=1.1, alpha=0.8, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detection delay (days)")
    ax.set_title("A   Delay expands with slower growth", loc="left",
                 fontweight="bold")
    ax.set_ylim(0, 44)
    ax.legend(loc="upper left")
    ax.text(2.34, 33.6, "benchmark", fontsize=7, color=GREY, ha="right")
    ax.text(2.34, 8.7, "benchmark", fontsize=7, color=GREY, ha="right")

    ax = axes[1]
    width = 0.34
    for offset, table, colour, name in [
        (-width / 2, reference, BLUE, "Reference"),
        (width / 2, measured, ORANGE, "Measured"),
    ]:
        ratios = [table.loc[table["rule"] == r, "size_ratio"].iloc[0] for r in rules]
        ax.bar(positions + offset, ratios, width=width, color=colour,
               edgecolor="none", label=name, zorder=2)

    ax.axhline(5.17, color=RED, linewidth=1.8, zorder=3)
    ax.text(2.42, 5.36, "analytical\nprediction 5.17", fontsize=7.2,
            color=RED, ha="right", va="bottom")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Epidemic size ratio at detection")
    ax.set_title("B   Size ratio remains stable", loc="left", fontweight="bold")
    ax.set_ylim(0, 7.2)

    fig.savefig("figure4_reference_versus_measured.png")
    fig.savefig("figure4_reference_versus_measured.pdf")
    plt.close(fig)
    print("Figure 4 written")


# ---------------------------------------------------------------------
# Figure 5. Vector schematic of the experimental design
# ---------------------------------------------------------------------

def figure_five() -> None:
    """
    Schematic of the experimental design.

    Two pipelines receive the same submitted genomes and differ only in the
    classifier applied to them. The figure carries no numerical results beyond
    the two recall values that define the contrast.
    """
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_xlim(0.2, 9.8)
    ax.set_ylim(1.1, 11.9)
    ax.axis("off")

    def box(x, y, w, h, text, edge=BLUE, size=8.6, face="white"):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.10,rounding_size=0.12",
            facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=4)

    def arrow(x1, y1, x2, y2, colour=GREY, width=1.1):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
            linewidth=width, color=colour, zorder=2, shrinkA=1, shrinkB=1))

    LEFT, RIGHT = 2.9, 7.1

    box(5, 11.3, 5.0, 0.78, "Emerging lineage")
    arrow(5, 10.9, 5, 10.3)
    box(5, 9.9, 5.0, 0.78, "Infections")
    arrow(5, 9.5, 5, 8.9)
    box(5, 8.5, 5.0, 0.78, "Sequencing and submission")

    # Bracket connector: one stream of submitted genomes feeding both pipelines
    ax.plot([5, 5], [8.11, 7.78], color=GREY, linewidth=1.1, zorder=2)
    ax.plot([LEFT, RIGHT], [7.78, 7.78], color=GREY, linewidth=1.1, zorder=2)
    arrow(LEFT, 7.78, LEFT, 7.42, colour=BLUE)
    arrow(RIGHT, 7.78, RIGHT, 7.42, colour=ORANGE)

    for x, colour, title, recall in [
        (LEFT, BLUE, "Clean classifier", "recall 92.6%"),
        (RIGHT, ORANGE, "Poisoned classifier", "recall 17.9%"),
    ]:
        box(x, 6.85, 3.5, 1.05, f"{title}\n{recall}", edge=colour, size=8.4)
        arrow(x, 6.3, x, 5.65, colour=colour)
        box(x, 5.15, 3.5, 0.78, "Detection rule", edge=colour, size=8.4)
        arrow(x, 4.72, x, 4.05, colour=colour)

    box(LEFT, 3.5, 3.5, 0.92, "Detection day", edge=BLUE, size=8.4)
    box(RIGHT, 3.5, 3.5, 0.92, "Detection day", edge=ORANGE, size=8.4)

    ax.annotate("", xy=(RIGHT, 2.62), xytext=(LEFT, 2.62),
                arrowprops=dict(arrowstyle="<|-|>", color="black",
                                linewidth=1.4, mutation_scale=12))
    ax.text(5, 2.12, "surveillance delay", ha="center", va="center",
            fontsize=9.2, fontweight="bold")
    ax.text(5, 1.55,
            "the only difference between the two pipelines\nis the classifier",
            ha="center", va="center", fontsize=7.6, color=GREY)

    fig.savefig("figure5_design_schematic.png")
    fig.savefig("figure5_design_schematic.pdf")
    plt.close(fig)
    print("Figure 5 written")


# ---------------------------------------------------------------------

def main() -> None:
    cfg = dict(R.CONFIG)
    clean_recall = R.normalise_rows(R.CLEAN_MATRIX)[R.SOURCE_INDEX, R.SOURCE_INDEX]
    pois_recall = R.normalise_rows(R.POISONED_MATRIX_70)[R.SOURCE_INDEX, R.SOURCE_INDEX]
    benchmark = R.analytic_delay(clean_recall, pois_recall, cfg["growth_rate"])

    detect_collection, detect_submission = figure_one()
    print("  collection date detections:",
          {k: str(v.date()) if v is not None else None
           for k, v in detect_collection.items()})
    print("  submission date detections (censored):",
          {k: str(v.date()) if v is not None else None
           for k, v in detect_submission.items()})

    figure_two()

    print("running the reference configuration for the delay distribution ...")
    replicates = R.run_experiment(cfg, R.POISONED_MATRIX_70, cfg["n_replicates"])
    figure_three(replicates, benchmark)

    reference = pd.read_csv("table_sim_main.csv")
    measured = pd.read_csv("table_measured_rate.csv")
    figure_four(reference, measured)

    figure_five()
    print("\nAll figures written as PNG and PDF.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 6 plotting script.

Reads data/benchmark_results.csv (written by scripts/benchmark.c via
`make bench`) and produces four PNGs in data/:

  latency_vs_n.png              median (solid) and p95 (dashed) query
                                 latency vs. dataset size N, one line
                                 per method, dimensionality fixed.
  latency_vs_dim.png            same, vs. vector dimensionality, N fixed.
  recall_vs_search_budget.png   single tree's recall@5 vs. search_budget.
  recall_vs_num_trees.png       forest's recall@5 vs. num_trees.

Percentiles rather than a single mean, per the latency-measurement
methodology settled before this was written (Gil Tene, "How NOT to
Measure Latency"): an average hides how the slow tail behaves, so the
median and the 95th percentile are plotted separately rather than
collapsed into one number.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Categorical colors, fixed per method across every plot in this file,
# so a method's identity never repaints between charts. First three
# slots of the validated default categorical palette (dataviz skill,
# references/palette.md): these three clear the colorblind-safety
# checks pairwise in both light and dark mode.
COLOR = {
    "brute_force": "#2a78d6",  # blue
    "single_tree": "#eb6834",  # orange
    "forest": "#1baf7a",       # aqua
}
LABEL = {
    "brute_force": "Brute force",
    "single_tree": "Single tree",
    "forest": "Forest",
}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_latency(df, sweep, x_col, x_label, out_path):
    subset = df[df["sweep"] == sweep]
    stats = (
        subset.groupby(["method", x_col])["latency_ns"]
        .agg(p50=lambda s: s.quantile(0.50), p95=lambda s: s.quantile(0.95))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)

    for method in ("brute_force", "single_tree", "forest"):
        m = stats[stats["method"] == method].sort_values(x_col)
        if m.empty:
            continue
        color = COLOR[method]
        ax.plot(m[x_col], m["p50"] / 1e6, color=color, linewidth=2,
                 marker="o", markersize=6, label=LABEL[method])
        ax.plot(m[x_col], m["p95"] / 1e6, color=color, linewidth=1.5,
                 linestyle="--", marker="o", markersize=4, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Query latency (ms)")
    ax.set_title(f"{x_label}: median (solid) vs. p95 (dashed) latency")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_recall(df, sweep, method, x_col, x_label, out_path):
    subset = df[(df["sweep"] == sweep) & (df["method"] == method)]
    stats = subset.groupby(x_col)["recall_at_5"].mean().reset_index().sort_values(x_col)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)
    ax.plot(stats[x_col], stats["recall_at_5"], color=COLOR[method],
             linewidth=2, marker="o", markersize=6, label=LABEL[method])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Recall@5 (mean over queries)")
    ax.set_title(f"Recall@5 vs. {x_label.lower()}")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    df = pd.read_csv("data/benchmark_results.csv")

    plot_latency(df, "latency_vs_n", "n", "Dataset size (N)",
                 "data/latency_vs_n.png")
    plot_latency(df, "latency_vs_dim", "dim", "Vector dimensionality (D)",
                 "data/latency_vs_dim.png")
    plot_recall(df, "recall_vs_budget", "single_tree", "search_budget",
                "Search budget", "data/recall_vs_search_budget.png")
    plot_recall(df, "recall_vs_num_trees", "forest", "num_trees",
                "Number of trees", "data/recall_vs_num_trees.png")


if __name__ == "__main__":
    main()

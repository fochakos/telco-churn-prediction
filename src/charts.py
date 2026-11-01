"""Draw the README charts from the data and the outputs of train.py.

Usage (from the project root, after train.py):
    python src/charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, load_clean  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 24,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt(b.get_height()),
                ha="center", va="bottom", color=INK, fontsize=9.5)


def churn_drivers():
    df = load_clean()
    df["tenure_band"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72],
                               labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    by_contract = df.groupby("Contract")["Churn"].mean()
    by_tenure = df.groupby("tenure_band", observed=True)["Churn"].mean()
    overall = df["Churn"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 5]})
    for ax, series, title, xlabel in [
        (axes[0], by_contract, "By contract", ""),
        (axes[1], by_tenure, "By tenure", "Months as a customer"),
    ]:
        bars = ax.bar(series.index.astype(str), series.values, width=0.6, color=BLUE)
        bar_labels(ax, bars, lambda v: f"{v:.0%}")
        ax.axhline(overall, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(0, 0.62)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(xlabel)
    axes[1].text(4.45, overall + 0.012, f"average {overall:.0%}", ha="right", color=MUTED, fontsize=9)
    fig.suptitle("New customers on month-to-month contracts churn the most",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.955, "Share of customers who left, Telco dataset (7,043 customers)",
             color=INK_2, fontsize=9.5)
    fig.savefig(FIG / "01_churn_drivers.png")
    plt.close(fig)


def profit_curve():
    curve = pd.read_csv(REPORTS / "profit_curve.csv")
    metrics = json.loads((REPORTS / "test_metrics.json").read_text())
    t = metrics["threshold"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(curve["threshold"], curve["profit_train_oof"], color=BLUE, linewidth=2)
    best = curve.loc[curve["threshold"].sub(t).abs().idxmin()]
    ax.scatter([t], [best["profit_train_oof"]], s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"chosen threshold {t:.2f}", (t, best["profit_train_oof"]), xytext=(0, -22),
                textcoords="offset points", ha="center", color=INK, fontsize=9.5)
    ax.set_ylim(top=curve["profit_train_oof"].max() * 1.08)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("Contacting fewer, riskier customers earns more")
    subtitle(ax, "Expected retention-campaign profit by score threshold (training set, out-of-fold)")
    ax.set_xlabel("Contact customers with a churn score at or above")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "02_profit_curve.png")
    plt.close(fig)


def importance():
    imp = pd.read_csv(REPORTS / "permutation_importance.csv").head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(imp["feature"], imp["importance"], height=0.6, color=BLUE,
            xerr=imp["std"], error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    ax.set_title("Tenure is by far the strongest signal")
    subtitle(ax, "Drop in test PR-AUC when each field is shuffled (permutation importance, top 8)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(FIG / "03_feature_importance.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    churn_drivers()
    profit_curve()
    importance()
    print(f"Saved charts to {FIG}")


if __name__ == "__main__":
    main()

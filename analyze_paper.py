"""
Aggregate OOD CSVs and training histories for the paper: sanity checks, report, figures.

Run from project root:  python analyze_paper.py
Outputs: analysis_report.md, results_discussion_detail.md, figures/*.pdf, tables/*.tex/csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import MODE_CONFIG

HERE = Path(__file__).resolve().parent
EXPECTED_MODES = tuple(MODE_CONFIG.keys())


def _load_history_best_val_acc(history_path: Path) -> tuple[float, int] | None:
    """Return (val_acc, epoch) at the checkpoint selected by minimum val_loss (matches train.py)."""
    if not history_path.is_file():
        return None
    with open(history_path, encoding="utf-8") as f:
        hist = json.load(f)
    if not hist:
        return None
    best_idx = min(range(len(hist)), key=lambda i: hist[i]["val_loss"])
    row = hist[best_idx]
    return float(row["val_acc"]), int(row["epoch"])


def load_ood_csvs(checkpoints_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    issues: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for p in sorted(checkpoints_dir.glob("ood_*.csv")):
        stem = p.stem  # ood_all -> mode is after ood_
        if not stem.startswith("ood_"):
            continue
        mode = stem[len("ood_") :]
        try:
            df = pd.read_csv(p)
        except Exception as e:
            issues.append(f"BLOCKING: failed to read {p}: {e}")
            continue
        frames[mode] = df
    return frames, issues


def validate_ood(mode: str, df: pd.DataFrame, issues: list[str]) -> None:
    required = {"corruption", "severity", "mode", "T", "mi", "brier", "ace", "accuracy"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        issues.append(f"BLOCKING: {mode}: missing columns {sorted(missing_cols)}")
        return

    if df["mode"].nunique() > 1:
        issues.append(f"WARNING: {mode}: multiple values in 'mode' column")

    n_c = df["corruption"].nunique()
    n_sev = df["severity"].nunique()
    expected_rows = n_c * 5
    if sorted(df["severity"].unique().tolist()) != [1, 2, 3, 4, 5]:
        issues.append(f"WARNING: {mode}: severity values are not exactly 1..5")

    if len(df) != n_c * n_sev:
        issues.append(
            f"WARNING: {mode}: row count {len(df)} != n_corruptions({n_c}) * n_severity({n_sev})"
        )

    # CIFAR-10-C full benchmark: 19 corruptions x 5 = 95 rows
    if n_c != 19 or len(df) != 95:
        issues.append(
            f"INFO: {mode}: expected 95 rows (19 corruptions x 5 severities) for full CIFAR-10-C; "
            f"got {len(df)} rows, {n_c} distinct corruptions."
        )

    t_vals = df["T"].unique()
    if len(t_vals) != 1:
        issues.append(f"BLOCKING: {mode}: T column not constant: {t_vals}")
    else:
        t = int(t_vals[0])
        if mode == "none" and t != 1:
            issues.append(f"BLOCKING: {mode}: expected T=1, got T={t}")
        elif mode != "none" and t != 50:
            issues.append(f"BLOCKING: {mode}: expected T=50, got T={t}")

    if df[["mi", "brier", "ace", "accuracy"]].isna().any().any():
        issues.append(f"BLOCKING: {mode}: NaNs in metric columns")

    if mode == "none":
        mi_max = df["mi"].max()
        if mi_max > 1e-5:
            issues.append(f"WARNING: {mode}: MI should be ~0 for T=1; max MI={mi_max}")
    else:
        if (df["mi"] < 0).any():
            issues.append(f"BLOCKING: {mode}: negative MI values")
        if (df["mi"] == 0).all():
            issues.append(f"WARNING: {mode}: all MI zero (unexpected for T>1)")


def aggregate_by_severity(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for mode, df in sorted(dfs.items()):
        g = df.groupby("severity", sort=True)[["accuracy", "ace", "brier", "mi"]].agg(["mean", "std"])
        for sev in range(1, 6):
            if sev not in g.index:
                continue
            rows.append(
                {
                    "mode": mode,
                    "severity": sev,
                    "accuracy_mean": g.loc[sev, ("accuracy", "mean")],
                    "accuracy_std": g.loc[sev, ("accuracy", "std")],
                    "ace_mean": g.loc[sev, ("ace", "mean")],
                    "ace_std": g.loc[sev, ("ace", "std")],
                    "brier_mean": g.loc[sev, ("brier", "mean")],
                    "brier_std": g.loc[sev, ("brier", "std")],
                    "mi_mean": g.loc[sev, ("mi", "mean")],
                    "mi_std": g.loc[sev, ("mi", "std")],
                }
            )
    return pd.DataFrame(rows)


def plot_metric_vs_severity(
    dfs: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    modes_order: tuple[str, ...],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sev = np.arange(1, 6)
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(modes_order)))
    for i, mode in enumerate(modes_order):
        if mode not in dfs:
            continue
        df = dfs[mode]
        means = df.groupby("severity")[metric].mean().reindex(sev)
        label = mode.replace("_", " ")
        ax.plot(sev, means.values, marker="o", ms=3, lw=1.2, label=label, color=cmap[i])
    ax.set_xticks(sev)
    ax.set_xlabel("CIFAR-10-C severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", fontsize=7)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _cor_short(name: str) -> str:
    s = str(name)
    return s[:-4] if s.lower().endswith(".npy") else s


def write_discussion_detail(
    dfs: dict[str, pd.DataFrame],
    checkpoints_dir: Path,
    project_root: Path,
) -> None:
    """
    Rich markdown + CSV for group discussion: pivots, per-corruption at severity 5, rankings, head-to-head counts.
    """
    lines: list[str] = [
        "# Detailed results (for discussion)",
        "",
        "Auto-generated by `analyze_paper.py`. Metrics are **mean softmax** predictive (`p_bar`); "
        "MI uses MC disagreement (identically ~0 for `none`, T=1).",
        "",
        "## 1. In-distribution (clean val acc at min val_loss epoch)",
        "",
        "| Mode | Best val acc | Epoch |",
        "|------|-------------|-------|",
    ]
    for mode in EXPECTED_MODES:
        hp = checkpoints_dir / f"resnet18_{mode}_history.json"
        got = _load_history_best_val_acc(hp)
        if got is None:
            lines.append(f"| `{mode}` | — | — |")
        else:
            acc, ep = got
            lines.append(f"| `{mode}` | {acc:.4f} | {ep} |")
    lines.extend(["", "---", "", "## 2. Mean over all corruptions (by severity)", ""])

    def pivot_mean(metric: str) -> pd.DataFrame:
        cols = {}
        for mode in EXPECTED_MODES:
            if mode not in dfs:
                continue
            g = dfs[mode].groupby("severity")[metric].mean()
            cols[mode] = g
        return pd.DataFrame(cols).reindex(range(1, 6))

    for metric, title in [
        ("accuracy", "Mean accuracy"),
        ("ace", "Mean ACE"),
        ("brier", "Mean Brier"),
        ("mi", "Mean MI"),
    ]:
        pv = pivot_mean(metric)
        lines.append(f"### {title}")
        lines.append("")
        lines.append(pv.to_markdown(floatfmt=".4f"))
        lines.append("")

    lines.extend(["---", "", "## 3. Mode rankings at each severity (by mean accuracy over corruptions)", ""])
    acc_pivot = pivot_mean("accuracy")
    for sev in range(1, 6):
        col = acc_pivot.loc[sev].sort_values(ascending=False)
        order = ", ".join(f"`{m}` ({col[m]:.4f})" for m in col.index)
        lines.append(f"- **Severity {sev}:** {order}")
    lines.append("")

    lines.extend(["---", "", "## 4. Severity 5 — every corruption (wide tables)", ""])
    lines.append("Use these to argue about *which* corruptions drive the averages.")
    lines.append("")

    metrics_cols = ["accuracy", "ace", "brier", "mi"]
    for metric in metrics_cols:
        pieces = []
        for mode in EXPECTED_MODES:
            if mode not in dfs:
                continue
            sub = dfs[mode][dfs[mode]["severity"] == 5][["corruption", metric]].rename(
                columns={metric: mode}
            )
            pieces.append(sub.set_index("corruption"))
        if not pieces:
            continue
        wide = pd.concat(pieces, axis=1)
        wide = wide.sort_index()
        wide.index = [_cor_short(i) for i in wide.index]
        lines.append(f"### {metric} (severity 5)")
        lines.append("")
        lines.append(wide.to_markdown(floatfmt=".4f"))
        lines.append("")

    out_csv = project_root / "tables" / "ood_severity5_by_corruption.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # long format for spreadsheets: corruption, mode, accuracy, ace, brier, mi at sev 5
    long_rows: list[dict] = []
    for mode in EXPECTED_MODES:
        if mode not in dfs:
            continue
        sub = dfs[mode][dfs[mode]["severity"] == 5].copy()
        sub["corruption_short"] = sub["corruption"].map(_cor_short)
        for _, r in sub.iterrows():
            long_rows.append(
                {
                    "corruption": r["corruption_short"],
                    "mode": mode,
                    "accuracy": r["accuracy"],
                    "ace": r["ace"],
                    "brier": r["brier"],
                    "mi": r["mi"],
                }
            )
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    lines.extend(
        [
            "---",
            "",
            "## 5. Head-to-head at severity 5 (per corruption)",
            "",
            "Counts are over the 19 corruptions. "
            "Ties (absolute difference under 1e-6) are counted as ties.",
            "",
        ]
    )

    def compare_pair(mode_a: str, mode_b: str, metric: str, higher_better: bool) -> tuple[int, int, int]:
        if mode_a not in dfs or mode_b not in dfs:
            return 0, 0, 0
        da = dfs[mode_a][dfs[mode_a]["severity"] == 5].set_index("corruption")[metric]
        db = dfs[mode_b][dfs[mode_b]["severity"] == 5].set_index("corruption")[metric]
        idx = da.index.intersection(db.index)
        win_a = win_b = tie = 0
        eps = 1e-6
        for c in idx:
            va, vb = float(da[c]), float(db[c])
            if abs(va - vb) < eps:
                tie += 1
            elif (va > vb) == higher_better:
                win_a += 1
            else:
                win_b += 1
        return win_a, win_b, tie

    pairs = [
        ("all", "last_layer_strict"),
        ("all", "first_block"),
        ("all", "none"),
        ("first_block", "last_layer_strict"),
    ]
    for a, b in pairs:
        lines.append(f"### `{a}` vs `{b}`")
        lines.append("")
        lines.append("| Metric | `{a}` wins | `{b}` wins | ties |".format(a=a, b=b))
        lines.append("|--------|------------|-------------|------|")
        for metric, hib in [("accuracy", True), ("ace", False), ("brier", False), ("mi", True)]:
            wa, wb, t = compare_pair(a, b, metric, hib)
            lines.append(f"| {metric} | {wa} | {wb} | {t} |")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 6. Largest accuracy drop from severity 1 to 5 (per mode)",
            "",
            "Corruption where acc(sev1) − acc(sev5) is largest (hardest *shift along severity* for that mode).",
            "",
        ]
    )
    for mode in EXPECTED_MODES:
        if mode not in dfs:
            continue
        df = dfs[mode]
        d1 = df[df["severity"] == 1].set_index("corruption")["accuracy"]
        d5 = df[df["severity"] == 5].set_index("corruption")["accuracy"]
        drop = (d1 - d5).sort_values(ascending=False)
        top3 = drop.head(3)
        lines.append(f"**{mode}** — top 3 drops:")
        for c, v in top3.items():
            lines.append(f"- `{_cor_short(c)}`: Δacc = {v:.4f}")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 7. Files for spreadsheets",
            "",
            f"- `{out_csv.relative_to(project_root)}` — long format, severity 5 only.",
            f"- `{Path('figures/summary_aggregate.csv')}` — mean ± std by mode × severity.",
            "",
        ]
    )

    out_md = project_root / "results_discussion_detail.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clean_val_tex(checkpoints_dir: Path, out_tex: Path) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    lines = [
        r"% Auto-generated by analyze_paper.py — best val_acc at epoch with minimum val_loss",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Mode & Best val.\ acc.\ & Epoch \\",
        r"\hline",
    ]
    rows_tex: list[str] = []
    for mode in EXPECTED_MODES:
        hp = checkpoints_dir / f"resnet18_{mode}_history.json"
        got = _load_history_best_val_acc(hp)
        if got is None:
            issues.append(f"WARNING: missing or empty history for {mode}: {hp}")
            rows_tex.append(f"{mode} & --- & --- \\\\")
            continue
        acc, ep = got
        rows_tex.append(f"\\texttt{{{mode}}} & {acc:.4f} & {ep} \\\\")
    lines.extend(rows_tex)
    lines.extend([r"\hline", r"\end{tabular}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    return issues, rows_tex


def append_issue_sections(report_lines: list[str], issues: list[str]) -> list[str]:
    blocking = [x for x in issues if x.startswith("BLOCKING:")]
    warnings = [x for x in issues if x.startswith("WARNING:")]
    info = [x for x in issues if x.startswith("INFO:")]
    notes = [x for x in issues if x.startswith("NOTE:")]

    def section(title: str, items: list[str]) -> None:
        report_lines.append(f"### {title}")
        report_lines.append("")
        if not items:
            report_lines.append("_(none)_")
        else:
            for it in items:
                report_lines.append(f"- {it}")
        report_lines.append("")

    section("Blocking", blocking)
    section("Warnings", warnings)
    section("Info", info)
    section("Notes (interpretation)", notes)
    return blocking


def append_aggregate_sections(
    report_lines: list[str],
    dfs: dict[str, pd.DataFrame],
    fig_dir: Path,
    ckpt: Path,
    no_plots: bool,
) -> None:
    report_lines.extend(["", "---", "", "## Aggregates (mean over corruptions)", ""])
    if not dfs:
        report_lines.append("_No OOD CSVs found; skipping aggregates and plots._")
        return

    agg = aggregate_by_severity(dfs)
    agg_path = fig_dir / "summary_aggregate.csv"
    agg.to_csv(agg_path, index=False)
    report_lines.append(f"Written `{agg_path.relative_to(HERE)}`.")
    try:
        write_discussion_detail(dfs, ckpt, HERE)
        report_lines.append(
            f"Discussion detail: `{Path('results_discussion_detail.md')}` and "
            f"`tables/ood_severity5_by_corruption.csv`."
        )
    except Exception as e:
        report_lines.append(f"WARNING: could not write discussion detail: {e}")
    report_lines.append("")
    report_lines.append("### Severity 5 snapshot (mean ± std over corruptions)")
    report_lines.append("")
    s5 = agg[agg["severity"] == 5].set_index("mode")
    for mode in EXPECTED_MODES:
        if mode not in s5.index:
            continue
        r = s5.loc[mode]
        report_lines.append(
            f"- **{mode}**: acc {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}, "
            f"ACE {r['ace_mean']:.4f} ± {r['ace_std']:.4f}, "
            f"Brier {r['brier_mean']:.4f} ± {r['brier_std']:.4f}, "
            f"MI {r['mi_mean']:.6f} ± {r['mi_std']:.6f}"
        )
    report_lines.append("")

    if no_plots:
        return
    plot_metric_vs_severity(
        dfs,
        "accuracy",
        "Mean accuracy",
        "CIFAR-10-C: mean accuracy vs severity",
        fig_dir / "ood_mean_accuracy_severity.pdf",
        EXPECTED_MODES,
    )
    plot_metric_vs_severity(
        dfs,
        "ace",
        "Mean ACE",
        "CIFAR-10-C: mean ACE vs severity",
        fig_dir / "ood_mean_ace_severity.pdf",
        EXPECTED_MODES,
    )
    plot_metric_vs_severity(
        dfs,
        "mi",
        "Mean MI",
        "CIFAR-10-C: mean MI vs severity",
        fig_dir / "ood_mean_mi_severity.pdf",
        EXPECTED_MODES,
    )
    report_lines.append(
        f"Figures: `{fig_dir.relative_to(HERE)}/ood_mean_accuracy_severity.pdf`, "
        f"`ood_mean_ace_severity.pdf`, `ood_mean_mi_severity.pdf`."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper analysis: sanity report + figures")
    ap.add_argument("--checkpoints-dir", type=Path, default=HERE / "checkpoints")
    ap.add_argument("--figures-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--tables-dir", type=Path, default=HERE / "tables")
    ap.add_argument("--no-plots", action="store_true", help="Skip PDF figure generation")
    args = ap.parse_args()

    ckpt = args.checkpoints_dir.resolve()
    fig_dir = args.figures_dir.resolve()
    tab_dir = args.tables_dir.resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = [
        "# Analysis report (auto-generated)",
        "",
        f"Checkpoints directory: `{ckpt}`",
        "",
        "## Significant issues",
        "",
    ]

    dfs, load_issues = load_ood_csvs(ckpt)
    issues: list[str] = list(load_issues)

    for m in EXPECTED_MODES:
        p = ckpt / f"ood_{m}.csv"
        if m not in dfs:
            issues.append(f"BLOCKING: missing `{p.name}`")

    for mode, df in sorted(dfs.items()):
        validate_ood(mode, df, issues)

    issues.append(
        "NOTE: For mode `none`, MI is ~0 by construction (T=1); interpret MI only for MC dropout modes."
    )
    blocking = append_issue_sections(report_lines, issues)

    if not blocking:
        report_lines.append("**No blocking issues detected.**")
    else:
        report_lines.append("**Fix blocking issues before relying on figures for the paper.**")
    append_aggregate_sections(report_lines, dfs, fig_dir, ckpt, args.no_plots)

    report_lines.extend(["", "## In-distribution (clean val acc at best val_loss epoch)", ""])
    hist_issues, _ = write_clean_val_tex(ckpt, tab_dir / "clean_val_acc.tex")
    for h in hist_issues:
        report_lines.append(f"- {h}")
    report_lines.append("")
    report_lines.append(f"Generated `{tab_dir.relative_to(HERE)}/clean_val_acc.tex` for \\input in LaTeX.")

    report_path = HERE / "analysis_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nWrote {report_path}")

    if blocking:
        sys.exit(1)


if __name__ == "__main__":
    main()

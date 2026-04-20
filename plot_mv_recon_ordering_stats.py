#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_RE = re.compile(
    r"Idx:\s*(?P<scene_id>[^,]+),\s*"
    r"Acc:\s*(?P<acc>[^,]+),\s*"
    r"Comp:\s*(?P<comp>[^\s]+)\s*-\s*"
    r"Acc_med:\s*(?P<acc_med>[^,]+),\s*"
    r"Compc_med:\s*(?P<comp_med>[^\s]+)"
)


def parse_run(run_dir: Path, dataset_name: str):
    rows = []
    log_dir = run_dir / dataset_name
    for log_path in sorted(log_dir.glob("logs_*.txt")):
        if log_path.name == "logs_all.txt":
            continue
        for line in log_path.read_text().splitlines():
            match = METRIC_RE.search(line)
            if match is None:
                continue
            data = match.groupdict()
            rows.append(
                {
                    "run": run_dir.name,
                    "scene_id": data["scene_id"],
                    "acc": float(data["acc"]),
                    "comp": float(data["comp"]),
                    "acc_med": float(data["acc_med"]),
                    "comp_med": float(data["comp_med"]),
                }
            )
    return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_run_means(run_rows, metrics, output_path: Path):
    runs = [row["run"] for row in run_rows]
    x = np.arange(len(runs))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        y = np.array([row[metric] for row in run_rows], dtype=np.float64)
        mean = y.mean()
        std = y.std(ddof=1) if len(y) > 1 else 0.0
        ax.plot(x, y, marker="o", linewidth=1.4, label=metric)
        ax.axhline(mean, color="black", linestyle="--", linewidth=1, label=f"mean={mean:.6f}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, label=f"std={std:.6f}")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(runs, rotation=45, ha="right")
    axes[-1].set_xlabel("run")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scene_variance(scene_rows, metrics, output_path: Path):
    scenes = sorted({row["scene_id"] for row in scene_rows})
    x = np.arange(len(scenes))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        y = []
        for scene in scenes:
            values = [row[metric] for row in scene_rows if row["scene_id"] == scene]
            y.append(np.var(values, ddof=1) if len(values) > 1 else 0.0)
        ax.bar(x, y)
        ax.set_ylabel(f"{metric} var")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(scenes, rotation=45, ha="right")
    axes[-1].set_xlabel("scene")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scene_mean_with_variance(scene_rows, metrics, output_path: Path):
    scenes = sorted({row["scene_id"] for row in scene_rows})
    x = np.arange(len(scenes))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        variances = []
        for scene in scenes:
            values = np.array(
                [row[metric] for row in scene_rows if row["scene_id"] == scene],
                dtype=np.float64,
            )
            means.append(values.mean())
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            stds.append(std)
            variances.append(std * std)

        means = np.array(means)
        stds = np.array(stds)
        variances = np.array(variances)

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=f"{metric} mean +/- std",
        )
        ax.fill_between(x, means - stds, means + stds, alpha=0.15)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

        ax_var = ax.twinx()
        ax_var.plot(
            x,
            variances,
            color="tab:red",
            marker="x",
            linestyle=":",
            linewidth=1,
            label="variance",
        )
        ax_var.set_ylabel("variance", color="tab:red")
        ax_var.tick_params(axis="y", labelcolor="tab:red")

        lines, labels = ax.get_legend_handles_labels()
        lines_var, labels_var = ax_var.get_legend_handles_labels()
        ax.legend(lines + lines_var, labels + labels_var, loc="best")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(scenes, rotation=45, ha="right")
    axes[-1].set_xlabel("scene")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("eval_results/mv_recon/ordering_exp/OVGGT"),
    )
    parser.add_argument("--dataset", default="7scenes")
    parser.add_argument("--expected-scenes", type=int, default=7)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.root / "stats")
    metrics = ["acc", "comp", "acc_med", "comp_med"]

    all_scene_rows = []
    run_rows = []
    skipped = []

    for run_dir in sorted(args.root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        rows = parse_run(run_dir, args.dataset)
        unique_scenes = sorted({row["scene_id"] for row in rows})
        if len(unique_scenes) < args.expected_scenes and not args.include_incomplete:
            skipped.append((run_dir.name, len(unique_scenes)))
            continue
        if not rows:
            skipped.append((run_dir.name, 0))
            continue

        all_scene_rows.extend(rows)
        run_mean = {"run": run_dir.name, "num_scenes": len(unique_scenes)}
        for metric in metrics:
            run_mean[metric] = float(np.mean([row[metric] for row in rows]))
        run_rows.append(run_mean)

    if not run_rows:
        raise SystemExit("No complete runs found.")

    summary_rows = []
    for metric in metrics:
        values = np.array([row[metric] for row in run_rows], dtype=np.float64)
        summary_rows.append(
            {
                "metric": metric,
                "num_runs": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "var": values.var(ddof=1) if len(values) > 1 else 0.0,
                "min": values.min(),
                "max": values.max(),
            }
        )

    write_csv(
        out_dir / "scene_metrics.csv",
        all_scene_rows,
        ["run", "scene_id", "acc", "comp", "acc_med", "comp_med"],
    )
    write_csv(
        out_dir / "run_means.csv",
        run_rows,
        ["run", "num_scenes", "acc", "comp", "acc_med", "comp_med"],
    )
    write_csv(
        out_dir / "summary.csv",
        summary_rows,
        ["metric", "num_runs", "mean", "std", "var", "min", "max"],
    )

    plot_run_means(run_rows, metrics, out_dir / "run_means.png")
    plot_scene_mean_with_variance(all_scene_rows, metrics, out_dir / "scene_mean_variance.png")
    plot_scene_variance(all_scene_rows, metrics, out_dir / "scene_variance.png")

    print(f"Used {len(run_rows)} runs.")
    if skipped:
        print("Skipped incomplete runs:")
        for run, count in skipped:
            print(f"  {run}: {count}/{args.expected_scenes} scenes")
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'run_means.png'}")
    print(f"Wrote {out_dir / 'scene_mean_variance.png'}")
    print(f"Wrote {out_dir / 'scene_variance.png'}")
    for row in summary_rows:
        print(
            f"{row['metric']}: mean={row['mean']:.6f}, "
            f"std={row['std']:.6f}, var={row['var']:.6f}, "
            f"min={row['min']:.6f}, max={row['max']:.6f}"
        )


if __name__ == "__main__":
    main()

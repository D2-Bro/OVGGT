#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


LAYER_RE = re.compile(r"^(?P<prefix>.*?)(?P<idx>\d+)$")


def layer_sort_key(layer: str):
    match = LAYER_RE.match(layer)
    if match is None:
        return layer, -1
    return match.group("prefix"), int(match.group("idx"))


def parse_jsonl(path: Path):
    ratios_by_layer = defaultdict(list)
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                print(f"Skipping malformed JSON at line {line_no}")
                continue

            layer = row.get("layer", "unlabeled")
            total_current = row.get("total_current", 0)
            evicted_current = row.get("evicted_current", [])

            if total_current <= 0:
                skipped += 1
                continue

            if isinstance(evicted_current, (int, float)):
                evicted_current = [evicted_current]

            for evicted in evicted_current:
                ratios_by_layer[layer].append(float(evicted) / float(total_current))

    return ratios_by_layer, skipped


def summarize(ratios_by_layer):
    rows = []
    for layer in sorted(ratios_by_layer, key=layer_sort_key):
        values = ratios_by_layer[layer]
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        rows.append(
            {
                "layer": layer,
                "count": count,
                "mean": mean,
                "variance": variance,
                "std": variance ** 0.5,
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "count", "mean", "variance", "std", "min", "max"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plt is None:
        write_svg_summary(rows, output_path.with_suffix(".svg"))
        return

    layers = [row["layer"] for row in rows]
    means = [row["mean"] for row in rows]
    variances = [row["variance"] for row in rows]
    stds = [row["std"] for row in rows]
    counts = [row["count"] for row in rows]
    x = list(range(len(layers)))

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(layers) * 0.5), 8), sharex=True)

    axes[0].bar(x, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.85)
    axes[0].set_ylabel("Eviction ratio mean")
    max_mean_with_std = max([mean + std for mean, std in zip(means, stds)] + [0.0])
    axes[0].set_ylim(0.0, min(1.0, max(1.0, max_mean_with_std * 1.08)))
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, variances, color="tab:red", alpha=0.8)
    axes[1].set_ylabel("Eviction ratio variance")
    axes[1].grid(True, axis="y", alpha=0.3)

    label_text = [f"{layer}\n(n={count})" for layer, count in zip(layers, counts)]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_text, rotation=45, ha="right")
    axes[1].set_xlabel("Layer")

    fig.suptitle("Current-token eviction ratio by layer")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_svg_summary(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [row["layer"] for row in rows]
    means = [row["mean"] for row in rows]
    variances = [row["variance"] for row in rows]
    counts = [row["count"] for row in rows]

    width = max(1100, 90 * len(rows))
    height = 760
    margin_left = 80
    margin_right = 30
    mean_top = 70
    plot_height = 230
    gap = 120
    var_top = mean_top + plot_height + gap
    axis_width = width - margin_left - margin_right
    bar_gap = 12
    bar_width = max(10, (axis_width - bar_gap * (len(rows) - 1)) / max(len(rows), 1))
    max_var = max(max(variances), 1e-12)

    def esc(text):
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def bar_rects(values, top, max_value, color):
        rects = []
        labels = []
        for idx, value in enumerate(values):
            x = margin_left + idx * (bar_width + bar_gap)
            bar_h = 0 if max_value <= 0 else (value / max_value) * plot_height
            y = top + plot_height - bar_h
            rects.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_h:.2f}" '
                f'fill="{color}" opacity="0.86" />'
            )
            labels.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" '
                f'font-size="11">{value:.4f}</text>'
            )
        return "\n".join(rects + labels)

    x_labels = []
    for idx, (layer, count) in enumerate(zip(layers, counts)):
        x = margin_left + idx * (bar_width + bar_gap) + bar_width / 2
        label = f"{layer} (n={count})"
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 58}" text-anchor="end" '
            f'font-size="11" transform="rotate(-45 {x:.2f} {height - 58})">{esc(label)}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2:.2f}" y="32" text-anchor="middle" font-size="22" font-family="sans-serif">Current-token eviction ratio by layer</text>

<text x="20" y="{mean_top + plot_height / 2:.2f}" transform="rotate(-90 20 {mean_top + plot_height / 2:.2f})" text-anchor="middle" font-size="14" font-family="sans-serif">mean</text>
<line x1="{margin_left}" y1="{mean_top + plot_height}" x2="{width - margin_right}" y2="{mean_top + plot_height}" stroke="#333" />
<line x1="{margin_left}" y1="{mean_top}" x2="{margin_left}" y2="{mean_top + plot_height}" stroke="#333" />
<text x="{margin_left - 10}" y="{mean_top + 4}" text-anchor="end" font-size="11">1.0</text>
<text x="{margin_left - 10}" y="{mean_top + plot_height}" text-anchor="end" font-size="11">0.0</text>
{bar_rects(means, mean_top, 1.0, "#1f77b4")}

<text x="20" y="{var_top + plot_height / 2:.2f}" transform="rotate(-90 20 {var_top + plot_height / 2:.2f})" text-anchor="middle" font-size="14" font-family="sans-serif">variance</text>
<line x1="{margin_left}" y1="{var_top + plot_height}" x2="{width - margin_right}" y2="{var_top + plot_height}" stroke="#333" />
<line x1="{margin_left}" y1="{var_top}" x2="{margin_left}" y2="{var_top + plot_height}" stroke="#333" />
<text x="{margin_left - 10}" y="{var_top + 4}" text-anchor="end" font-size="11">{max_var:.4f}</text>
<text x="{margin_left - 10}" y="{var_top + plot_height}" text-anchor="end" font-size="11">0.0</text>
{bar_rects(variances, var_top, max_var, "#d62728")}

{chr(10).join(x_labels)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
    print(f"matplotlib is not installed; wrote SVG fallback: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-layer current-token eviction ratio statistics from eviction_debug.jsonl."
    )
    parser.add_argument("--input", type=Path, default=Path("eviction_debug.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eviction_debug_stats.png"))
    parser.add_argument("--csv", type=Path, default=Path("eviction_debug_stats.csv"))
    args = parser.parse_args()

    ratios_by_layer, skipped = parse_jsonl(args.input)
    if not ratios_by_layer:
        raise SystemExit(f"No valid eviction records found in {args.input}")

    rows = summarize(ratios_by_layer)
    write_csv(args.csv, rows)
    plot_summary(rows, args.out)

    if plt is None:
        print(f"Wrote plot: {args.out.with_suffix('.svg')}")
    else:
        print(f"Wrote plot: {args.out}")
    print(f"Wrote CSV: {args.csv}")
    if skipped:
        print(f"Skipped records: {skipped}")


if __name__ == "__main__":
    main()

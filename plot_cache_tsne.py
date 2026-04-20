#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def load_dump(path: Path, batch_idx: int):
    dump = torch.load(path, map_location="cpu")
    features = dump["features"][batch_idx].float()
    frame_ids = dump["frame_ids"][batch_idx].long()
    token_ids = dump["token_ids"][batch_idx].long()
    return dump, features, frame_ids, token_ids


def standardize(features: torch.Tensor):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / (std + 1e-6)


def embed_features(features: torch.Tensor, method: str, perplexity: float, seed: int):
    features = standardize(features)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "scikit-learn is required for --method tsne. "
                "Install it or rerun with --method pca."
            ) from exc

        perplexity = min(perplexity, max(1.0, (features.shape[0] - 1) / 3))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(features.numpy())
        return torch.from_numpy(coords).float()

    if method == "pca":
        _, _, v = torch.pca_lowrank(features, q=2, center=False)
        return features @ v[:, :2]

    raise ValueError(f"Unknown method: {method}")


def frame_stats(features: torch.Tensor, frame_ids: torch.Tensor):
    rows = []
    for frame_id in sorted(frame_ids.unique().tolist()):
        mask = frame_ids == frame_id
        frame_features = features[mask]
        centroid = frame_features.mean(dim=0)
        distances = torch.linalg.norm(frame_features - centroid, dim=-1)
        rows.append(
            {
                "frame_id": int(frame_id),
                "count": int(frame_features.shape[0]),
                "intra_distance_mean": float(distances.mean()),
                "intra_distance_std": float(distances.std(unbiased=False)),
                "intra_distance_min": float(distances.min()),
                "intra_distance_max": float(distances.max()),
            }
        )
    return rows


def write_stats_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "count",
        "intra_distance_mean",
        "intra_distance_std",
        "intra_distance_min",
        "intra_distance_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frame_color(frame_id: int):
    hue = (frame_id * 0.61803398875) % 1.0
    return hsv_to_rgb(hue, 0.72, 0.88)


def hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def plot_with_matplotlib(coords, frame_ids, dump, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_frames = sorted(frame_ids.unique().tolist())
    colors = [tuple(channel / 255 for channel in frame_color(int(frame))) for frame in frame_ids.tolist()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.72, linewidths=0)

    for frame in unique_frames:
        mask = frame_ids == frame
        centroid = coords[mask].mean(dim=0)
        color = tuple(channel / 255 for channel in frame_color(int(frame)))
        ax.scatter(centroid[0], centroid[1], marker="x", s=80, c=[color], linewidths=2)
        ax.text(centroid[0], centroid[1], str(int(frame)), fontsize=8)

    ax.set_title(f"{dump.get('layer', 'unknown')} step={dump.get('step', 'unknown')}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_svg(coords, frame_ids, dump, output_path: Path):
    output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1000, 820
    margin = 60
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    def sx(value):
        return margin + (float(value) - x_min) / x_span * (width - 2 * margin)

    def sy(value):
        return height - margin - (float(value) - y_min) / y_span * (height - 2 * margin)

    points = []
    for coord, frame_id in zip(coords, frame_ids):
        r, g, b = frame_color(int(frame_id))
        points.append(
            f'<circle cx="{sx(coord[0]):.2f}" cy="{sy(coord[1]):.2f}" r="2.2" '
            f'fill="rgb({r},{g},{b})" opacity="0.72" />'
        )

    labels = []
    for frame in sorted(frame_ids.unique().tolist()):
        mask = frame_ids == frame
        centroid = coords[mask].mean(dim=0)
        r, g, b = frame_color(int(frame))
        cx, cy = sx(centroid[0]), sy(centroid[1])
        labels.append(
            f'<text x="{cx:.2f}" y="{cy:.2f}" font-size="11" '
            f'fill="rgb({r},{g},{b})">{int(frame)}</text>'
        )

    title = f"{dump.get('layer', 'unknown')} step={dump.get('step', 'unknown')}"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="30" text-anchor="middle" font-size="20">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />
{chr(10).join(points)}
{chr(10).join(labels)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
    print(f"matplotlib is not installed; wrote SVG fallback: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cached token key features colored by source frame."
    )
    parser.add_argument("--input", type=Path, required=True, help="A cache TSNE dump .pt file.")
    parser.add_argument("--out", type=Path, default=Path("cache_tsne.png"))
    parser.add_argument("--stats-csv", type=Path, default=Path("cache_tsne_frame_stats.csv"))
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dump, features, frame_ids, _ = load_dump(args.input, args.batch_idx)
    stats_rows = frame_stats(features, frame_ids)
    write_stats_csv(args.stats_csv, stats_rows)

    coords = embed_features(features, args.method, args.perplexity, args.seed)
    if plt is None:
        plot_svg(coords, frame_ids, dump, args.out)
    else:
        plot_with_matplotlib(coords, frame_ids, dump, args.out)
        print(f"Wrote plot: {args.out}")
    print(f"Wrote frame stats: {args.stats_csv}")


if __name__ == "__main__":
    main()

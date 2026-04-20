#!/usr/bin/env python3
import argparse
import csv
import math
import re
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


def standardize(features: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None):
    if mean is None:
        mean = features.mean(dim=0, keepdim=True)
    if std is None:
        std = features.std(dim=0, keepdim=True)
    return (features - mean) / (std + 1e-6)


def fit_first_frame_pca(features: torch.Tensor, frame_ids: torch.Tensor, frame_id: int = None):
    if frame_id is None:
        frame_id = int(frame_ids.min())

    mask = frame_ids == frame_id
    if not bool(mask.any()):
        raise ValueError(f"No tokens found for PCA reference frame {frame_id}")

    reference_features = features[mask]
    mean = reference_features.mean(dim=0, keepdim=True)
    std = reference_features.std(dim=0, keepdim=True)
    standardized_reference = standardize(reference_features, mean, std)
    _, _, v = torch.pca_lowrank(standardized_reference, q=2, center=False)
    return {
        "frame_id": frame_id,
        "mean": mean,
        "std": std,
        "basis": v[:, :2],
    }


def project_fixed_pca(features: torch.Tensor, projection):
    features = standardize(features, projection["mean"], projection["std"])
    return features @ projection["basis"]


def embed_features(features: torch.Tensor, method: str, perplexity: float, seed: int, projection=None):
    if method == "pca" and projection is not None:
        return project_fixed_pca(features, projection)

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


def collect_input_paths(paths, max_step: int = None):
    input_paths = []
    for path in paths:
        if path.is_dir():
            input_paths.extend(sorted(path.glob("step_*.pt")))
        else:
            input_paths.append(path)

    input_paths = sorted(input_paths, key=step_sort_key)
    if max_step is not None:
        input_paths = [
            path for path in input_paths
            if step_number(path) is None or step_number(path) <= max_step
        ]
    if not input_paths:
        raise SystemExit("No input .pt files found.")
    return input_paths


def step_number(path: Path):
    match = re.search(r"step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def step_sort_key(path: Path):
    step = step_number(path)
    if step is not None:
        return step, path.name
    return math.inf, path.name


def output_stem(method: str, dump, input_path: Path):
    step = dump.get("step")
    if step is not None:
        return f"cache_{method}_step_{int(step):06d}"
    return f"cache_{method}_{input_path.stem}"


def reference_label(path: Path, dump):
    step = dump.get("step")
    if step is not None:
        return f"{path.name}:frame"
    return f"{path}:frame"


def axis_limits(coords_by_path):
    mins = []
    maxes = []
    for coords in coords_by_path.values():
        mins.append(coords.min(dim=0).values)
        maxes.append(coords.max(dim=0).values)

    xy_min = torch.stack(mins).min(dim=0).values
    xy_max = torch.stack(maxes).max(dim=0).values
    span = xy_max - xy_min
    pad = torch.clamp(span * 0.05, min=1e-6)
    return xy_min - pad, xy_max + pad


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


def plot_with_matplotlib(coords, frame_ids, dump, output_path: Path, limits=None, title_suffix=""):
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

    if limits is not None:
        xy_min, xy_max = limits
        ax.set_xlim(float(xy_min[0]), float(xy_max[0]))
        ax.set_ylim(float(xy_min[1]), float(xy_max[1]))

    ax.set_title(f"{dump.get('layer', 'unknown')} step={dump.get('step', 'unknown')}{title_suffix}")
    ax.set_xlabel("fixed PCA 1")
    ax.set_ylabel("fixed PCA 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_svg(coords, frame_ids, dump, output_path: Path, limits=None, title_suffix=""):
    output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1000, 820
    margin = 60
    x = coords[:, 0]
    y = coords[:, 1]
    if limits is None:
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
    else:
        xy_min, xy_max = limits
        x_min, x_max = float(xy_min[0]), float(xy_max[0])
        y_min, y_max = float(xy_min[1]), float(xy_max[1])
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

    title = f"{dump.get('layer', 'unknown')} step={dump.get('step', 'unknown')}{title_suffix}"
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
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more cache TSNE dump .pt files, or directories containing step_*.pt.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Single-file output path.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stats-csv", type=Path, default=None, help="Single-file stats CSV path.")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--method", choices=["tsne", "pca"], default="pca")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pca-reference-frame",
        type=int,
        default=None,
        help="Frame id used to fit the fixed PCA basis. Defaults to the first frame id in the first step.",
    )
    parser.add_argument(
        "--pca-reference-input",
        type=Path,
        default=None,
        help="Dump .pt file used to fit the fixed PCA basis. Defaults to the first selected input.",
    )
    parser.add_argument("--max-step", type=int, default=None, help="Only plot step_*.pt files up to this step.")
    args = parser.parse_args()

    input_paths = collect_input_paths(args.input, max_step=args.max_step)
    if len(input_paths) > 1 and args.method != "pca":
        raise SystemExit("Multiple inputs are supported only with --method pca, because t-SNE has no fixed axes.")

    torch.manual_seed(args.seed)
    reference_input = args.pca_reference_input or input_paths[0]
    if reference_input.is_dir():
        reference_paths = collect_input_paths([reference_input], max_step=args.max_step)
        reference_input = reference_paths[0]

    reference_dump, reference_features, reference_frame_ids, _ = load_dump(reference_input, args.batch_idx)
    projection = None
    title_suffix = ""
    if args.method == "pca":
        projection = fit_first_frame_pca(
            reference_features,
            reference_frame_ids,
            frame_id=args.pca_reference_frame,
        )
        title_suffix = f" fixed PCA {reference_label(reference_input, reference_dump)}={projection['frame_id']}"

    dumps_by_path = {}
    frame_ids_by_path = {}
    coords_by_path = {}
    stats_paths_by_path = {}

    for input_path in input_paths:
        dump, features, frame_ids, _ = load_dump(input_path, args.batch_idx)
        coords = embed_features(features, args.method, args.perplexity, args.seed, projection=projection)
        stem = output_stem(args.method, dump, input_path)
        if len(input_paths) == 1:
            stats_path = args.stats_csv if args.stats_csv is not None else args.out_dir / f"{stem}_stats.csv"
        else:
            stats_path = args.out_dir / f"{stem}_stats.csv"
        write_stats_csv(stats_path, frame_stats(features, frame_ids))

        dumps_by_path[input_path] = dump
        frame_ids_by_path[input_path] = frame_ids
        coords_by_path[input_path] = coords
        stats_paths_by_path[input_path] = stats_path

    limits = axis_limits(coords_by_path) if args.method == "pca" else None

    for input_path in input_paths:
        dump = dumps_by_path[input_path]
        frame_ids = frame_ids_by_path[input_path]
        coords = coords_by_path[input_path]
        stem = output_stem(args.method, dump, input_path)

        if len(input_paths) == 1:
            output_path = args.out if args.out is not None else args.out_dir / f"{stem}.png"
            stats_path = args.stats_csv if args.stats_csv is not None else args.out_dir / f"{stem}_stats.csv"
        else:
            output_path = args.out_dir / f"{stem}.png"
        stats_path = stats_paths_by_path[input_path]

        if plt is None:
            written_path = plot_svg(coords, frame_ids, dump, output_path, limits=limits, title_suffix=title_suffix)
        else:
            plot_with_matplotlib(coords, frame_ids, dump, output_path, limits=limits, title_suffix=title_suffix)
            written_path = output_path
            print(f"Wrote plot: {written_path}")
        print(f"Wrote frame stats: {stats_path}")


if __name__ == "__main__":
    main()

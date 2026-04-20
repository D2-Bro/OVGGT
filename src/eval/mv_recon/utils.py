import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch
import os


def _knn_cpu(query_points, ref_points):
    ref_points_kd_tree = KDTree(ref_points)
    workers = int(os.environ.get("MV_RECON_KNN_CPU_WORKERS", "-1"))
    return ref_points_kd_tree.query(query_points, workers=workers)


def _knn_gpu(query_points, ref_points):
    query_points = np.asarray(query_points, dtype=np.float32)
    ref_points = np.asarray(ref_points, dtype=np.float32)

    if query_points.size == 0 or ref_points.size == 0:
        return _knn_cpu(query_points, ref_points)

    if not torch.cuda.is_available():
        return _knn_cpu(query_points, ref_points)

    query_chunk = int(os.environ.get("MV_RECON_KNN_QUERY_CHUNK", "2048"))
    ref_chunk = int(os.environ.get("MV_RECON_KNN_REF_CHUNK", "32768"))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    try:
        ref_tensor = torch.as_tensor(ref_points, dtype=torch.float32, device=device)
        distances_all = []
        indices_all = []

        with torch.no_grad():
            for q_start in range(0, query_points.shape[0], query_chunk):
                q_end = min(q_start + query_chunk, query_points.shape[0])
                query_tensor = torch.as_tensor(
                    query_points[q_start:q_end], dtype=torch.float32, device=device
                )
                best_dist = torch.full(
                    (q_end - q_start,), float("inf"), dtype=torch.float32, device=device
                )
                best_idx = torch.zeros(
                    (q_end - q_start,), dtype=torch.long, device=device
                )

                for r_start in range(0, ref_tensor.shape[0], ref_chunk):
                    r_end = min(r_start + ref_chunk, ref_tensor.shape[0])
                    distances = torch.cdist(query_tensor, ref_tensor[r_start:r_end])
                    chunk_dist, chunk_idx = torch.min(distances, dim=1)
                    update_mask = chunk_dist < best_dist
                    best_dist = torch.where(update_mask, chunk_dist, best_dist)
                    best_idx = torch.where(
                        update_mask, chunk_idx + r_start, best_idx
                    )

                distances_all.append(best_dist.cpu().numpy())
                indices_all.append(best_idx.cpu().numpy())

        return np.concatenate(distances_all), np.concatenate(indices_all)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _knn_cpu(query_points, ref_points)


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    distances, _ = _knn_gpu(gt_points, rec_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    distances, idx = _knn_gpu(rec_points, gt_points)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    distances, idx = _knn_gpu(gt_points, rec_points)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou

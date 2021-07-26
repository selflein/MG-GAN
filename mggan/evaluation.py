from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from mggan.manifold import Manifold
from mggan.metrics import compute_metrics_from_batch
from mggan.utils import to_numpy
from mggan.visualization import plot_trajectories, plot_trajectories_by_idxs


def adjust_seq_start_end_for_mask(seq_start_end, remove_mask):
    assert seq_start_end[-1][1] == len(remove_mask)
    offsets = [
        0,
    ] + np.cumsum(remove_mask).tolist()

    new_seq = []
    for start, end in seq_start_end:
        new_start = start - offsets[start]
        new_end = end - offsets[end]
        new_seq.append((new_start, new_end))

    assert new_seq[-1][1] == np.sum(~remove_mask)
    return new_seq


def get_same_obs_indices(eval_ds):
    obs_trajs = to_numpy(eval_ds.obs_traj)
    seq_start_end = eval_ds.seq_start_end

    same_scene_dict = defaultdict(list)
    for scene_idx, (start, end) in enumerate(seq_start_end):
        scene_key = tuple(
            [*obs_trajs[start:end].reshape(-1).tolist(), eval_ds.scene_list[scene_idx]]
        )
        same_scene_dict[scene_key].append([i for i in range(start, end)])
    return list(same_scene_dict.values())


def evaluate_ade_fde(eval_ds, preds, n_preds_list):
    gt_trajs = to_numpy(eval_ds.pred_traj)
    seq_start_end = eval_ds.seq_start_end

    pred_mask = np.isnan(gt_trajs).any(-1).any(-1)
    start_end = adjust_seq_start_end_for_mask(seq_start_end, pred_mask)
    gt_trajs = gt_trajs[~pred_mask]
    preds = preds[:, :, ~pred_mask]

    metrics_accum = defaultdict(lambda: np.zeros((2,)))
    for scene_idx, (start, end) in enumerate(tqdm(start_end, desc="Eval ADE/FDE")):
        if start == end:
            continue

        scaling = 1.0

        # Convert to pixels for these datasets
        if eval_ds.dataset_name in ("stanford", "gofp"):
            scaling = 1.0 / eval_ds.images[eval_ds.scene_list[scene_idx]]["ratio"]

        for n_preds in n_preds_list:
            other_metrics = compute_metrics_from_batch(
                torch.from_numpy(preds[:, :n_preds, start:end]) * scaling,
                torch.from_numpy(gt_trajs[start:end]).transpose(0, 1) * scaling,
                [[0, end - start]],
                None,
                "raw",
            )

            for key, (value, count) in other_metrics.items():
                metrics_accum[f"{key} k={n_preds}"] += value, count

    avg_metrics = defaultdict(float)
    for key, (value, count) in metrics_accum.items():
        avg_metrics[key] = value / count
    return avg_metrics


def get_oracle_preds(eval_ds, num_preds):
    gt_trajs = to_numpy(eval_ds.pred_traj)
    ds_size, pred_len, _ = gt_trajs.shape

    same_scenes_indices = get_same_obs_indices(eval_ds)

    preds = np.zeros((ds_size, num_preds, pred_len, 2))
    for same_scene_indices in same_scenes_indices:
        for same_ped_indices in zip(*same_scene_indices):
            for ped in same_ped_indices:
                ped_pred_idxs = np.random.choice(same_ped_indices, size=num_preds)
                # (num_preds, pred_len, 2)
                ped_preds = gt_trajs[ped_pred_idxs]
                preds[ped] = ped_preds

    # (ds size, num_samples, pred_len, 2)
    # -> (pred_len, num_samples, ds size, 2)
    return preds.transpose(2, 1, 0, 3)


def evaluate_precision_recall(
    eval_ds, all_preds, manifold_radius, n_preds_list, debug=False
):
    gt_trajs = to_numpy(eval_ds.pred_traj)
    num_preds = max(n_preds_list)

    same_scenes_indices = get_same_obs_indices(eval_ds)

    pred_mask = np.isnan(gt_trajs).any(-1).any(-1)
    not_pred_mask_indices = np.where(~pred_mask)[0]

    # (pred_len, num_samples, batch_size, 2)
    # -> (batch_size, num_samples, pred_len, 2)
    preds = all_preds.transpose(2, 1, 0, 3)

    metrics_accum = defaultdict(lambda: np.zeros((2,)))
    for same_scene_indices in same_scenes_indices:
        for same_ped_indices in zip(*same_scene_indices):
            same_ped_indices = np.array(same_ped_indices)
            same_ped_indices = np.intersect1d(same_ped_indices, not_pred_mask_indices)
            if len(same_ped_indices) == 0:
                continue

            gt_man_samples = gt_trajs[same_ped_indices]
            gt_man = Manifold(gt_man_samples, manifold_radius)

            # (len(same_ped_indices) * n_samples, pred_len, 2)
            cur_preds = preds[same_ped_indices].reshape(-1, *preds.shape[2:])

            precision = gt_man.compute_metric(cur_preds[:num_preds])
            metrics_accum["Precision"] += precision, 1.0

            if debug:
                obs_trajs = to_numpy(eval_ds.obs_traj)
                traj_idx = same_ped_indices[0]

                inside_man = gt_man.compute_inside(cur_preds)
                plot_trajectories_by_idxs(
                    obs_trajs[traj_idx], None, cur_preds, inside_man.astype(np.int)
                )
                plt.show()
                plt.close()

                plot_trajectories(obs_trajs[traj_idx], None, gt_man_samples)
                plt.show()
                plt.close()

            for n_samples in n_preds_list:
                pred_man = Manifold(cur_preds[:n_samples], manifold_radius)
                recall = pred_man.compute_metric(gt_man_samples)
                metrics_accum[f"Recall k={n_samples}"] += recall, 1.0

    avg_metrics = defaultdict(float)
    for key, (value, count) in metrics_accum.items():
        avg_metrics[key] = value / count
    return avg_metrics

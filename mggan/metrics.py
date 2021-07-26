import torch
import numpy as np
import torch.nn.functional as F


def min_scene_error(error, seq_start_end):
    """Get the minimum prediction error for an entire scene prediction.

    Args:
        error: Tensor of shape (num predictions, b).
        seq_start_end: Indexes of the scene along the batch dimension of shape
            (n, 2).
    Returns:
        Minimum sum of prediction error for each scene within the batch.
    """
    min_scene_errors_sum = 0
    for start, end in seq_start_end:
        err = error[:, start:end]
        scene_err = torch.sum(err, 1)
        min_scene_err = scene_err.min(0)[0].item()
        min_scene_errors_sum += min_scene_err
    return min_scene_errors_sum


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


def compute_pred_diversity(preds):
    """Compute the diversity of predictions. To do this, the mean cosine
    similarity is computed for every combination of flattened vectors of
    the relative predictions.

    Args:
        preds: Tensor of shape (pred_len, num_preds, 2)

    Returns:
        Float in the range [0, 1] representing the diversity of samples.
    """
    num_preds = preds.size(1)
    preds = preds.permute(1, 0, 2).reshape(num_preds, -1)

    cos_sims = []
    for i in range(num_preds):
        pred = preds[i]
        indices = torch.arange(num_preds)
        indices_except_i = indices[indices != i]
        cos_sims.append(
            F.cosine_similarity(
                pred.repeat(num_preds - 1, 1), preds[indices_except_i]
            ).mean()
        )

    return 1 - torch.mean(torch.tensor(cos_sims))


def compute_metrics_from_batch(
    preds,
    gt,
    sub_batches,
    mode="mean",
    mode_thresh=3.0,
):
    """Compute all metrics for a single batch of predictions.

    Args:
        preds: Tensor of shape (pred_len, k, b, 2)
        gt: Tensor of shape (pred_len, b, 2)
        sub_batches: List with shape (n, 2) denoting start and end of a scene
            within a batch.
        mode: One of `mean` or `raw`.

    Returns:
        Dictionary with all metrics.
    """
    pred_len, k, b, _ = preds.shape
    metrics = {}

    ades, fdes = [], []
    for i in range(preds.size(1)):
        # Compute sums for numerical stability when taking the mean
        ades.append(displacement_error(preds[:, i], gt, mode="raw"))
        fdes.append(final_displacement_error(preds[-1, i], gt[-1], mode="raw"))

    ades = torch.stack(ades)
    fdes = torch.stack(fdes)
    min_fde = min_scene_error(fdes, sub_batches)
    min_ade = min_scene_error(ades, sub_batches)
    metrics["FDE"] = np.array([min_fde, b])
    metrics["ADE"] = np.array([min_ade, pred_len * b])

    # Compute percentage of batch having a prediction hitting the
    # correct mode, i.e., minFDE lower than 2 meters
    percent_mode = (fdes.min(0)[0] < mode_thresh).float().sum()
    metrics["Mode"] = np.array([percent_mode.item(), b])

    if mode == "mean":
        return {k: (v / c) for k, (v, c) in metrics.items()}
    return metrics

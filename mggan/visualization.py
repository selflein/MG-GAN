import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe


def plot_trajectories(
    history=None, gt=None, preds=None, img=None, scale=20.0, ax=None, lw=3
):
    """Plots the groundtruth together with predictions.

    Args:
        gt: Numpy array of shape (x, 2) where x is the length of the trajectory.
        preds: Numpy array of shape (n, x, 2) where n is the number of
            predictions and x is the length of the prediction, or (c, n, x, 2)
            where each of the c (n, x, 2) arrays is plotted with a different
            color.
        history: Numpy array of shape (x, 2) where x is the length of
            the history for which the predictions were done.

    Returns:
        fig: Matplotlib figure with the plotted groundtruth and predictions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = None

    if img is not None:
        ax.imshow(img)
        height, width, _ = img.shape
        ax.set_ylim(height, 0)
        ax.set_xlim(0, width)
    else:
        ax.set_aspect("equal", adjustable="datalim")

    cmap = plt.get_cmap("tab10")
    if preds is not None:
        preds = preds[np.newaxis, :] if len(preds.shape) == 3 else preds

        for c, pred_group in enumerate(preds):
            last_inp = np.repeat(
                history[-1][np.newaxis, np.newaxis, :], len(pred_group), axis=0
            )
            pred_group = np.concatenate([last_inp, pred_group], axis=1)
            color = cmap(c)
            for pred in pred_group:
                scaled_pred = pred * scale
                ax.plot(scaled_pred[:, 0], scaled_pred[:, 1], color=color, linewidth=lw)
                ax.scatter(
                    scaled_pred[-1:, 0],
                    scaled_pred[-1:, 1],
                    c=np.array([color]),
                    s=lw * 7,
                )

    if gt is not None and not np.isnan(gt).any():
        gt = np.concatenate([history[-1][np.newaxis], gt])
        scaled_gt = gt * scale
        ax.plot(scaled_gt[:, 0], scaled_gt[:, 1], "k")
        ax.plot(scaled_gt[-1, 0], scaled_gt[-1, 1], "ko")

    if history is not None:
        scaled_hist = history * scale
        ax.plot(scaled_hist[:, 0], scaled_hist[:, 1], "k", linewidth=lw)
        ax.scatter(scaled_hist[-1:, 0], scaled_hist[-1:, 1], c="k", s=lw * 7)

    return fig, ax


def plot_scene(
    in_xy=None, gt_xy=None, pred_xy=None, img=None, scale=1.0, ax=None, lw=3
):
    """Plot entire scene with trajectories.

    `n` is the number of pedestrians in a scene, `c` number of generators and
    `x` is trajectory length (usually 8 for history  and 12 for future).

    Args:
        in_xy: Numpy array of shape (n, x, 2)
        pred_xy: Numpy array of shape (n, c, m, x, 2)
        gt_xy: Numpy array of shape (n, x, 2)
        img: Image to plot on.
        scale: Transformation from trajectory coordinates to image coordinates.

    Returns:
        Tuple with figure and axes of the produced plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig, ax = ax.get_figure(), ax

    if img is not None:
        ax.imshow(img)
    else:
        ax.set_aspect("equal", adjustable="datalim")

    nums = [trj.shape[0] for trj in [in_xy, gt_xy, pred_xy] if trj is not None]
    assert len(set(nums)) == 1, "First dimension has to be the same for inputs."

    for j in range(nums[0]):
        hist = in_xy[j] if in_xy is not None else None
        gt = gt_xy[j] if gt_xy is not None else None
        pred = pred_xy[j] if pred_xy is not None else None
        plot_trajectories(hist, gt, pred, img=None, scale=scale, ax=ax, lw=lw)

    return fig, ax


def plot_trajectories_by_idxs(
    history=None,
    gt=None,
    preds=None,
    idxs=None,
    img=None,
    scale=20.0,
    ax=None,
    lw=3,
    ls="-",
    plot_hist=True,
    hist_color="k",
):
    """Plots the groundtruth together with predictions.

    Args:
        gt: Numpy array of shape (x, 2) where x is the length of the trajectory.
        preds: Numpy array of shape (n, x, 2) where n is the number of
            predictions and x is the length of the prediction.
        color_idx: Numpy array of shape (n,) which contains indexes which of the
            `n` predicts to draw with the same color.
        history: Numpy array of shape (x, 2) where x is the length of
            the history for which the predictions were done.

    Returns:
        fig: Matplotlib figure with the plotted groundtruth and predictions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = None

    if img is not None:
        ax.imshow(img, alpha=1)
        height, width, _ = img.shape
        ax.set_ylim(height, 0)
        ax.set_xlim(0, width)
    else:
        ax.set_aspect("equal", adjustable="datalim")

    outline = mpe.withStroke(linewidth=lw + 2, foreground="black")
    cmap = plt.get_cmap("tab10")
    if preds is not None:
        for c, idx in enumerate(np.unique(idxs)):
            idx_mask = idxs == idx
            pred_group = preds[idx_mask]

            last_inp = np.repeat(
                history[-1][np.newaxis, np.newaxis, :], len(pred_group), axis=0
            )
            pred_group = np.concatenate([last_inp, pred_group], axis=1)
            color = cmap(idx)
            for pred in pred_group:
                scaled_pred = pred * scale
                ax.plot(
                    scaled_pred[:, 0],
                    scaled_pred[:, 1],
                    color=color,
                    linewidth=lw,
                    path_effects=[outline],
                    zorder=1,
                )
                ax.scatter(
                    scaled_pred[-1:, 0],
                    scaled_pred[-1:, 1],
                    color=np.array([color]),
                    s=lw * 20,
                    zorder=2,
                    edgecolor="black",
                )

    if gt is not None and not np.isnan(gt).any():
        gt = np.concatenate([history[-1][np.newaxis], gt])
        scaled_gt = gt * scale
        ax.plot(scaled_gt[:, 0], scaled_gt[:, 1], "k", zorder=5)
        ax.plot(scaled_gt[-1, 0], scaled_gt[-1, 1], "ko", zorder=6)

    # outline = mpe.withStroke(linewidth=lw + 3, foreground="white")
    if history is not None and plot_hist:
        scaled_hist = history * scale
        ax.plot(
            scaled_hist[:, 0],
            scaled_hist[:, 1],
            hist_color,
            linewidth=lw,
            zorder=3,
            ls=ls,
        )
        ax.scatter(
            scaled_hist[-1:, 0],
            scaled_hist[-1:, 1],
            c=hist_color,
            s=lw * 15,
            zorder=4,
            path_effects=[outline],
        )

    return fig, ax


def plot_trajectories_by_idxs_scene(
    sub_batches,
    history=None,
    gt=None,
    preds=None,
    idxs=None,
    img=None,
    scale=20.0,
    ax=None,
    lw=3,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = None

    for start, end in sub_batches:
        for idx in range(start, end):
            plot_trajectories_by_idxs(
                history[:, idx] if history is not None else None,
                gt[:, idx] if gt is not None else None,
                preds[idx] if preds is not None else None,
                img=img,
                idxs=idxs[idx] if idxs is not None else None,
                scale=scale,
                lw=lw,
                ax=ax,
            )

    return fig, ax


if __name__ == "__main__":
    s = np.arange(0, 10, 1)
    groundtruth = s[:, np.newaxis].repeat(2, axis=1)
    predictions = groundtruth[np.newaxis].repeat(3, axis=0) + 9
    for i in range(len(predictions)):
        predictions[i, :, 1] -= i
    figure = plot_trajectories(groundtruth, predictions)
    plt.show()

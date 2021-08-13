from collections import defaultdict

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import autograd


def noise_loss(model, noise_sampler, alpha):
    loss = 0.0
    for p, n in zip(model.parameters(), noise_sampler):
        n.normal_(mean=0, std=alpha)
        loss += torch.sum(p * n)
    return loss


def get_gan_labels(shape, smoothness=0.1):
    label_fake = (torch.zeros(shape) + np.random.uniform(0, smoothness)).requires_grad_(
        False
    )
    label_real = (
        torch.ones(shape) * np.random.uniform(1 - smoothness, 1.0)
    ).requires_grad_(False)
    return label_real, label_fake


def get_sghmc_noise(model, gpu=False):
    if gpu:
        return [torch.zeros(p.size(), device="cuda") for p in model.parameters()]
    return [torch.zeros(p.size()) for p in model.parameters()]


def to_numpy(x):
    return x.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_gradient_penalty(
    netD, in_xy, in_dxdy, gt_xy, gt_dxdy, pred_xy, pred_dxdy, gp_lambda=10
):
    assert len(gt_xy) == len(pred_xy)

    alpha = torch.rand(1, in_xy.size(1), 1)

    inter_xy = (alpha * gt_xy + ((1 - alpha) * pred_xy)).requires_grad_()
    inter_dxdy = (alpha * gt_dxdy + ((1 - alpha) * pred_dxdy)).requires_grad_()
    in_xy_grad = in_xy.clone().requires_grad_()
    in_dxdy_grad = in_dxdy.clone().requires_grad_()
    with torch.backends.cudnn.flags(enabled=False):
        inputs = in_xy_grad, in_dxdy_grad, inter_xy, inter_dxdy
        disc_interpolates = netD(*inputs)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=inputs,
            grad_outputs=torch.ones(disc_interpolates.size()),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_traj_4d(obsv_p, pred_p=None):
    """Augment tensors of positions into positions+velocity"""
    obsv_v = obsv_p[:, 1:] - obsv_p[:, :-1]
    obsv_v = torch.cat([obsv_v[:, 0:1, :], obsv_v], dim=1)
    if pred_p is None:
        return obsv_p, obsv_v
    pred_p_1 = torch.cat([obsv_p[:, -1].unsqueeze(1), pred_p[:, :-1]], dim=1)
    pred_v = pred_p - pred_p_1
    return obsv_p, obsv_v, pred_p, pred_v


def load_hparams_from_tags_csv(tags_csv):
    from argparse import Namespace
    import pandas as pd

    tags_df = pd.read_csv(tags_csv)
    dic = tags_df.to_dict(orient="records")

    ns_dict = {row["key"]: convert(row["value"]) for row in dic}

    return ns_dict


def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def convert(val):
    constructors = [int, float, str]

    if type(val) is str:
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False

    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass
    return val


def make_mlp(dim_list, activation="relu", batch_norm=False, dropout=0):
    layers = []
    if len(dim_list) > 2:
        for dim_in, dim_out in zip(dim_list[:-2], dim_list[1:-1]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
    model = nn.Sequential(*layers)
    return model


def gan_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape)
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_global_noise(dim, sub_batches, noise_type):
    noise = []
    for start, end in sub_batches:
        n = gan_noise((1, dim), noise_type)
        noise.append(n.repeat(end - start, 1))
    return torch.cat(noise)


class GradNormLogger:
    def __init__(self):
        self.grad_norms = defaultdict(list)

    def update(self, model: torch.nn.Module, norm_type: float = 2.0):
        total_norm = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    module_name = name.split(".")[0]
                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    self.grad_norms[module_name].append(grad)
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1.0 / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        self.grad_norms["grad_norm_total"].append(grad)

    def reset(self):
        self.grad_norms = defaultdict(list)

    def write(self, writer, global_step):
        for module, grads in self.grad_norms.items():
            writer.add_histogram(
                f"gradient_histograms/{module}", np.array(grads), global_step
            )


def get_versions(logs_dir):
    versions = []
    for version in logs_dir.iterdir():
        if not version.is_dir() or "version" not in version.name:
            continue

        chkpt_dir = version / "checkpoints"
        if not chkpt_dir.is_dir() or len(list(chkpt_dir.iterdir())) == 0:
            continue

        if not (version / "meta_tags.csv").is_file():
            continue

        version_number = int(version.stem.split("_")[1])
        if (version / "checkpoints" / "checkpoint_best.pth").is_file():
            checkpoint = "best"
            versions.append((version_number, checkpoint))
        else:
            checkpoints = []
            for checkpoint in (version / "checkpoints").iterdir():
                idx = checkpoint.stem.split("_")[1]
                try:
                    idx = int(idx)
                except Exception:
                    continue
                checkpoints.append(idx)
            if len(checkpoints) > 0:
                checkpoint = max(checkpoints)
                versions.append((version_number, checkpoint))
    return versions


def get_selection_indices(sampled_gen_idxs):
    """
    Get array which contains the count for how many times a particular
    element has already been seen in the row at each position
    e.g. [1, 2, 3, 1] -> [0, 0, 0, 1]
    """
    batch_size = sampled_gen_idxs.size(0)
    sample_offset = torch.zeros_like(sampled_gen_idxs)
    for b in range(batch_size):
        _, inverse, counts = torch.unique(
            sampled_gen_idxs[b], return_inverse=True, return_counts=True
        )
        for i, count in enumerate(counts):
            sample_offset[b, inverse == i] += torch.arange(count)
    return sample_offset


def pandas_to_latex(df_table, index=True, multicolumn=False, **kwargs) -> None:
    latex = df_table.to_latex(multicolumn=multicolumn, index=index, **kwargs)

    if multicolumn:
        latex_lines = latex.splitlines()

        insert_line_counter = 0
        for j, level in enumerate(df_table.columns.levels[:-1]):
            midrule_str = ""
            codes = np.array(df_table.columns.codes[j])
            indices = np.nonzero(codes[:-1] != codes[1:])[0].tolist()

            for start, end in zip([-1] + indices, indices + [len(codes) - 1]):
                if end - start > 1:
                    hstart = start + 2
                    hend = end + 1
                    midrule_str += f"\cmidrule(l){{{hstart}-{hend}}} "

            latex_lines.insert(3 + insert_line_counter, midrule_str)
            insert_line_counter += j + 2
        latex = "\n".join(latex_lines)

    return latex


def get_similar_trajectories(
    eval_ds,
    distance_threshold,
    direction_threshold,
    speed_threshold,
    radius=2,
    filter_hist_colliding=False,
):
    obs = to_numpy(eval_ds.obs_traj)
    obs_rel = to_numpy(eval_ds.obs_traj_rel)
    gt_trajs = to_numpy(eval_ds.pred_traj)
    ped_ids = eval_ds.ped_ids
    framerate = 1 / eval_ds.time_step

    mask = np.isnan(gt_trajs).any(-1).any(-1)

    scene_to_traj_idxs = defaultdict(list)
    traj_idx_to_other = {}
    for i, (start, end) in enumerate(eval_ds.seq_start_end):
        scene_to_traj_idxs[eval_ds.scene_list[i]].extend(list(range(start, end)))
        for j in range(start, end):
            traj_idx_to_other[j] = [id_ for id_ in range(start, end) if id_ != j]

    length_last_offset_vector = np.linalg.norm(
        obs_rel[:, -1], ord=2, axis=1, keepdims=True
    )
    last_offset_vector = obs_rel[:, -1] / (length_last_offset_vector + 1e-7)

    similar_trajs_dict = {}
    for i, (start, end) in enumerate(tqdm(eval_ds.seq_start_end)):
        for idx in range(start, end):
            if idx in similar_trajs_dict.keys():
                continue

            if mask[idx]:
                continue

            remaining_idxs = np.array(scene_to_traj_idxs[eval_ds.scene_list[i]])
            remaining_idxs = remaining_idxs[~mask[remaining_idxs]]

            distances = np.linalg.norm(
                obs[idx][None, -1] - obs[remaining_idxs, -1], ord=2, axis=-1
            )
            remaining_idxs = remaining_idxs[distances < distance_threshold]

            if not np.isclose(length_last_offset_vector[idx], 0.0):
                cos_sim = np.dot(
                    last_offset_vector[idx],
                    last_offset_vector[remaining_idxs].transpose(1, 0),
                )
                remaining_idxs = remaining_idxs[cos_sim > direction_threshold]

            speed = (
                np.abs(
                    length_last_offset_vector[idx]
                    - length_last_offset_vector[remaining_idxs, 0]
                )
                * framerate
            )
            remaining_idxs = remaining_idxs[speed < speed_threshold]

            remaining_ped_ids = ped_ids[remaining_idxs]
            for idx_similar in remaining_idxs:
                # Only take closest trajectory for one pedestrian ID
                idxs = [idx_similar]
                other_ped = remaining_ped_ids != ped_ids[idx_similar]
                cur_remaining_ped_idxs = remaining_ped_ids[other_ped]
                cur_remaining_idxs = remaining_idxs[other_ped]
                for id_ in np.unique(cur_remaining_ped_idxs):
                    idxs_for_ped_id = cur_remaining_idxs[cur_remaining_ped_idxs == id_]
                    distances = np.linalg.norm(
                        obs[idx_similar][None, -1] - obs[idxs_for_ped_id, -1],
                        ord=2,
                        axis=-1,
                    )
                    idxs.append(idxs_for_ped_id[np.argmin(distances)])
                cur_remaining_idxs = np.array(idxs)

                offset = obs[cur_remaining_idxs, -1] - obs[idx_similar, -1]
                selected_gt_trajs = gt_trajs[cur_remaining_idxs] - offset[:, None]

                # Filter groundtruths which collide
                if filter_hist_colliding:
                    other_traj_ids_in_scene = traj_idx_to_other[idx_similar]

                    if len(other_traj_ids_in_scene) > 0:
                        do_collide_accum = []
                        for other_ped_gt in gt_trajs[other_traj_ids_in_scene]:
                            d = np.linalg.norm(
                                selected_gt_trajs - other_ped_gt[None],
                                ord=2,
                                axis=-1,
                            )
                            do_collide_accum.append((d < radius).any(1))
                        do_collide = np.stack(do_collide_accum).any(0)
                        selected_gt_trajs = selected_gt_trajs[~do_collide]

                similar_trajs_dict[idx_similar] = selected_gt_trajs

    return similar_trajs_dict

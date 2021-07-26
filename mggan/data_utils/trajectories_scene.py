import copy
import math
import logging
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

from mggan.data_utils.BaseTrajectories import BaseDataset

logger = logging.getLogger(__name__)


def rotate(X, center, alpha):
    XX = X.copy()

    XX[:, 0] = (
        (X[:, 0] - center[0]) * np.cos(alpha)
        + (X[:, 1] - center[1]) * np.sin(alpha)
        + center[0]
    )
    XX[:, 1] = (
        -(X[:, 0] - center[0]) * np.sin(alpha)
        + (X[:, 1] - center[1]) * np.cos(alpha)
        + center[1]
    )

    return XX


def flatten(l):
    return (
        flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else [])
        if type(l) is list
        else [l]
    )


def seq_collate_scene(data):
    (
        obs_traj_list,
        pred_traj_list,
        obs_traj_rel_list,
        pred_traj_rel_list,
        scene_img_list,
        features_list,
        occupancy_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_traj_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    obs_traj = torch.cat(obs_traj_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_traj_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_traj_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_traj_rel_list, dim=0).permute(1, 0, 2)

    scene_img_list = tuple(flatten(list(scene_img_list)))
    try:
        features_image = torch.cat(features_list, dim=0)
    except:
        features_image = torch.empty(1)

    return {
        "in_xy": obs_traj,
        "gt_xy": pred_traj,
        "in_dxdy": obs_traj_rel,
        "gt_dxdy": pred_traj_rel,
        "size": torch.LongTensor([obs_traj.size(1)]),
        "scene_img": scene_img_list,
        "features": features_image,
        "occupancy": occupancy_list,
        "seq_start_end": seq_start_end,
    }


class TrajectoryDatasetEval(BaseDataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, **kwargs):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - delim: Delimiter in the dataset files
        """
        super().__init__(**kwargs)

        self.__dict__.update(locals())

        scene_nr = []
        self.scene_list = []
        self.image_list = []
        self.wall_points_dict = {}
        self.walls_list = []
        ped_ids = []
        seq_list = []

        collect_data = True
        for path in [file for file in self.all_files if ".jpg" in file]:
            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]

            img_parts = scene.split("-")

            if self.load_occupancy and img_parts[-1] == "op":
                scene = img_parts[-2]
                self.load_image(path, scene)

            elif not self.load_occupancy and img_parts[-1] != "op":
                self.load_image(path, scene)
                continue

        if len(self.images) == 0:
            assert False, "No valid imges in folder"

        num_peds_in_seq = []
        for path in [file for file in self.all_files if ".txt" in file]:
            if not collect_data:
                break
            if self.special_scene and self.special_scene not in path:
                continue

            scene_path, data_type = path.split(".")
            scene = scene_path.split("/")[-1]

            if data_type == "txt":
                scene = "_".join(scene.split("_")[1:])

                self.logger.info("preparing %s" % scene)
                data = self.load_file(path, self.delim)

                frames = np.unique(data[:, 0]).tolist()
                frame_data_dict = defaultdict(list)
                for data_idx in range(data.shape[0]):
                    frame_data_dict[data[data_idx, 0]].append(data[data_idx][None])
                frame_data = [
                    np.concatenate(v)
                    for _, v in sorted(frame_data_dict.items(), key=lambda kv: kv[0])
                ]

                num_sequences = int(math.ceil((len(frames) - self.seq_len) / self.skip))

                for idx in range(0, num_sequences * self.skip, self.skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx : idx + self.seq_len], axis=0
                    )
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    num_peds = 0
                    peds_scene = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]

                        if (curr_ped_seq[1:, 0] - curr_ped_seq[:-1, 0] != 1).any() or (
                            len(curr_ped_seq) != self.seq_len
                        ):
                            continue

                        ped_ids.append(ped_id)
                        num_peds += 1

                        # If not a active prediction, pad with NaNs so they
                        # are not evaluated
                        ped_seq = curr_ped_seq[:, 2:4].copy()
                        if curr_ped_seq.shape[1] == 5:
                            if (curr_ped_seq[:, 4] == 0).any():
                                ped_seq[self.obs_len :, :] = np.nan

                        peds_scene.append(ped_seq)

                    if num_peds > 0:
                        num_peds_in_seq.append(num_peds)

                        seq_list.append(np.stack((peds_scene), axis=0))
                        self.scene_list.append(scene)

        self.ped_ids = np.array(ped_ids, np.int)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        self.trajectory = np.concatenate(seq_list, axis=0)

        print("scene list", len(self.scene_list))
        print("trajectories", len(self.trajectory))

        if self.scale:
            self.scale_func()
        if self.norm2meters:
            print("norming to meters")
            self.scale2meters()

        self.wall_available = False

    @property
    def obs_traj(self):
        return torch.from_numpy(self.trajectory[:, : self.obs_len]).float()

    @property
    def pred_traj(self):
        return torch.from_numpy(self.trajectory[:, self.obs_len :]).float()

    @property
    def obs_traj_rel(self):
        traj_rel = self.trajectory[:, 1:] - self.trajectory[:, :-1]
        return torch.from_numpy(traj_rel[:, : self.obs_len - 1]).float()

    @property
    def pred_traj_rel(self):
        traj_rel = self.trajectory[:, 1:] - self.trajectory[:, :-1]
        return torch.from_numpy(traj_rel[:, self.obs_len - 1 :]).float()

    def get_scene(self, index):
        (
            in_xy,
            gt_xy,
            in_dxdy,
            gt_dxdy,
            scene_img,
            features,
            walls,
        ) = self.__getitem__(index)

        return {
            "in_xy": in_xy.permute(1, 0, 2),
            "gt_xy": gt_xy.permute(1, 0, 2),
            "in_dxdy": in_dxdy.permute(1, 0, 2),
            "gt_dxdy": gt_dxdy.permute(1, 0, 2),
            "scene_img": scene_img,
            "features": features.squeeze(0),
            "seq_start_end": [[0, in_xy.size(0)]],
        }

    def scale_func(self):
        for index in np.arange(len(self.seq_start_end)):
            start, end = self.seq_start_end[index]
            scene = self.scene_list[index]
            ratio = self.images[scene]["scale_factor"]
            self.trajectory[start:end] *= ratio

    def __getitem__(self, index):
        (start, end) = self.seq_start_end[index]
        scale_factor_small = self.img_scaling / self.scaling_small
        scale_factor_tiny = self.img_scaling / self.scaling_tiny
        xy_orig = self.trajectory[start:end]
        scene = self.scene_list[index]
        img = self.images[scene]["scaled_image"]

        xy = xy_orig.copy()

        if self.wall_available:
            wall_p = self.wall_points_dict[scene]

        if self.format == "pixel":
            scale2orig = 1 / self.images[scene]["scale_factor"]
        elif self.format == "meter":
            scale2orig = self.img_scaling
        else:
            assert False, " Not valid format '{}': 'meters' or 'pixel'".format(
                self.format
            )

        center = np.array(img.size) / 2.0
        corners = np.array(
            [[0, 0], [0, img.height], [img.width, img.height], [img.width, 0]]
        )

        if self.data_augmentation and self.phase == "train":
            alpha = np.random.rand() * 2 * np.pi
            rand_num = np.random.choice(np.arange(3))
        else:
            rand_num = 0
            alpha = 0

        if rand_num != 0:
            if rand_num == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                xy[:, :, 0] = img.width * scale2orig - xy[:, :, 0]

                if self.wall_available:
                    for i in range(len(wall_p)):
                        wall_p[i][:, 0] = img.width * scale2orig - wall_p[i][:, 0]

            elif rand_num == 2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                xy[:, :, 1] = img.height * scale2orig - xy[:, :, 1]
                # transform wall
                if self.wall_available:
                    for i in range(len(wall_p)):
                        wall_p[i][:, 1] = img.height * scale2orig - wall_p[i][:, 1]

        img = img.rotate(alpha / np.pi * 180, expand=True)

        corners_trans = rotate(corners, center, alpha)
        offset = corners_trans.min(axis=0)
        corners_trans -= offset

        if self.wall_available:
            wall_points = []
            for wall in wall_p.copy():
                wall_points.append(
                    rotate(wall, center * scale2orig, alpha) - offset * scale2orig
                )
            wall_p = wall_points
        else:
            wall_p = torch.empty(1)

        xy = xy.reshape((end - start) * self.seq_len, -1)
        xy = rotate(xy.copy(), center * scale2orig, alpha) - offset * scale2orig
        xy = xy.reshape((end - start), self.seq_len, -1)

        small_image = img.resize(
            (
                int(round(img.width * scale_factor_small)),
                int(round(img.height * scale_factor_small)),
            ),
            Image.ANTIALIAS,
        )
        tiny_image = img.resize(
            (
                int(round(img.width * scale_factor_tiny)),
                int(round(img.height * scale_factor_tiny)),
            ),
            Image.ANTIALIAS,
        )

        scene_image = {
            "ratio": self.images[scene]["ratio"],
            "scene": scene,
            "scaled_image": copy.copy(img),
            "small_image": copy.copy(small_image),
            "tiny_image": copy.copy(tiny_image),
        }

        xy = torch.from_numpy(xy).float()
        dxdy = xy[:, 1:] - xy[:, :-1]
        obs_traj = xy[:, : self.obs_len]
        pred_traj = xy[:, self.obs_len:]

        feature_list = []
        cropped_img_list = []

        for idx in np.arange(0, end - start):
            features_id, cropped_img = self.ImageFeatures_small(
                scene_image, obs_traj[idx], pred_traj[idx]
            )

            feature_list.append(features_id)
            cropped_img_list.append(cropped_img)

        features = torch.cat(feature_list)

        scene_image = (end - start) * [scene_image]

        return [
            obs_traj,
            pred_traj,
            dxdy[:, : self.obs_len - 1],
            dxdy[:, self.obs_len - 1:],
            scene_image,
            features,
            wall_p,
        ]

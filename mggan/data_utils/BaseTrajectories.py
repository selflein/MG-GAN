import os
import copy
import logging

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from mggan.data_utils import experiments


logger = logging.getLogger(__name__)


def re_im(img):
    img = (img + 1) / 2.0
    return img


class BaseDataset(Dataset):
    def __init__(
        self,
        save=False,
        load_p=True,
        dataset_name="stanford",
        phase="test",
        obs_len=8,
        pred_len=12,
        time_step=0.4,
        skip=1,
        data_augmentation=0,
        scale_img=True,
        max_num=None,
        load_occupancy=False,
        logger=logger,
        special_scene=None,
        scaling_small=0.5,
        scaling_tiny=0.25,
        margin_in=32,
        margin_out=16,
        margin_tiny=8,
        **kwargs
    ):
        super().__init__()
        self.__dict__.update(locals())

        self.save_dict = copy.copy(self.__dict__)

        self.dataset = getattr(experiments, self.dataset_name)()

        self.__dict__.update(self.dataset.get_dataset_args())
        self.data_dir = self.dataset.get_file_path(self.phase)
        self.seq_len = self.obs_len + self.pred_len
        self.images = {}

        all_files = os.listdir(self.data_dir)
        self.all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

    def print_wall(self, wall_points):
        for wall in wall_points:
            for index in range(1, len(wall) + 2):
                ind = [index % len(wall), (index + 1) % len(wall)]

                pos = wall[ind, :] / self.img_scaling
                plt.plot(pos[:, 0], pos[:, 1], color="r")

    def load_image(self, _path, scene):
        img = Image.open(_path)
        if "stanford" in self.dataset_name or "gofp" in self.dataset_name:
            if "stanford" in self.dataset_name:
                ratio = self.homography.loc[
                    (
                        (self.homography["File"] == "{}.jpg".format(scene))
                        & (self.homography["Version"] == "A")
                    ),
                    "Ratio",
                ].iloc[0]
            elif "gofp" in self.dataset_name:
                ratio = self.homography[scene]

            scale_factor = ratio / self.img_scaling

            old_width = img.size[0]
            old_height = img.size[1]

            new_width = int(round(old_width * scale_factor))
            new_height = int(round(old_height * scale_factor))

            scaled_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            scaled_img = img
            scale_factor = 1
            ratio = 1.0

        width = scaled_img.size[0]
        height = scaled_img.size[1]

        scale_factor_small = self.img_scaling / self.scaling_small
        small_width = int(round(width * scale_factor_small))
        small_height = int(round(height * scale_factor_small))
        small_image = scaled_img.resize((small_width, small_height), Image.ANTIALIAS)

        scale_factor_tiny = self.img_scaling / self.scaling_tiny
        tiny_width = int(round(width * scale_factor_tiny))
        tiny_height = int(round(height * scale_factor_tiny))
        tiny_image = scaled_img.resize((tiny_width, tiny_height), Image.ANTIALIAS)

        self.images.update(
            {
                scene: {
                    "ratio": ratio,
                    "scale_factor": scale_factor,
                    "scaled_image": scaled_img,
                    "small_image": small_image,
                    "tiny_image": tiny_image,
                }
            }
        )

    def get_ratio(self, scene):
        return self.images[scene]["ratio"]

    def scale2meters(self):
        self.trajectory *= self.img_scaling
        self.format = "meter"

    def load_file(self, _path, delim="tab"):
        if delim == "tab":
            delim = "\t"
        elif delim == "space":
            delim = " "

        df = pd.read_csv(_path, header=None, delimiter=delim)
        df.columns = self.data_columns

        if "label" and "lost" in df:
            data_settings = {"label": "Pedestrian", "lost": 0}

            for name, item in data_settings.items():
                df = df[df[name] == item]

        if self.dataset_name in ["stanford", "gofp"]:
            df = df[df["frame"] % int(round(self.framerate * self.time_step)) == 0]
            df["frame"] /= int(round(self.framerate * self.time_step))

        columns_experiment = ["frame", "ID", "x", "y"]

        if "is_active" in self.data_columns:
            columns_experiment.append("is_active")
        df = df[columns_experiment]

        return np.asarray(df.values)

    def __len__(self):
        return len(self.seq_start_end)

    def plot(self, index, modes=["in", "gt"], image_type="scaled", final_mask=False):
        out = self.get_scene(index)
        image = out["scene_img"][0]

        if image_type == "orig":
            img_label = "img"
            img = image[img_label]
            scale = 1
        elif image_type == "scaled":
            img_label = "scaled_image"
            img = image[img_label]
            if self.format == "meter":
                scale = 1.0 / self.img_scaling
        elif image_type == "small":
            img_label = "small_image"
            img = image[img_label]
            scale = 1.0 / self.scaling_small

        elif image_type == "tiny":
            img_label = "tiny_image"
            img = image[img_label]
            scale = 1.0 / self.scaling_tiny
        elif image_type == "patch":
            img = re_im(out["features"][:3].permute(1, 2, 0))
            scale = 1.0 / self.scaling_small

        else:
            assert False, "'{}' not valid <image_type>".format(image_type)

        center = out["in_xy"][-1, 0] * scale

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)

        if final_mask:
            rel_scaling = (2 * self.margin_in + 1) / (2 * self.margin_out + 1)

            mask = out["prob_mask"][0]

            final_pos_out = torch.cat(torch.where(mask == 1)).float()

            final_pos = torch.zeros_like(final_pos_out)
            final_pos[0] = final_pos_out[1]
            final_pos[1] = final_pos_out[0]
            center_pixel_small = out["in_xy"][-1, 0] * scale
            if not image_type == "patch":
                final_pos_pixel_centered = (
                    (final_pos - self.margin_out)
                    * self.scaling_small
                    * scale
                    * rel_scaling
                )

                final_pos_pixel = final_pos_pixel_centered + center_pixel_small

            elif image_type == "patch":
                final_pos_pixel = final_pos * rel_scaling
                mask = mask.numpy()
                mask = (mask * 255).astype(np.uint8)

                mask = Image.fromarray(mask, mode="L")
                mask = mask.resize(
                    (int(2 * self.margin_in + 1), int(2 * self.margin_in + 1))
                )
                ax.imshow(mask, alpha=0.5)
            plt.plot(final_pos_pixel[0], final_pos_pixel[1], "x")

        for m in modes:
            if m == "gt":
                marker = "-"
            else:
                marker = "-"

            scene_trajs = out["{}_xy".format(m)]
            for t in range(scene_trajs.shape[1]):
                traj = scene_trajs[:, t] * scale
                traj = traj.cpu().numpy()

                if image_type == "patch":
                    traj = traj + (self.margin_in) - center.cpu().numpy()

                if np.isnan(traj).any():
                    continue

                ax.plot(
                    (traj[:, 0]).astype(int),
                    (traj[:, 1]).astype(int),
                    linestyle=marker,
                    linewidth=int(3),
                )

        plt.show()

    def ImageFeatures_small(
        self, scene_image, trajectory, prediction, image_type="small_image"
    ):
        if self.format == "meter":
            scale = 1.0 / self.scaling_small
        else:
            scale = 1

        img = scene_image[image_type]
        center_meter = trajectory[-1].cpu().numpy()  # center in meter

        center_pixel_small = center_meter * scale
        center_scaled = center_pixel_small.astype(int)
        x_center, y_center = center_scaled

        cropped_img = img.crop(
            (
                int(x_center - self.margin_in),
                int(y_center - self.margin_in),
                int(x_center + self.margin_in + 1),
                int(y_center + self.margin_in + 1),
            )
        )

        position = torch.zeros(
            self.margin_out * 2 + 1, self.margin_out * 2 + 1, 1, device="cpu"
        )
        position[self.margin_in, self.margin_in, 0] = 1

        img = -1 + torch.from_numpy(np.array(cropped_img) * 1.0) * 2.0 / 256
        img = torch.cat((img.float(), position), dim=2)

        img = img.permute(2, 0, 1).unsqueeze(0)

        return img, cropped_img

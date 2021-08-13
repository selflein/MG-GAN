from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

import cv2

root_path = Path(os.path.realpath(__file__)).parent.parent.parent


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class Experiment:
    """The experiment objects store mainly paths to train and testfiles as well as homography matrices"""

    def __init__(self):
        super(Experiment, self).__init__()

        self.data_path = ""
        self.video_file = ""
        self.trajectory_file = ""
        self.static_image_file = ""
        self.obstacle_image_file = ""
        self.test_dir = ""
        self.train_dir = ""
        self.val_dir = ""
        self.name = ""
        self.H = []
        self.homography = []

        self.scaling = 0.05
        self.get_name()
        self.data_path = root_path / "data" / "datasets" / self.name
        self.test_dir = self.data_path / "test"
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"

    def get_name(self):
        self.name = self.__class__.__name__

    def get_file_path(self, phase):
        if phase == "test":
            self.dataDir = self.test_dir

        elif phase == "train":
            self.dataDir = self.train_dir

        elif phase == "val":
            self.dataDir = self.val_dir
        else:
            raise AssertionError('"phase" must be either train, val or test.')
        return str(self.dataDir)

    def init_args_dict(self):
        self.args_dict = {}
        for item in self.args:
            self.args_dict[item] = getattr(self, item)

    def get_dataset_args(self):
        # self.init_args_dict()
        return self.args_dict

    def plot_image(self):
        img = plt.imread(self.static_image_file)
        plt.imshow(img)

    def load_data(self):
        print(self.trajectory_file)
        self.world_data = np.loadtxt(self.trajectory_file)
        self.world_data = self.world_data[:, (0, 1, 2, 4)]
        return self.world_data

    def world2pixel(self):
        self.world_data = np.concatenate(
            (self.world_data, np.ones((1, len(self.world_data))).T), axis=1
        )
        pixel_data = self.world_data * 0
        # transform coordinates to pixel space
        for i, arr in enumerate(self.world_data[:, 2:5]):
            pixel_data[i, 2:5] = np.dot(self.H_inv, arr)
        pixel_data[:, 2:4] = pixel_data[:, 2:4] / pixel_data[:, 4][:, None]
        self.pixel_data = pixel_data

    def warp_obstacle(self):
        self.load_data()
        self.world2pixel()

        im_src = cv2.imread(str(self.obstacle_image_file))
        stat_img = cv2.imread(str(self.static_image_file))
        print(im_src.shape[0])

        corners = np.array(
            [
                [im_src.shape[1], im_src.shape[0]],
                [0, im_src.shape[0]],
                [im_src.shape[1], 0],
                [0, 0],
            ]
        )

        corners = np.concatenate((corners, np.ones((1, len(corners))).T), axis=1)

        corners_real = corners * 0
        self.world_data[:, 2:4] -= np.min(self.world_data[:, 2:4], axis=0)

        h, status = cv2.findHomography(
            self.pixel_data[:, (3, 2)], self.world_data[:, (3, 2)] / self.scaling
        )
        self.world_shifted = self.world_data
        print(corners)
        for i, arr in enumerate(corners):
            corners_real[i] = np.dot(self.H, arr)

        corners_real[:, :2] = corners_real[:, :2] / corners_real[:, 2][:, None]

        corners_real -= np.min(corners_real, axis=0)

        self.world_obst = cv2.warpPerspective(
            im_src,
            h,
            (
                int(np.max(corners_real[:, 0]) / self.scaling),
                int(np.max(corners_real[:, 1]) / self.scaling),
            ),
            borderValue=(255, 255, 255),
        )
        self.world_stat = cv2.warpPerspective(
            stat_img,
            h,
            (
                int(np.max(corners_real[:, 0]) / self.scaling),
                int(np.max(corners_real[:, 1]) / self.scaling),
            ),
            borderValue=(255, 255, 255),
        )

    def save_shift(self):
        time_sorted = sorted(np.unique(self.world_shifted[:, 0]))
        min_time = time_sorted[0]
        rel_min = time_sorted[1] - time_sorted[0]
        self.world_shifted[:, 0] = (self.world_shifted[:, 0] - min_time) / rel_min
        np.savetxt(
            os.path.join(self.data_path, "{}.txt".format(self.name)),
            self.world_shifted[:, :4],
            fmt="%i\t%i\t%1.2f\t%1.2f",
        )

    def save_images(self):
        cv2.imwrite(
            os.path.join(self.data_path, "{}_op.jpg".format(self.name)), self.world_obst
        )
        cv2.imwrite(
            os.path.join(self.data_path, "{}.jpg".format(self.name)), self.world_stat
        )

    def plot_points(self):

        self.plot_image()
        plt.scatter(self.pixel_data[:, 3], self.pixel_data[:, 2])
        plt.show()


class BiWi(Experiment):
    """The experiment objects store mainly paths to train and testfiles as well as homography matrices"""

    def __init__(self):
        super().__init__()
        self.delim = "tab"
        self.args_dict = {
            "norm2meters": False,
            "data_columns": ["frame", "ID", "y", "x"],
            "delim": "tab",
            "wall_available": False,
            "scale": False,
            "img_scaling": 0.05,
            "format": "meter",
        }


class stanford(Experiment):
    def __init__(self):
        super().__init__()

        self.args_dict = {
            "norm2meters": True,
            "data_columns": [
                "ID",
                "xmin, left",
                "ymin, left",
                "xmax, right",
                "ymax, right",
                "frame",
                "lost",
                "occuluded",
                "generated",
                "label",
                "x",
                "y",
            ],
            "delim": "tab",
            "homography": pd.read_csv(
                os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"
            ),
            "wall_available": False,
            "scale": True,
            "img_scaling": 0.05,
            "format": "pixel",
            "framerate": 30,
        }


class stanford_synthetic(Experiment):
    def __init__(self):
        super().__init__()

        self.args_dict = {
            "norm2meters": False,
            "data_columns": [
                "ID",
                "xmin, left",
                "ymin, left",
                "xmax, right",
                "ymax, right",
                "frame",
                "lost",
                "occuluded",
                "generated",
                "label",
                "x",
                "y",
                "_",
            ],
            "delim": "tab",
            "img_scaling": 0.05,
            "wall_available": True,
            "scale": False,
            "homography": pd.read_csv(
                os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"
            ),
            "format": "meter",
            "norm2meters": False,
            "framerate": 30,
        }


class social_stanford_synthetic(Experiment):
    def __init__(self):
        super().__init__()

        self.args_dict = {
            "norm2meters": False,
            "data_columns": [
                "ID",
                "xmin, left",
                "ymin, left",
                "xmax, right",
                "ymax, right",
                "frame",
                "lost",
                "occuluded",
                "generated",
                "label",
                "x",
                "y",
                "_",
                "other_ped",
            ],
            "delim": "tab",
            "img_scaling": 0.05,
            "wall_available": True,
            "scale": False,
            "homography": pd.read_csv(
                os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"
            ),
            "format": "meter",
            "norm2meters": False,
            "framerate": 30,
        }


class stanford_synthetic_2(Experiment):
    def __init__(self):
        super().__init__()

        self.args_dict = {
            "norm2meters": False,
            "data_columns": [
                "ID",
                "xmin, left",
                "ymin, left",
                "xmax, right",
                "ymax, right",
                "frame",
                "lost",
                "occuluded",
                "generated",
                "label",
                "x",
                "y",
                "_",
            ],
            "delim": "tab",
            "img_scaling": 0.05,
            "wall_available": True,
            "scale": False,
            "homography": pd.read_csv(
                os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"
            ),
            "format": "meter",
            "norm2meters": False,
            "framerate": 30,
        }


class stanford_synthetic_4(Experiment):
    def __init__(self):
        super().__init__()

        self.args_dict = {
            "norm2meters": False,
            "data_columns": [
                "ID",
                "xmin, left",
                "ymin, left",
                "xmax, right",
                "ymax, right",
                "frame",
                "lost",
                "occuluded",
                "generated",
                "label",
                "x",
                "y",
                "_",
            ],
            "delim": "tab",
            "img_scaling": 0.05,
            "wall_available": True,
            "scale": False,
            "homography": pd.read_csv(
                os.path.join(self.data_path, "H_SDD.txt"), delimiter="\t"
            ),
            "format": "meter",
            "norm2meters": False,
            "framerate": 30,
        }


class eth(BiWi):
    def __init__(self):
        super().__init__()

        self.video_file = self.data_path / "seq_eth.avi"
        self.trajectory_file = self.data_path / "eth_raw.txt"
        self.static_image_file = self.data_path / "eth_static.jpg"
        self._image_file = self.data_path / "eth_static.jpg"
        self.obstacle_image_file = self.data_path / "annotations.jpg"
        self.test_dir = self.data_path / "test"
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"

        self.H = np.array(
            [
                [2.8128700e-02, 2.0091900e-03, -4.6693600e00],
                [8.0625700e-04, 2.5195500e-02, -5.0608800e00],
                [3.4555400e-04, 9.2512200e-05, 4.6255300e-01],
            ]
        )
        self.H_inv = inv(self.H)


class hotel(BiWi):
    def __init__(self):
        super().__init__()

        self.video_file = self.data_path / "seq_hotel.avi"
        self.trajectory_file = self.data_path / "hotel_raw.txt"
        self.static_image_file = self.data_path / "hotel_static.jpg"
        self.obstacle_image_file = self.data_path / "annotations.jpg"
        self.test_dir = self.data_path / "test"
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"

        self.H = np.array(
            [
                [1.1048200e-02, 6.6958900e-04, -3.3295300e00],
                [-1.5966000e-03, 1.1632400e-02, -5.3951400e00],
                [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
            ]
        )
        self.H_inv = inv(self.H)


class univ(BiWi):
    def __init__(self):
        super().__init__()

        self.video_file = self.data_path / "students001.avi"
        self.trajectory_file = self.data_path / "univ_raw.txt"
        self.static_image_file = self.data_path / "univ_static.jpg"
        self.obstacle_image_file = self.data_path / "annotations.jpg"
        self.test_dir = self.data_path / "test"
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"

        self.H = np.array(
            [
                [0.032529736503653, -0.000730604859308, -7.969749046103707],
                [0.000883577230612, 0.026589331317173, -8.754694531864281],
                [0.001039809003515, 0.000025010101498, 1.007920696981254],
            ]
        )

        self.H_inv = inv(self.H)


class zara1(BiWi):
    def __init__(self):
        super().__init__()

        self.video_file = self.data_path / "crowds_zara01.avi"
        self.trajectory_file = self.data_path / "zara1_raw.txt"
        self.static_image_file = self.data_path / "zara_static.jpg"
        self.obstacle_image_file = self.data_path / "annotations.jpg"
        self.test_dir = self.data_path / "test"
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"

        self.H = np.array(
            [
                [-2.59600906e-02, -4.14338866e-07, 7.83994785e00],
                [-1.08705701e-03, 2.16676796e-02, 5.56418836e00],
                [6.05674393e-07, -8.00267888e-08, 1.00000000e00],
            ]
        )
        self.H_inv = inv(self.H)


class zara2(BiWi):
    def __init__(self):
        super().__init__()

        self.video_file = self.data_path / "crowds_zara02.avi"
        self.trajectory_file = self.data_path / "zara2_raw.txt"
        self.static_image_file = self.data_path / "zara_static.jpg"
        self.obstacle_image_file = self.data_path / "annotations.jpg"
        self.test_dir = self.data_path / "test"
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"

        self.H = np.array(
            [
                [-2.5956517e-02, -5.1572804e-18, 7.8388681e00],
                [-1.0953874e-03, 2.1664330e-02, -1.0032272e01],
                [1.9540125e-20, 4.2171410e-19, 1.0000000e00],
            ]
        )

        self.H_inv = inv(self.H)


class gofp(Experiment):
    def __init__(self):
        super().__init__()
        self.args_dict = {
            "norm2meters": True,
            "data_columns": [
                "frame",
                "ID",
                "x",
                "y",
                "moment",
                "old frame",
                "old_ID",
                "is_active",
            ],
            "delim": "tab",
            "img_scaling": 0.05,
            "wall_available": False,
            "scale": True,
            "homography": {
                "zara1": 0.03109532180986424,
                "eth": 0.06668566952360758,
                "hotel": 0.0225936169079401,
                "0000": 0.042200689823829046,
                "0400": 0.07905284109247492,
                "0401": 0.0598454105469989,
                "0500": 0.04631904070838066,
                "zara2": 0.03109532180986424,
            },
            "format": "pixel",
            "framerate": 10,
        }

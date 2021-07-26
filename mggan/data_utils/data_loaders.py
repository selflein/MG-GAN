import torch
import numpy as np

from mggan.data_utils.trajectories_scene import (
    TrajectoryDatasetEval,
    seq_collate_scene,
)


def get_dataloader(
    dataset,
    phase,
    augment=False,
    batch_size=8,
    workers=0,
    shuffle=False,
    split=None,
):
    assert phase in ("train", "val", "test")

    if phase in ("val", "test") and augment is True:
        print("No augmentation during validation or testing.")
        augment = False

    if dataset in (
        "stanford_synthetic",
        "stanford_synthetic_2",
        "social_stanford_synthetic",
    ):
        ds = TrajectoryDatasetEval(
            dataset_name=dataset,
            phase=phase,
            margin_in=16,
            margin_out=16,
            load_occupancy=False,
            scaling_small=1.2,
            data_augmentation=int(augment),
        )

        if split in ("upper", "lower"):
            if split == "lower":
                selector = ds.trajectory[:, 8, 1] > 16.0
            else:
                selector = ds.trajectory[:, 8, 1] <= 16.0

            new_scene_list = []
            new_trajectory = []
            new_ped_ids = []
            new_seq_start_end = []
            last_end = 0
            for scene_idx, (start, end) in enumerate(ds.seq_start_end):
                if selector[start:end].any():
                    new_scene_list.append(ds.scene_list[scene_idx])
                    new_trajectory.append(ds.trajectory[start:end])
                    new_ped_ids.append(ds.ped_ids[start:end])

                    next_end = last_end + end - start
                    new_seq_start_end.append([last_end, next_end])
                    last_end = next_end

            ds.trajectory = np.concatenate(new_trajectory)
            ds.ped_ids = np.concatenate(new_ped_ids)
            ds.seq_start_end = new_seq_start_end
            ds.scene_list = new_scene_list

        collate_fn = seq_collate_scene
    elif dataset == "stanford":
        ds = TrajectoryDatasetEval(
            dataset_name="stanford",
            phase=phase,
            margin_in=16,
            margin_out=16,
            load_occupancy=False,
            scaling_small=0.7,
            data_augmentation=int(augment),
        )
        collate_fn = seq_collate_scene
    elif dataset.lower() in ("eth", "hotel", "zara1", "zara2", "univ", "gofp"):
        ds = TrajectoryDatasetEval(
            dataset_name=dataset,
            phase=phase,
            margin_in=16,
            margin_out=16,
            load_occupancy=False,
            scaling_small=0.5,
            data_augmentation=int(augment),
        )
        collate_fn = seq_collate_scene
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader

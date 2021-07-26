#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from argparse import ArgumentParser

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from mggan.model.train import PiNetMultiGeneratorGAN
from mggan.data_utils.data_loaders import get_dataloader
from mggan.evaluation import (
    evaluate_ade_fde,
    evaluate_precision_recall,
)

parser = ArgumentParser()
parser.add_argument("--split", default="all")
parser.add_argument("--device", default="cuda")
parser.add_argument("--radius", type=float, default=2.0)
parser.add_argument("--model_path")
parser.add_argument("--output_folder", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--phase", required=True, choices=["train", "val", "test"])
parser.add_argument("--eval_set", default=None)
parser.add_argument(
    "--pred_strat",
    default="all",
    choices=["all", "sampling", "expected", "smart_expected", "rejection"],
)
parser.add_argument("--no-precision-recall", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    phase = args.phase
    show_plots = False
    num_preds_list = list(range(1, 21))

    checkpoint = args.checkpoint
    if args.pred_strat == "all":
        pred_strats = ["smart_expected", "expected", "sampling"]
    else:
        pred_strats = [args.pred_strat]

    disc_radius = args.radius

    split = args.split
    assert split in ("lower", "upper", "all")

    model = Path(args.model_path).stem

    output_csv = (
        Path(args.output_folder)
        / f"{model}_{phase}_{checkpoint}_{split}_{args.pred_strat}_radius_{disc_radius}.csv"
    )
    print(output_csv)

    device = args.device
    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    torch.set_grad_enabled(False)

    model_dirs = [d for d in Path(args.model_path).iterdir() if "version" in d.stem]

    models = []
    for pred_strat in pred_strats:
        for model_dir in model_dirs:
            models.append((model_dir, pred_strat))

    all_results = defaultdict(list)
    for model_dir, pred_strat in tqdm(models):
        try:
            model, config = PiNetMultiGeneratorGAN.load_from_path(model_dir, checkpoint)
        except Exception as e:
            print(e)
            model, config = PiNetMultiGeneratorGAN.load_from_path(model_dir, "best")

        if config.num_gens == 1 and pred_strat not in ("sampling", "rejection"):
            continue

        if config.weighting_target == "none" and "smart" in pred_strat:
            continue

        model.G.to(device)
        model.device = device
        model.G.eval()
        config.augment = False

        if args.eval_set is not None:
            all_results["Training dataset"].append(config.dataset)
            config.dataset = args.eval_set

        test_loader, _ = get_dataloader(
            config.dataset, phase, batch_size=32, split=split
        )

        all_results["Model"].append(config.name)
        all_results["# Generators"].append(config.num_gens)
        all_results["Decoder dim"].append(config.decoder_h_dim)
        all_results["Generator params"].append(config.num_gen_parameters)
        all_results["Prediction strategy"].append(pred_strat)
        all_results["Mode"].append(config.experiment)
        all_results["Use MGAN"].append(config.gan_type)
        all_results["Prior"].append(config.weighting_target)
        all_results["Dataset"].append(config.dataset)
        all_results["maximization samples"].append(config.num_samples)
        all_results["expectation samples"].append(config.num_expectation_samples)
        all_results["L2 loss weight"].append(config.l2_loss_weight)
        all_results["clf loss weight"].append(config.clf_loss_weight)
        all_results["sigma"].append(config.sigma)

        metric_dict = {}
        preds = model.get_predictions(
            test_loader, max(num_preds_list), strategy=pred_strat
        )
        metric_dict.update(evaluate_ade_fde(test_loader.dataset, preds, num_preds_list))
        if not args.no_precision_recall:
            metric_dict.update(
                evaluate_precision_recall(
                    test_loader.dataset, preds, disc_radius, num_preds_list
                )
            )

        for k, v in metric_dict.items():
            all_results[k].append(v)

        df = pd.DataFrame(all_results)
        df.to_csv(output_csv)

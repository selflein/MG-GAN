import abc
import math
from pathlib import Path
from statistics import mean
from argparse import Namespace

from test_tube import Experiment

from mggan.utils import *
from mggan.data_utils.data_loaders import get_dataloader
from mggan.model.config import get_parser

# Set seeds for reproducibility
torch.random.manual_seed(145325)
np.random.seed(435346)

# Issue with CuDNN LSTM implementation
# https://github.com/pytorch/pytorch/issues/27837
torch.backends.cudnn.enabled = True

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class MultiGeneratorGAN(abc.ABC):
    def __init__(self, generator, discriminator, config, writer):
        self.writer: Experiment = writer
        self.config = config
        self.device = torch.device("cuda" if config.gpus else "cpu")
        self.D = discriminator.to(self.device)
        self.G = generator.to(self.device)
        self.l2_weight = self.config.l2_loss_weight
        self.gan_type = self.config.gan_type

        self.log_dir = Path(
            self.writer.get_data_path(self.writer.name, self.writer.version)
        )
        self.model_save_dir = self.log_dir / "checkpoints"
        self.model_save_dir.mkdir(exist_ok=True)

        if config.gpus:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # setup optimizer
        self.optimizerD = torch.optim.AdamW(
            self.D.parameters(), lr=self.config.d_lr, betas=(config.beta1, 0.999)
        )
        self.optimizerG = torch.optim.AdamW(
            self.G.parameters(), lr=self.config.g_lr, betas=(config.beta1, 0.999)
        )

        self.lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizerD, config.epochs, eta_min=0, last_epoch=-1
        )
        self.lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizerG, config.epochs, eta_min=0, last_epoch=-1
        )

        self.epoch = 0

        # setup loss function
        criterion_bce = nn.BCELoss(reduction="none").to(self.device)
        criterion_mse = nn.MSELoss(reduction="none").to(self.device)
        if self.config.gan_obj == "NS":
            phi_1 = lambda dreal, lreal, lfake: criterion_bce(dreal, lreal)
            phi_2 = lambda dfake, lreal, lfake: criterion_bce(dfake, lfake)
            phi_3 = lambda dfake, lreal, lfake: criterion_bce(dfake, lreal)
        elif self.config.gan_obj == "MM":
            phi_1 = lambda dreal, lreal, lfake: criterion_bce(dreal, lreal)
            phi_2 = lambda dfake, lreal, lfake: criterion_bce(dfake, lfake)
            phi_3 = lambda dfake, lreal, lfake: -criterion_bce(dfake, lfake)
        elif self.config.gan_obj == "LS":
            phi_1 = lambda dreal, lreal, lfake: criterion_mse(dreal, lreal)
            phi_2 = lambda dfake, lreal, lfake: criterion_mse(dfake, lfake)
            phi_3 = lambda dfake, lreal, lfake: criterion_mse(dfake, lreal)
        elif self.config.gan_obj == "W":
            phi_1 = lambda dreal, lreal, lfake: -dreal.mean()
            phi_2 = lambda dfake, lreal, lfake: dfake.mean()
            phi_3 = lambda dfake, lreal, lfake: -dfake.mean()
        else:
            raise ValueError("Objective not supported")

        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.phi_3 = phi_3

    def train(self):
        train_loader = get_dataloader(
            dataset=self.config.dataset,
            phase="train",
            augment=self.config.augment,
            batch_size=self.config.batch_size,
            workers=self.config.workers,
            shuffle=True,
        )
        val_loader = get_dataloader(
            dataset=self.config.dataset,
            phase="val",
            augment=False,
            batch_size=self.config.batch_size,
            workers=self.config.workers,
            shuffle=False,
        )
        total_iterations = 0

        track_metric = "val/ADE k=20"
        min_track_metric = math.inf

        for epoch in range(self.config.epochs):
            self.epoch += 1
            self.D.train()
            self.G.train()
            metrics = defaultdict(list)
            for i, batch in enumerate(tqdm(train_loader)):
                # inp.shape (hist_len, b, 2),
                in_xy = batch["in_xy"].to(self.device)
                in_dxdy = batch["in_dxdy"].to(self.device)
                b = in_xy.size(1)
                sub_batches = (
                    batch["seq_start_end"]
                    if "seq_start_end" in batch
                    else list(zip(range(b), range(1, b + 1)))
                )
                # gt.shape: (pred_len, b, 2)
                gt_xy = batch["gt_xy"].to(self.device)
                gt_dxdy = batch["gt_dxdy"].to(self.device)

                # Filter masked trajectories
                # Shape: (b,)
                loss_mask = ~gt_xy.isnan().any(2).any(0)
                gt_dxdy = gt_dxdy[:, loss_mask]
                gt_xy = gt_xy[:, loss_mask]

                img = batch["features"].to(self.device) if "features" in batch else None

                if (total_iterations % self.config.num_gen_steps == 0) or (
                    self.epoch >= self.config.keep_gen_steps
                ):
                    for u in range(self.config.num_unrolling_steps + 1):
                        self.discriminator_step(
                            in_xy,
                            in_dxdy,
                            gt_xy,
                            gt_dxdy,
                            sub_batches,
                            metrics,
                            loss_mask,
                            img,
                        )

                        if u == 0 and self.config.num_unrolling_steps > 0:
                            backup = self.D.state_dict()

                self.generator_step(
                    in_xy, in_dxdy, gt_xy, gt_dxdy, sub_batches, metrics, loss_mask, img
                )
                self.net_chooser_step(
                    in_xy, in_dxdy, gt_xy, gt_dxdy, sub_batches, metrics, loss_mask, img
                )

                if self.config.num_unrolling_steps > 0:
                    self.D.load_state_dict(backup)

                if i % 10 == 0 and self.gan_type == "probgan":
                    # Update history discriminators aggregations
                    self.D.update_hist()

                total_iterations += 1

            """ Validation """
            if self.epoch % self.config.val_every == 0:
                self.D.eval()
                self.G.eval()
                with torch.no_grad():
                    m = self.check_accuracy(
                        val_loader,
                        vis=True,
                        prefix="val/",
                        num_k=self.config.top_k_test,
                    )
                    for k, v in m.items():
                        metrics[f"val/{k}"].append(v)

                cur_track_metric = mean(metrics[track_metric])
                if cur_track_metric < min_track_metric:
                    tqdm.write(
                        f'Saving best model... "{track_metric}: Before: '
                        f"{min_track_metric}, After: {cur_track_metric}"
                    )
                    min_track_metric = cur_track_metric
                    self.save(checkpoint_name="checkpoint_best.pth")

            metrics = {k: np.mean(v) for k, v in metrics.items()}
            self.writer.log(metrics, epoch)
            if self.epoch % self.config.save_every == 0:
                self.save()

            self.l2_weight *= self.config.l2_decay_rate
            self.lr_schedulerD.step()
            self.lr_schedulerG.step()
            self.writer.save()

    @abc.abstractmethod
    def generator_step(
        self,
        in_xy,
        in_dxdy,
        gt_xy,
        gt_dxdy,
        sub_batches,
        train_metrics,
        loss_mask,
        img=None,
    ):
        pass

    @abc.abstractmethod
    def discriminator_step(
        self,
        in_xy,
        in_dxdy,
        gt_xy,
        gt_dxdy,
        sub_batches,
        train_metrics,
        loss_mask,
        img=None,
    ):
        pass

    @abc.abstractmethod
    def check_accuracy(self, loader, vis=False, prefix="", num_k=20):
        pass

    def save(self, checkpoint_name=None):
        save_obj = {
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "gen_opt": self.optimizerG.state_dict(),
            "disc_opt": self.optimizerD.state_dict(),
        }
        if not checkpoint_name:
            checkpoint_name = "checkpoint_{}.pth".format(self.epoch)
        torch.save(save_obj, self.model_save_dir / checkpoint_name)

    @abc.abstractmethod
    def predict(self, in_dxdy, in_xy, sub_batches, img=None, num=20, noise=None):
        pass

    @classmethod
    def load(cls, log_path: Path, exp_name: str, version: int, checkpoint):
        version_dir = log_path / exp_name / "version_{}".format(version)
        checkpoint_dir = version_dir / "checkpoints"

        if checkpoint == "latest":
            checkpoint_epochs = []
            for name in checkpoint_dir.iterdir():
                chkpt = name.stem.split("_")[1]
                if chkpt != "best":
                    checkpoint_epochs.append(int(chkpt))
            checkpoint = max(checkpoint_epochs)

        checkpoint_path = checkpoint_dir / "checkpoint_{}.pth".format(checkpoint)
        state_dicts = torch.load(checkpoint_path)
        tags_csv = version_dir / "meta_tags.csv"
        config = load_hparams_from_tags_csv(tags_csv)

        defaults = get_argparse_defaults(get_parser())
        defaults.update(config)
        config = Namespace(**defaults)

        g, d = cls.construct_model(config)
        writer = Experiment(log_path, name=exp_name, version=version)
        m = cls(g, d, config, writer)

        m.G.load_state_dict(state_dicts["generator"], strict=False)
        m.D.load_state_dict(state_dicts["discriminator"], strict=False)
        try:
            m.optimizerD.load_state_dict(state_dicts["disc_opt"])
            m.optimizerG.load_state_dict(state_dicts["gen_opt"])
        except Exception as e:
            print("Could not restore optimizers.", str(e))
            pass

        return m, config

    @classmethod
    def load_from_path(cls, version_path: Path, checkpoint="best"):
        assert "version" in version_path.stem, (
            "Input path should point to " "model version directory."
        )
        exp_folder = version_path.parent.parent
        model_name = version_path.parent.name
        version = int(version_path.stem.split("_")[1])
        model, config = cls.load(exp_folder, model_name, version, checkpoint)
        return model, config

    def test(self, num_k=20, batch_size=8, **kwargs):
        assert (
            self.config.inp_format == "rel"
        ), "Norm params only valid for relative input coordinates"

        test_loader, _ = get_dataloader(
            dataset=self.config.dataset,
            phase="test",
            augment=False,
            batch_size=batch_size,
            workers=self.config.workers,
            shuffle=False,
        )
        return self.check_accuracy(test_loader, vis=False, num_k=num_k, **kwargs)

    @abc.abstractmethod
    def net_chooser_step(
        self, in_xy, in_dxdy, gt_xy, gt_dxdy, sub_batches, metrics, loss_mask, img
    ):
        pass

    @staticmethod
    @abc.abstractmethod
    def construct_model(config):
        pass

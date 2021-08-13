import time
import random
from math import ceil
from pathlib import Path
from functools import partial

import torch.nn.functional as F
from test_tube import Experiment
from torch.distributions.categorical import Categorical

from mggan.utils import *
from mggan.model.config import get_parser
from mggan.evaluation import evaluate_ade_fde
from mggan.abstract_train import MultiGeneratorGAN
from mggan.model.model_factory import construct_model


class PiNetMultiGeneratorGAN(MultiGeneratorGAN):
    def __init__(self, generator, discriminator, config, writer):
        super().__init__(generator, discriminator, config, writer)
        assert self.gan_type in ("mgan", "gan", "infogan"), self.gan_type

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
        b = in_xy.size(1)
        gen_inp = (in_xy, in_dxdy, sub_batches)

        # Shape (num_samples, batch_size, z_size)
        noise = torch.stack(
            [
                get_global_noise(self.config.noise_dim, sub_batches, "gaussian")
                for _ in range(self.config.num_samples)
            ]
        )

        # Shape: (pred_len, num_samples, b , 2)
        gen_out, _, gen_idxs = self.G(
            *gen_inp,
            noise=noise,
            all_gen_out=False,
            img=img,
            mask=loss_mask,
            num_samples=self.config.num_samples,
        )

        G_train_loss = torch.tensor(0.0)

        # L2 loss
        if self.config.l2_loss_type != "none":
            # Shape (num_samples, b)
            l2_losses = torch.norm(gen_out.abs - gt_xy[:, None], dim=-1, p=2)
            if self.config.l2_loss_type == "mse":
                l2_losses = l2_losses ** 2
            l2_losses = l2_losses.sum(0)

            # Compute L2 loss as minimum for entire scene prediction
            min_l2 = 0.0
            for start, end in sub_batches:
                scene_l2_losses = l2_losses[:, start:end]
                sum_scene_loss = torch.sum(scene_l2_losses, 1)
                min_scene_loss = torch.min(sum_scene_loss)
                min_l2 += min_scene_loss

            min_l2 /= b
            train_metrics["train/L2_loss"].append(min_l2.item())
            G_train_loss += self.config.l2_loss_weight * min_l2

        # Adversarial loss
        disc_out = self.D(
            in_xy,
            in_dxdy,
            gen_out.abs,
            gen_out.rel,
            sub_batches,
            img=img,
            mask=loss_mask,
        )
        if isinstance(disc_out, tuple):
            disc_out, branch_out = disc_out
            # Shape (batch_size * num_samples, _)
            branch_out = branch_out.flatten(0, 1)

        adv_loss = self.phi_3(disc_out, *get_gan_labels(disc_out.shape))
        # Reweigh the loss according to how often a generator was sampled
        idxs, counts = torch.unique(gen_idxs, return_counts=True)
        for cur_idx, count in zip(idxs, counts):
            adv_loss[gen_idxs == cur_idx] /= count
        adv_loss = adv_loss.mean()
        train_metrics["train/gen_loss"].append(adv_loss.item())
        G_train_loss += adv_loss

        if self.gan_type == "mgan":
            # MGAN loss/ Info Loss (in this case the MGAN reconstruction loss
            # equals Info Loss since we have a discrete random variable
            # indicating the one hot encoding)
            classifier_loss = F.cross_entropy(
                branch_out, gen_idxs.view(-1), reduction="none"
            )
            classifier_loss = classifier_loss.reshape_as(gen_idxs)
            for cur_idx, count in zip(idxs, counts):
                classifier_loss[gen_idxs == cur_idx] /= count
            classifier_loss = classifier_loss.mean()
            train_metrics["train/info_mgan_loss"].append(classifier_loss.item())
            G_train_loss += self.config.clf_loss_weight * classifier_loss

        elif self.gan_type == "infogan":
            # InfoGAN objective on continuous part of latent code
            assert self.G.n_gs == 1

            # Shape (batch_size * num_samples, code_dim)
            noise_target = noise[:, loss_mask, :3].transpose(0, 1).flatten(0, 1)

            info_loss = 0.5 * F.mse_loss(branch_out, noise_target)
            info_loss /= self.config.num_samples
            train_metrics["train/info_loss"].append(info_loss.item())
            G_train_loss += info_loss

        # Back propagation
        self.D.zero_grad()
        self.G.zero_grad()
        G_train_loss.backward()
        if self.config.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                self.G.parameters(), self.config.clipping_threshold_g
            )
        self.optimizerG.step()

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
        train_loss = torch.tensor(0.0)
        """ Train discriminator with real data """
        real_result = self.D(
            in_xy, in_dxdy, gt_xy, gt_dxdy, sub_batches, img=img, mask=loss_mask
        )
        if isinstance(real_result, tuple):
            real_result = real_result[0]

        real_loss = self.phi_1(real_result, *get_gan_labels(real_result.shape)).mean()

        noise = get_global_noise(self.config.noise_dim, sub_batches, "gaussian")[None]
        with torch.no_grad():
            gen_out, net_chooser_weights, gen_labels_gt = self.G(
                in_xy,
                in_dxdy,
                sub_batches,
                noise=noise,
                all_gen_out=False,
                img=img,
                num_samples=1,
                mask=loss_mask,
            )

        disc_out = self.D(
            in_xy,
            in_dxdy,
            gen_out.abs,
            gen_out.rel,
            sub_batches,
            img=img,
            mask=loss_mask,
        )

        if self.gan_type == "mgan":
            disc_out, branch_out = disc_out
            # Both shape (batch_size * num_samples, ...)
            ce_loss = F.cross_entropy(branch_out.flatten(0, 1), gen_labels_gt.flatten())
            train_metrics["train/info_mgan_disc_loss"].append(ce_loss.item())
            train_loss += ce_loss
        elif self.gan_type == "infogan":
            assert self.G.n_gs == 1
            disc_out, latent_reg_out = disc_out
            # Shape (batch_size * num_samples, code_dim)
            noise_target = noise[:, loss_mask, :3].transpose(0, 1).flatten(0, 1)

            info_loss = 0.5 * F.mse_loss(latent_reg_out.flatten(0, 1), noise_target)
            train_metrics["train/disc_info_loss"].append(info_loss.item())
            train_loss += info_loss

        fake_loss = self.phi_2(disc_out, *get_gan_labels(disc_out.shape)).mean()

        train_loss += real_loss + fake_loss
        train_metrics["train/discr_loss"].append((fake_loss + real_loss).item())

        if self.config.gan_obj == "W":
            train_loss += calc_gradient_penalty(
                self.D, in_xy, in_dxdy, gt_xy, gt_dxdy, gen_out.abs, gen_out.rel
            )

        self.D.zero_grad()
        train_loss.backward()
        if self.config.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(
                self.D.parameters(), self.config.clipping_threshold_d
            )
        self.optimizerD.step()

    def get_predictions(self, loader, num_preds=20, strategy="sampling"):
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

        self.D.eval()
        self.G.eval()
        pred_func = self.get_predict_func(strategy)
        all_preds = []
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            in_xy = batch["in_xy"].to(self.device)
            in_dxdy = batch["in_dxdy"].to(self.device)

            b = in_dxdy.size(1)
            sub_batches = (
                batch["seq_start_end"]
                if "seq_start_end" in batch
                else list(zip(range(b), range(1, b + 1)))
            )
            img = batch["features"].to(self.device) if "features" in batch else None

            # Generate multiple predictions
            preds, preds_rel, probs, gen_idxs = pred_func(
                in_dxdy,
                in_xy,
                sub_batches,
                img=img,
                num=num_preds,
            )
            all_preds.append(to_numpy(preds))
        return np.concatenate(all_preds, 2)

    def check_accuracy(
        self,
        loader,
        vis=False,
        prefix="",
        num_k=20,
        predict_strategy="sampling",
        debug=False,
        **kwargs,
    ):
        preds = self.get_predictions(loader, num_preds=num_k, strategy=predict_strategy)
        metrics = evaluate_ade_fde(loader.dataset, preds, [num_k])
        return metrics

    def predict(
        self, in_dxdy, in_xy, sub_batches, img=None, num=20, noise=None, mask=None
    ):
        """Predict trajectories based on the given input.

        Args:
            in_dxdy: Offsets of the observed trajectory (obs_len, b, 2).
            in_xy: Observed trajectory of shape (obs_len, b, 2).
            sub_batches: Tensor of shape (n, 2) with start and end index of
                the sub-batches.
            num: Number of predictions to make.

        Returns:
            Numpy array of shape (pred_len, num_samples, b, 2).
        """
        self.G.eval()

        with torch.no_grad():
            preds, net_chooser_out, gen_idxs = self.G(
                in_xy,
                in_dxdy,
                sub_batches,
                noise=noise,
                all_gen_out=False,
                img=img,
                num_samples=num,
                mask=mask,
            )
            probs = torch.softmax(net_chooser_out, 1)
        assert preds.abs.shape[1] == num
        return preds.abs, preds.rel, to_numpy(probs), to_numpy(gen_idxs)

    def predict_expected(
        self, in_dxdy, in_xy, sub_batches, img=None, num=20, noise=None, mask=None
    ):
        """See predict(...)"""
        self.G.eval()

        with torch.no_grad():
            # Shape: (pred_len, num_samples, num_gens, b, 2)
            preds, net_chooser_out, gen_idxs = self.G(
                in_xy,
                in_dxdy,
                sub_batches,
                noise=noise,
                all_gen_out=True,
                img=img,
                num_samples=num,
                mask=mask,
            )
            probs = to_numpy(torch.softmax(net_chooser_out, 1))
            expected_num = np.round(probs * num).astype(np.int)

            # Fill missing uniformly according to ranking of probabilities
            sort_idxs = np.argsort(-expected_num, axis=-1)
            num_samples_missing = num - np.sum(expected_num, 1)
            filler = np.zeros_like(expected_num)
            for b, num_missing in enumerate(num_samples_missing):
                num_missing_abs = np.abs(num_missing)
                uniq, counts = np.unique(
                    np.tile(sort_idxs[b], num_missing_abs)[:num_missing_abs],
                    return_counts=True,
                )
                filler[b, uniq] += np.sign(num_missing) * counts

            expected_num += filler
            assert (np.sum(expected_num, 1) == num).all()

            batch_abs, batch_rel, sample_idxs = [], [], []
            for b_idx in range(expected_num.shape[0]):
                idxs = []
                for i in range(num):
                    for idx in sort_idxs[b_idx]:
                        if expected_num[b_idx, idx] > 0:
                            idxs.append(idx)
                            expected_num[b_idx, idx] -= 1
                sample_idxs.append(torch.tensor(idxs[:num]))

            sample_idxs = torch.stack(sample_idxs, 0)

        pred_len, _, num_gens, b, _ = preds.abs.shape
        offsets = get_selection_indices(sample_idxs)

        idxs = sample_idxs + offsets * num_gens
        out_xy = preds.abs.reshape(pred_len, -1, b, 2)
        out_dxdy = preds.rel.reshape(pred_len, -1, b, 2)

        # Select the sampled indices
        # Shape (pred_len, num_samples, batch_size, 2)
        out_xy = out_xy[:, idxs, torch.arange(b).unsqueeze(1)].transpose(1, 2)
        out_dxdy = out_dxdy[:, idxs, torch.arange(b).unsqueeze(1)].transpose(1, 2)
        assert out_xy.shape[1] == num
        return out_xy, out_dxdy, probs, to_numpy(sample_idxs)

    def predict_uniform(
        self,
        in_dxdy,
        in_xy,
        sub_batches,
        img=None,
        num=20,
        noise=None,
        eps=0.0,
        mask=None,
    ):
        """See predict(...)"""
        self.G.eval()

        num_gens = self.G.n_gs
        with torch.no_grad():
            # Shape: (pred_len, num_samples, num_gens, b, 2)
            preds, net_chooser_out, gen_idxs = self.G(
                in_xy,
                in_dxdy,
                sub_batches,
                noise=noise,
                all_gen_out=True,
                img=img,
                num_samples=num * num_gens,
                mask=mask,
            )
            probs = torch.softmax(net_chooser_out, 1)

            # Select generators over threshold in descending order of their
            # original probability
            over_thresh = probs > eps
            over_thresh_sum = torch.sum(over_thresh, 1)
            # If none over threshold take all uniform
            over_thresh[over_thresh_sum < 1.0] = torch.ones(over_thresh.shape[1]).bool()

            batch_abs, batch_rel, sample_idxs = [], [], []
            pred_len, n_samples, num_gens, batch_size, _ = preds.abs.shape
            for b, gen_selector in enumerate(over_thresh):
                sort_idxs = torch.argsort(-probs[b, gen_selector])
                batch_abs.append(
                    preds.abs[:, :, gen_selector, b][:, :, sort_idxs].reshape(
                        pred_len, -1, 2
                    )[:, :num]
                )
                batch_rel.append(
                    preds.rel[:, :, gen_selector, b][:, :, sort_idxs].reshape(
                        pred_len, -1, 2
                    )[:, :num]
                )
                sample_idxs.append(
                    torch.arange(num_gens)[gen_selector][sort_idxs].repeat(num)[:num]
                )

            batch_abs = torch.stack(batch_abs, 2)
            batch_rel = torch.stack(batch_rel, 2)
            sample_idxs = torch.stack(sample_idxs, 0)

        assert batch_abs.shape[1] == num
        return batch_abs, batch_rel, to_numpy(probs), to_numpy(sample_idxs)

    def predict_smart_sampling(
        self,
        in_dxdy,
        in_xy,
        sub_batches,
        img=None,
        num=20,
        noise=None,
        eps=0.0,
        mask=None,
    ):
        """See predict(...)"""
        self.G.eval()

        num_gens = self.G.n_gs
        with torch.no_grad():
            # Shape: (pred_len, num_samples, num_gens, b, 2)
            preds, net_chooser_out, gen_idxs = self.G(
                in_xy,
                in_dxdy,
                sub_batches,
                noise=noise,
                all_gen_out=True,
                img=img,
                num_samples=num * num_gens,
                mask=mask,
            )
            probs = torch.softmax(net_chooser_out, 1)
            pred_len, num_samples, num_gens, batch_size, _ = preds.abs.shape

            # Uniform probability for generators over threshold
            over_thresh = (probs > eps).float()
            over_thresh_sum = torch.sum(over_thresh, 1)
            over_thresh[over_thresh_sum < 1.0] = torch.ones(over_thresh.shape[1])

            dist = Categorical(probs=over_thresh)
            samples = dist.sample((num,)).transpose(0, 1)

            sample_offsets = get_selection_indices(samples)

            idxs = samples + sample_offsets * num_gens

            # Select the sampled indices
            # Shape (pred_len, num_samples, batch_size, 2)
            out_xy = preds.abs.reshape(pred_len, -1, batch_size, 2)[
                :, idxs, torch.arange(batch_size).unsqueeze(1)
            ].transpose(1, 2)
            out_dxdy = preds.rel.reshape(pred_len, -1, batch_size, 2)[
                :, idxs, torch.arange(batch_size).unsqueeze(1)
            ].transpose(1, 2)
        assert out_xy.shape[1] == num
        return out_xy, out_dxdy, to_numpy(probs), to_numpy(samples)

    def predict_rejection(
        self,
        in_dxdy,
        in_xy,
        sub_batches,
        img=None,
        num=20,
        noise=None,
        sigma=1e-3,
        N=10,
        truncation_ratio=0.7,
        debug=False,
        mask=None,
    ):
        """See predict(...)"""
        self.G.eval()
        # Based on: Learning disconnected manifolds: no GAN’s land
        assert self.config.num_gens == 1, "Only implemented for single generator"
        assert 0.0 < truncation_ratio <= 1.0
        b = in_xy.shape[1]

        total_samples = num + ceil((1 - truncation_ratio) * num)

        if noise is None:
            noise = torch.stack(
                [
                    get_global_noise(self.config.noise_dim, sub_batches, "gaussian")
                    for _ in range(total_samples)
                ]
            )

        with torch.no_grad():
            # Shape: (pred_len, num_samples, num_gens, b, 2)
            preds, net_chooser_out, gen_idxs = self.G(
                in_xy,
                in_dxdy,
                sub_batches,
                noise=noise,
                all_gen_out=True,
                img=img,
                num_samples=total_samples,
                mask=mask,
            )
            # Shape: (b, num_samples, 1 * pred_len * 2)
            pred_vec = preds.abs.permute(3, 1, 2, 0, 4).reshape(b, total_samples, -1)
            probs = to_numpy(torch.softmax(net_chooser_out, 1))

            jacobian_frobenius_norm = torch.zeros(b, total_samples)
            for i in range(N):
                eps_i = (
                    torch.randn(total_samples, b, self.config.noise_dim) * sigma ** 2
                )
                preds_eps, _, _ = self.G(
                    in_xy,
                    in_dxdy,
                    sub_batches,
                    noise=noise + eps_i,
                    all_gen_out=True,
                    img=img,
                    num_samples=total_samples,
                    mask=mask,
                )
                pred_eps_vec = preds_eps.abs.permute(3, 1, 2, 0, 4).reshape(
                    b, total_samples, -1
                )
                summand = 1 / (sigma ** 2) * ((pred_eps_vec - pred_vec) ** 2).sum(-1)
                jacobian_frobenius_norm += summand

            jacobian_frobenius_norm /= N

        _, indices = torch.sort(jacobian_frobenius_norm, dim=1)
        if debug:
            gen_idxs[:] = 1
            gen_idxs[torch.arange(b)[None], indices[:, :num]] = 0
            return preds.abs.squeeze(2), preds.rel.squeeze(2), probs, to_numpy(gen_idxs)
        batch_abs = preds.abs[:, indices[:, :num], 0, torch.arange(b)[:, None]].permute(
            0, 2, 1, 3
        )
        batch_rel = preds.rel[:, indices[:, :num], 0, torch.arange(b)[:, None]].permute(
            0, 2, 1, 3
        )
        gen_idxs = gen_idxs[torch.arange(b)[:, None], indices[:, :num]]

        assert batch_abs.shape[1] == num
        return batch_abs, batch_rel, probs, to_numpy(gen_idxs)

    def get_predict_func(self, strategy: str):
        assert strategy in (
            "uniform_expected",
            "sampling",
            "expected",
            "rejection",
            "smart_expected",
            "smart_sampling",
            "uniform_sampling",
        )
        if strategy == "expected":
            return self.predict_expected
        elif strategy == "rejection":
            return self.predict_rejection
        elif strategy == "uniform_expected":
            return self.predict_uniform
        elif strategy == "smart_expected":
            return partial(self.predict_uniform, eps=1.0 / self.G.n_gs)
        elif strategy == "smart_sampling":
            return partial(self.predict_smart_sampling, eps=1.0 / self.G.n_gs ** 2)
        elif strategy == "uniform_sampling":
            return partial(self.predict_smart_sampling, eps=0.0)
        else:
            return self.predict

    def net_chooser_step(
        self, in_xy, in_dxdy, gt_xy, gt_dxdy, sub_batches, metrics, mask, img
    ):
        if self.config.weighting_target == "none":
            return

        # Shape (pred_len, num_samples, num_gens, b, 2)
        gen_out, net_chooser_weights, _ = self.G(
            in_xy,
            in_dxdy,
            sub_batches,
            noise=None,
            all_gen_out=True,
            img=img,
            num_samples=self.config.num_expectation_samples,
            mask=mask,
        )

        # Log the average weighting for the generators
        with torch.no_grad():
            probs = to_numpy(torch.softmax(net_chooser_weights, 1).mean(0))
            for i, prob in enumerate(probs):
                metrics[f"probs/Gen {i} probability"].append(prob)

        if self.config.weighting_target == "disc_scores":
            raise NotImplementedError
        elif self.config.weighting_target == "mgan":
            assert self.gan_type == "mgan"
            _, branch_out = self.D(
                in_xy, in_dxdy, gt_xy, gt_dxdy, sub_batches, mask=mask, img=img
            )
            out_probs = torch.softmax(net_chooser_weights, 1)
            target_probs = torch.softmax(branch_out, 1)
            loss = -(target_probs * out_probs.log()).sum(1).mean()
            reg = (0.9 ** self.epoch) * -(out_probs * out_probs.log()).sum(1).mean()
            loss -= reg

        elif self.config.weighting_target == "l2":
            # Shape (num_samples, num_gens, b)
            l2_dist = torch.norm(gen_out.abs - gt_xy[:, None, None], p=2, dim=-1).mean(
                0
            )
            # Shape (b, num_gens)
            per_gen_dist = l2_dist.min(0)[0].transpose(0, 1)
            min_idx = torch.argmin(per_gen_dist, dim=1)
            loss = F.cross_entropy(net_chooser_weights, min_idx)

        # Maximum likelihood assuming standard normal distributed errors
        elif self.config.weighting_target == "ml":
            # Shape (b, num_gens), Approximate P(Traj|Gen) with Monte Carlo samples
            out_probs = torch.softmax(net_chooser_weights, 1)
            log_prob = (
                torch.distributions.Normal(0, self.config.sigma)
                .log_prob(gen_out.abs - gt_xy[:, None, None])
                .sum([0, -1])
                .mean(0)
                .t()
            )
            # P(Gen|Traj) = (P(Traj|Gen) * P(Gen)) / P(Traj) via Bayes rule
            # with P(Traj) = sum(P(Traj|Gen))
            gen_prob = torch.softmax(log_prob, 1)
            loss = -(gen_prob * out_probs.log()).sum(1).mean()

        elif self.config.weighting_target == "endpoint":
            # Shape (num_samples, num_gens, b)
            l2_dist = torch.norm(gen_out.abs[-1] - gt_xy[-1, None, None], p=2, dim=-1)
            # Shape (b, num_gens)
            per_gen_dist = l2_dist.min(0)[0].transpose(0, 1)
            min_idx = torch.argmin(per_gen_dist, dim=1)
            loss = F.cross_entropy(net_chooser_weights, min_idx)

        else:
            raise ValueError("Weighting target does not exist")

        metrics["train/net_chooser_loss"].append(loss.item())

        loss *= self.config.pi_net_loss_weight

        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerG.step()

    @staticmethod
    def construct_model(config):
        return construct_model(config)


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.checkpoint:
        output_dir = Path(args.checkpoint)
        assert output_dir.is_dir()
        model, config = PiNetMultiGeneratorGAN.load_from_path(output_dir)
        config.gpus = True
        config.val_every = 1
    else:
        output_dir = Path(args.log_dir) / args.experiment
        output_dir.mkdir(exist_ok=True, parents=True)
        print(str(output_dir.resolve()))
        logger = Experiment(
            output_dir.resolve(),
            name=args.name,
            debug=args.debug,
            version=random.randint(10 ** 10, (10 ** 11) - 1),
        )

        G, D = construct_model(config=args)
        logger.argparse(args)
        model = PiNetMultiGeneratorGAN(G, D, args, logger)
        # Logger sometimes fails if not waiting
        time.sleep(5)
        logger.save()
    model.train()

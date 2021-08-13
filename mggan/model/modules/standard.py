import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from mggan.model.modules.cnn import AttentionGlobal
from mggan.model.modules.social import SocialAttention
from mggan.model.modules.social_gan import PoolHiddenNet
from mggan.utils import make_mlp, get_global_noise, get_selection_indices
from mggan.model.modules.common_modules import (
    TrajectoryEncoder,
    RelativeDecoder,
    get_input,
    GeneratorOutput,
)


class MultiGenerator(nn.Module):
    def __init__(
        self,
        z_size,
        encoder_h_dim,
        decoder_h_dim,
        social_feat_size,
        num_gens,
        pred_len,
        embedding_dim,
        inp_format,
        num_social_modules,
        pool_type,
        scene_dim,
        use_pinet,
        learn_prior=False,
    ):
        super(MultiGenerator, self).__init__()
        assert inp_format in ("rel", "abs", "abs_rel")
        assert num_social_modules in (0, 1, num_gens)
        assert pool_type in ("sways", "sgan")

        self.use_pinet = use_pinet
        self.inp_format = inp_format
        self.z_size = z_size
        self.embedding_dim = embedding_dim
        self.social_feat_size = social_feat_size
        self.n_social_modules = num_social_modules
        self.pool_type = pool_type
        self.decoder_h_dim = decoder_h_dim
        self.encoder_h_dim = encoder_h_dim
        self.scene_dim = scene_dim

        inp_size = 4 if inp_format == "abs_rel" else 2
        self.encoder = TrajectoryEncoder(
            inp_size=inp_size,
            hidden_size=encoder_h_dim,
            embedding_dim=embedding_dim,
            num_layers=1,
        )
        if scene_dim > 0:
            self.scene_encoder = AttentionGlobal(
                noise_attention_dim=0, PhysFeature=True, num_layers=2, channels_cnn=16
            )

        if self.social_feat_size > 0:
            if pool_type == "sways":
                self.social = SocialAttention(social_feat_size, encoder_h_dim)
            else:
                self.social = PoolHiddenNet(
                    embedding_dim=embedding_dim,
                    h_dim=encoder_h_dim,
                    mlp_dim=social_feat_size,
                    bottleneck_dim=encoder_h_dim,
                )

        self.gs = nn.ModuleList()
        for i in range(num_gens):
            decoder = RelativeDecoder(
                pred_len=pred_len,
                embedding_dim=embedding_dim,
                h_dim=decoder_h_dim,
                num_layers=1,
                social_feat_size=social_feat_size if social_feat_size > 0 else 0,
                z_size=z_size,
                dropout=0.0,
                inp_format=inp_format,
            )

            setattr(self, "G_{}".format(i), decoder)
            self.gs.append(decoder)

        self.n_gs = len(self.gs)
        self.pred_len = pred_len
        self.enc_h_to_dec_h = make_mlp(
            [encoder_h_dim + z_size + scene_dim + social_feat_size, decoder_h_dim],
            batch_norm=False,
        )

        assert not (
            use_pinet and learn_prior
        ), "Using conditional distribution already, `learn_prior` has no effect"
        self.net_chooser = nn.Sequential(
            nn.Linear(encoder_h_dim + scene_dim + social_feat_size, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, num_gens),
        )

        self.net_prior = nn.Parameter(
            torch.zeros(1, self.n_gs), requires_grad=learn_prior
        )

    def forward(
        self,
        in_xy,
        in_dxdy,
        sub_batches,
        noise=None,
        all_gen_out=True,
        img=None,
        num_samples=5,
        mask=None,
    ):
        """Generate new trajectories from past history.

        Args:
            in_xy: Input tensor of shape (history length, batch size, 2).
            in_dxdy: Input tensor of shape (history length - 1, batch size, 2).
            sub_batches: List with indexes where a scene starts and ends.
            noise: Optional fixed noise tensor for inference of shape
                (batch size, noise_dim).
            all_gen_out: If True all generators are outputting a prediction for
                every element in the batch.
            img: Image crop of shape (batch_size, 3, 32, 32)
            num_samples: Number of samples to generate.
            mask: Mask for which inputs to predict samples (Shape (batch_size,))

        Returns:
            Tensor of shape (pred_len, num_generators, b, 2)
            with predictions if all_gen_out is True and (pred_len, b, 2) o.w.
        """
        batch_size = in_xy.size(1)
        encoder_inp = get_input(in_xy, in_dxdy, self.inp_format)
        enc_h = self.encoder(encoder_inp)

        enc_features = [enc_h]
        if img is not None:
            scene_encoding = self.scene_encoder(img)
            enc_features.append(scene_encoding)

        if self.social_feat_size > 0:
            social_feats = self.social(in_xy, in_dxdy, enc_h, sub_batches)
            enc_features.append(social_feats)
        else:
            social_feats = torch.zeros(batch_size, self.social_feat_size)

        enc_h = torch.cat(enc_features, -1)

        if noise is not None:
            assert noise.shape == (num_samples, batch_size, self.z_size)
        else:
            noise = torch.stack(
                [
                    get_global_noise(self.z_size, sub_batches, "gaussian")
                    for _ in range(num_samples)
                ]
            )

        if mask is not None:
            in_xy = in_xy[:, mask]
            in_dxdy = in_dxdy[:, mask]
            enc_h = enc_h[mask]
            social_feats = social_feats[mask]
            noise = noise[:, mask]
            batch_size = torch.sum(mask)

        if all_gen_out:
            with torch.no_grad():
                pred_xy, pred_dxdy = self.forward_all(
                    in_xy,
                    in_dxdy,
                    enc_h,
                    noise=noise,
                    social_feats=social_feats,
                )
            net_chooser_out, sampled_gen_idxs = self.get_samples(enc_h, num_samples)

        else:
            with torch.no_grad():
                net_chooser_out, sampled_gen_idxs = self.get_samples(enc_h, num_samples)

            sample_offsets = get_selection_indices(sampled_gen_idxs)
            max_counts_idxs = torch.max(sample_offsets) + 1

            # Shape (pred_len, max_counts_idxs, num_gens, batch_size, 2)
            pred_xy, pred_dxdy = self.forward_all(
                in_xy,
                in_dxdy,
                enc_h,
                noise=noise[:max_counts_idxs],
                social_feats=social_feats,
            )
            # Shape (pred_len, max_counts_idxs * num_gens, batch_size, 2)
            pred_dxdy = pred_dxdy.reshape(self.pred_len, -1, batch_size, 2)
            pred_xy = pred_xy.reshape(self.pred_len, -1, batch_size, 2)

            idxs = sampled_gen_idxs + sample_offsets * self.n_gs

            # Select the sampled indices
            # Shape (pred_len, num_samples, batch_size, 2)
            pred_xy = pred_xy[:, idxs, torch.arange(batch_size).unsqueeze(1)].transpose(
                1, 2
            )
            pred_dxdy = pred_dxdy[
                :, idxs, torch.arange(batch_size).unsqueeze(1)
            ].transpose(1, 2)
        return GeneratorOutput(pred_dxdy, pred_xy), net_chooser_out, sampled_gen_idxs

    def get_samples(self, enc_h, num_samples=5):
        """Returns generator indexes of shape (batch size, num samples)"""
        if self.use_pinet:
            net_chooser_out = self.net_chooser(enc_h)
        else:
            net_chooser_out = self.net_prior.expand(enc_h.size(0), -1)
        dist = Categorical(logits=net_chooser_out)
        sampled_gen_idxs = dist.sample((num_samples,)).transpose(0, 1)
        return net_chooser_out, sampled_gen_idxs

    def forward_all(self, in_xy, in_dxdy, enc_h, noise, social_feats):
        """Runs all generators against the current batch.

        Args:
            in_xy: Input positions of shape (inp_len, batch_size, 2)
            in_dxdy: Input offsets of shape (inp_len, batch_size, 2)
            enc_h: Hidden state to initialize the decoders (LSTMs) with
                Shape (batch_size, enc dim).
            noise: Noise tensor (num_samples, batch_size, self.z_size).

        Returns:
            Two tensors of shape (pred_len, num_samples, num_gens, batch_size, 2)
             with predictions (positions and offsets) for every generator.
        """
        n_samples, b, z_size = noise.shape

        noise = noise.flatten(0, 1)
        enc_h = enc_h.repeat(n_samples, 1)
        in_xy = in_xy.repeat(1, n_samples, 1)
        in_dxdy = in_dxdy.repeat(1, n_samples, 1)
        enc_to_dec_inp = torch.cat([enc_h, noise], -1)
        social_feats = social_feats.repeat(n_samples, 1)

        # Shape: (1, num_samples * batch_size, dec dim)
        dec_h = self.enc_h_to_dec_h(enc_to_dec_inp).unsqueeze(0)
        state_tuple = (dec_h, torch.zeros_like(dec_h))

        preds_rel, preds_abs = [], []
        for i, g in enumerate(self.gs):
            # Shape (pred_len, num_samples * batch_size, 2)
            pred_abs, pred_rel = g(
                in_xy[-1], in_dxdy[-1], noise, social_feats, state_tuple
            )
            preds_rel.append(pred_rel.reshape(self.pred_len, n_samples, b, 2))
            preds_abs.append(pred_abs.reshape(self.pred_len, n_samples, b, 2))

        out_xy = torch.stack(preds_abs, dim=2)
        out_dxdy = torch.stack(preds_rel, dim=2)
        return out_xy, out_dxdy

    def forward_all_seq(self, in_xy, in_dxdy, enc_h, noise, social_feats):
        """Runs all generators against the current batch. (iteratively)

        Args:
            in_xy: Input positions of shape (inp_len, batch_size, 2)
            in_dxdy: Input offsets of shape (inp_len, batch_size, 2)
            enc_h: Hidden state to initialize the decoders (LSTMs) with
                Shape (batch_size, enc dim).
            noise: Noise tensor (num_samples, batch_size, self.z_size).

        Returns:
            Two tensors of shape (pred_len, num_samples, num_gens, batch_size, 2)
             with predictions (positions and offsets) for every generator.
        """
        n_samples, b, z_size = noise.shape

        samples_abs, samples_rel = [], []
        for s in range(n_samples):
            enc_to_dec_inp = torch.cat([enc_h, noise[s]], -1)
            # Shape: (1, batch_size, dec dim)
            dec_h = self.enc_h_to_dec_h(enc_to_dec_inp).unsqueeze(0)
            state_tuple = (dec_h, torch.zeros_like(dec_h))

            preds_rel, preds_abs = [], []
            for i, g in enumerate(self.gs):
                # Shape (pred_len, batch_size, 2)
                pred_abs, pred_rel = g(
                    in_xy[-1], in_dxdy[-1], noise[s], social_feats, state_tuple
                )
                preds_rel.append(pred_rel)
                preds_abs.append(pred_abs)

            # Shape (pred_len, num_gens, batch_size, 2)
            samples_abs.append(torch.stack(preds_abs, dim=1))
            samples_rel.append(torch.stack(preds_rel, dim=1))
        return torch.stack(samples_abs, 1), torch.stack(samples_rel, 1)

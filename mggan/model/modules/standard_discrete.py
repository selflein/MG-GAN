import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from mggan.model.modules.cnn import AttentionGlobal
from mggan.model.modules.social import SocialAttention
from mggan.model.modules.social_gan import PoolHiddenNet
from mggan.utils import make_mlp, get_global_noise
from mggan.model.modules.common_modules import (
    TrajectoryEncoder,
    RelativeDecoder,
    get_input,
    GeneratorOutput,
)


class DiscreteLatentGenerator(nn.Module):
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
        super().__init__()
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

        self.decoder = RelativeDecoder(
            pred_len=pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            num_layers=1,
            social_feat_size=encoder_h_dim if social_feat_size > 0 else 0,
            z_size=z_size,
            dropout=0.0,
            inp_format=inp_format,
        )

        self.n_gs = num_gens
        self.pred_len = pred_len
        self.enc_h_to_dec_h = make_mlp(
            [
                encoder_h_dim + z_size + scene_dim + z_size + social_feat_size,
                decoder_h_dim,
            ],
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

        self.one_hot_sample_encoder = make_mlp([num_gens, z_size, z_size])
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
            noise: Optional fixed noise tensor for inference of shape
                (batch size, noise_dim).
            all_gen_out: If True all generators are outputting a prediction for
                every element in the batch.

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
            out_xy, out_dxdy = [], []
            for i in range(num_samples):
                g_xy, g_dxdy = [], []
                for j in range(self.n_gs):
                    g_one_hot = (
                        F.one_hot(torch.tensor(j), self.n_gs)[None, :]
                        .repeat(batch_size, 1)
                        .float()
                    )
                    encoded_one_hot = self.one_hot_sample_encoder(g_one_hot)
                    inp_h = torch.cat([enc_h, encoded_one_hot], 1)
                    with torch.no_grad():
                        pred_xy, pred_dxdy = self.forward_all(
                            in_xy,
                            in_dxdy,
                            inp_h,
                            noise=noise[i],
                            social_feats=social_feats,
                        )

                    g_xy.append(pred_xy)
                    g_dxdy.append(pred_dxdy)
                out_xy.append(torch.stack(g_xy, 1))
                out_dxdy.append(torch.stack(g_dxdy, 1))

            out_xy = torch.stack(out_xy, 1)
            out_dxdy = torch.stack(out_dxdy, 1)
            net_chooser_out, sampled_gen_idxs = self.get_samples(enc_h, num_samples)

        else:
            with torch.no_grad():
                net_chooser_out, sampled_gen_idxs = self.get_samples(enc_h, num_samples)

            # Shape: (batch size, num_samples, num_gens) one-hot
            sampled_gen_idxs_one_hot = F.one_hot(sampled_gen_idxs, self.n_gs).float()
            out_xy, out_dxdy = [], []
            for i in range(num_samples):
                encoded_idxs = self.one_hot_sample_encoder(
                    sampled_gen_idxs_one_hot[:, i]
                )
                inp_h = torch.cat([enc_h, encoded_idxs], 1)
                pred_xy, pred_dxdy = self.forward_all(
                    in_xy,
                    in_dxdy,
                    inp_h,
                    noise=noise[i],
                    social_feats=social_feats,
                )
                out_xy.append(pred_xy)
                out_dxdy.append(pred_dxdy)

            # Shape (pred_len, num_samples, batch_size, 2)
            out_xy = torch.stack(out_xy, dim=1)
            out_dxdy = torch.stack(out_dxdy, dim=1)

        return GeneratorOutput(out_dxdy, out_xy), net_chooser_out, sampled_gen_idxs

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
            enc_h: Hidden state to initialize the decoders (LSTMs) with.
            noise: Noise tensor.

        Returns:
            Two tensors of shape (pred_len, batch_size, 2) with
             predictions (positions and offsets) for every generator.
        """
        enc_to_dec_inp = torch.cat([enc_h, noise], -1)
        dec_h = self.enc_h_to_dec_h(enc_to_dec_inp).unsqueeze(0)
        state_tuple = (dec_h, torch.zeros_like(dec_h))

        pred_abs, pred_rel = self.decoder(
            in_xy[-1], in_dxdy[-1], noise, social_feats, state_tuple
        )

        return pred_abs, pred_rel

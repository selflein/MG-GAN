from copy import deepcopy

import torch
import torch.nn as nn

from mggan.model.modules.cnn import AttentionGlobal
from mggan.model.modules.social import SocialAttention
from mggan.model.modules.social_gan import PoolHiddenNet
from mggan.model.modules.common_modules import TrajectoryEncoder


class MultiDiscriminatorTrajectory(nn.Module):
    def __init__(
        self,
        num_gens,
        num_discs,
        unbound_output,
        h_dim,
        inp_format,
        pred_len,
        gan_type,
        global_disc,
        scene_dim,
        pool_type="sgan",
    ):
        super(MultiDiscriminatorTrajectory, self).__init__()
        assert inp_format in ("rel", "abs", "abs_rel")
        assert gan_type in ("probgan", "mgan", "infogan", "gan")
        if not global_disc:
            print("Not using global discriminator.")
        self.inp_format = inp_format
        self.unbound_output = unbound_output
        self.n_ds = num_discs
        self.gan_type = gan_type
        self.global_disc = global_disc

        self.inp_size = 4 if inp_format == "abs_rel" else 2
        self.in_encoder = TrajectoryEncoder(
            hidden_size=h_dim,
            inp_size=self.inp_size,
            num_layers=1,
            embedding_dim=h_dim,
            return_hc=False,
        )

        self.in_encoder_fc = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim // 2, h_dim // 2),
        )

        self.pred_encoder = nn.Sequential(
            nn.Linear(pred_len * self.inp_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, h_dim // 2),
        )

        if self.global_disc:
            if pool_type == "sways":
                self.social = SocialAttention(h_dim, h_dim)
            else:
                self.social = PoolHiddenNet(
                    embedding_dim=16,
                    h_dim=h_dim,
                    mlp_dim=h_dim,
                    bottleneck_dim=h_dim,
                )
            h_dim *= 2

        if scene_dim > 0:
            self.scene_encoder = AttentionGlobal(
                noise_attention_dim=0, PhysFeature=True, num_layers=2, channels_cnn=8
            )
            h_dim += scene_dim

        self.discs = nn.ModuleList()
        for _ in range(num_discs):
            layers = [
                nn.Linear(h_dim, h_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim // 2, 1),
            ]
            if not self.unbound_output:
                layers.append(nn.Sigmoid())
            self.discs.append(nn.Sequential(*layers))

        if gan_type == "probgan":
            self.discs_hist = deepcopy(self.discs)

            # Freeze weights since only used as storage for history
            for disc in self.discs_hist:
                for p in disc.parameters():
                    p.requires_grad = False

            self.len_hist = 1.0
            self.eps = 1e-3
        elif gan_type == "infogan":
            self.code_reconstructor = nn.Sequential(
                nn.Linear(h_dim, h_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim // 2, 3),
            )
        elif gan_type == "mgan":
            self.gen_id_reconstructor = nn.Sequential(
                nn.Linear(h_dim, h_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim // 2, num_gens),
            )

        self.eps = 1e-7
        self.len_hist = 1.0

    def encode(self, in_xy, in_dxdy, pred_xy, pred_dxdy, mask=None):
        if self.inp_format == "rel":
            in_encoder_inp = in_dxdy
            pred_encoder_inp = pred_dxdy
        elif self.inp_format == "abs":
            in_encoder_inp = in_xy
            pred_encoder_inp = pred_xy
        else:
            in_encoder_inp = torch.cat([in_xy, in_dxdy], dim=-1)
            pred_encoder_inp = torch.cat([pred_xy, pred_dxdy], dim=-1)

        in_enc = self.in_encoder(in_encoder_inp)
        in_enc = self.in_encoder_fc(in_enc)

        pred_len, n_samples, b, _ = pred_xy.shape
        total_samples = in_xy.size(1) * n_samples

        # Shape: (n_samples, b, pred_len, 2)
        pred_encoder_inp = pred_encoder_inp.permute(1, 2, 0, 3)
        pred_enc = self.pred_encoder(pred_encoder_inp.reshape(n_samples * b, -1))
        # If masked then only use history by setting the encoding of the
        # future to 0
        if mask is not None:
            padded_pred_enc = torch.zeros(total_samples, pred_enc.size(1))
            padded_pred_enc[mask.repeat(n_samples)] = pred_enc
            pred_enc = padded_pred_enc

        # Shape (n_samples * batch_size, enc dim)
        enc = torch.cat([in_enc.repeat(n_samples, 1), pred_enc], dim=1)
        return enc

    def forward(
        self,
        in_xy,
        in_dxdy,
        pred_xy,
        pred_dxdy,
        seq_start_end,
        return_all=False,
        img=None,
        mask=None,
    ):
        """
        Args:
            in_xy: Shape (pred_len, batch_size, 2)
            in_dxdy: Shape (pred_len, batch_size, 2)
            pred_xy: Shape (pred_len, num_samples, b, 2)
            pred_dxdy: Shape (pred_len, num_samples, b, 2)
            seq_start_end:
            return_all:
            img:
            mask:

        Returns:
            output: Shape (batch_size, n_samples)
            branch_out: [Optional] Shape (batch_size, n_samples, branch_shape)
        """
        if len(pred_xy.shape) == 3:
            pred_xy = pred_xy.unsqueeze(1)
            pred_dxdy = pred_dxdy.unsqueeze(1)

        pred_len, n_samples, b, _ = pred_xy.shape

        # Shape (n_samples * batch_size, enc dim)
        enc = self.encode(in_xy, in_dxdy, pred_xy, pred_dxdy, mask)
        if self.global_disc:
            soc = self.social(
                in_xy.repeat(1, n_samples, 1),
                in_dxdy.repeat(1, n_samples, 1),
                enc,
                seq_start_end * n_samples,
            )
            classifier_inp = torch.cat([soc, enc], dim=1)
        else:
            classifier_inp = enc

        if mask is not None:
            classifier_inp = classifier_inp[mask.repeat(n_samples)]

        if img is not None:
            if mask is not None:
                img = img[mask]
            scene_encoding = self.scene_encoder(img).repeat(n_samples, 1)
            classifier_inp = torch.cat([classifier_inp, scene_encoding], 1)

        y = []
        for disc in self.discs:
            y.append(disc(classifier_inp))
        output = torch.cat(y, dim=1)

        if not self.unbound_output:
            output = output * (1 - 2 * self.eps) + self.eps

        if not return_all:
            output = output.mean(1)

        output = output.reshape(n_samples, b).t()
        if self.gan_type in ("probgan", "gan"):
            return output
        elif self.gan_type == "mgan":
            branch_out = self.gen_id_reconstructor(classifier_inp)
        elif self.gan_type == "infogan":
            branch_out = self.code_reconstructor(classifier_inp)
        else:
            raise NotImplementedError()

        return output, branch_out.reshape(n_samples, b, -1).transpose(0, 1)

    def forward_by_hist(
        self, in_xy, in_dxdy, pred_xy, pred_dxdy, seq_start_end, img=None
    ):
        enc = self.encode(in_xy, in_dxdy, pred_xy, pred_dxdy)
        if self.global_disc:
            soc = self.social(in_xy[0], enc, seq_start_end)
            classifier_inp = torch.cat([soc, enc], dim=1)
        else:
            classifier_inp = enc

        if img is not None:
            scene_encoding = self.scene_encoder(img, enc)
            classifier_inp = torch.cat([classifier_inp, scene_encoding], 1)

        y = []
        for disc in self.discs:
            y.append(disc(classifier_inp))
        output = torch.cat(y, dim=1)

        if not self.unbound_output:
            output = output * (1 - 2 * self.eps) + self.eps

        output = output.mean(1)
        return output

    def update_hist(self):
        self.len_hist += 1
        alpha = 1.0 / self.len_hist

        for d_hist, d in zip(self.discs_hist, self.discs):
            name_weight_mapping = {}

            # Get all weight tensors together with their name
            for name, tensor in d.named_parameters():
                if "weight" in name or "bias" in name:
                    name_weight_mapping[name] = tensor.data

            # Update the history tensors
            for name, tensor in d_hist.named_parameters():
                if "weight" in name or "bias" in name:
                    tensor.data = (
                        tensor.data * (1 - alpha) + name_weight_mapping[name] * alpha
                    )

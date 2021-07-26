""" From https://github.com/agrimgupta92/sgan/blob/master/sgan/models.py """

from copy import deepcopy

import torch
import torch.nn as nn

from mggan.utils import relative_to_abs, make_mlp
from mggan.model.modules.common_modules import GeneratorOutput


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape)
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim),
            torch.zeros(self.num_layers, batch, self.h_dim),
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""

    def __init__(
        self,
        seq_len,
        embedding_dim=64,
        h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        pooling_type="pool_net",
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if pool_every_timestep:
            if pooling_type == "pool_net":
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
            elif pooling_type == "spool":
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size,
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims, activation=activation, batch_norm=batch_norm, dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""

    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, h_dim, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, in_xy, in_dxdy, h_states, seq_start_end):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        end_pos = in_xy[-1]
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""

    def __init__(
        self,
        h_dim=64,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        neighborhood_size=2.0,
        grid_size=8,
        pool_dim=None,
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size)
            * self.grid_size
        )
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size)
            * self.grid_size
        )
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(
                seq_start_end
            )
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = (curr_end_pos[:, 0] >= bottom_right[:, 0]) + (
                curr_end_pos[:, 0] <= top_left[:, 0]
            )
            y_bound = (curr_end_pos[:, 1] >= top_left[:, 1]) + (
                curr_end_pos[:, 1] <= bottom_right[:, 1]
            )

            within_bound = x_bound + y_bound
            within_bound[0 :: num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        num_gs=1,
        embedding_dim=16,
        encoder_h_dim=32,
        decoder_h_dim=32,
        mlp_dim=64,
        num_layers=1,
        noise_dim=(8,),
        noise_type="gaussian",
        noise_mix_type="global",
        pooling_type="pool_net",
        pool_every_timestep=False,
        dropout=0.0,
        bottleneck_dim=8,
        activation="relu",
        batch_norm=False,
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == "none":
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.n_gs = num_gs

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.gs = nn.ModuleList()
        for i in range(num_gs):
            decoder = Decoder(
                pred_len,
                embedding_dim=embedding_dim,
                h_dim=decoder_h_dim,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                pool_every_timestep=pool_every_timestep,
                dropout=dropout,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                pooling_type=pooling_type,
                grid_size=grid_size,
                neighborhood_size=neighborhood_size,
            )
            setattr(self, f"G_{i}", decoder)
            self.gs.append(decoder)

        if pooling_type == "pool_net":
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
            )
        elif pooling_type == "spool":
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size,
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim,
                mlp_dim,
                decoder_h_dim - self.noise_first_dim,
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == "global":
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == "global":
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim
            or self.pooling_type
            or self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(
        self, obs_traj, obs_traj_rel, seq_start_end, all_gen_out=True, user_noise=None
    ):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        seq_start_end = torch.tensor(seq_start_end, requires_grad=False)

        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1
            )
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        # Predict Trajectory
        preds_rel = []
        preds_abs = []
        if all_gen_out:
            for g in self.gs:
                decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim)
                state_tuple = (decoder_h, decoder_c)
                pred_traj_fake_rel, _ = g(
                    last_pos, last_pos_rel, state_tuple, seq_start_end
                )
                preds_rel.append(pred_traj_fake_rel)
                preds_abs.append(relative_to_abs(pred_traj_fake_rel, last_pos))
            preds_rel = torch.stack(preds_rel, 1)
            preds_abs = torch.stack(preds_abs, 1)
        else:
            sp_size = (batch - 1) // self.n_gs + 1
            sub_batches_split = torch.split(seq_start_end, sp_size, dim=0)
            sb_sizes = [
                sb_split[-1][1].item() - sb_split[0][0].item()
                for sb_split in sub_batches_split
            ]

            dec_h_split = torch.split_with_sizes(decoder_h, sb_sizes, dim=1)
            last_pos_split = torch.split_with_sizes(last_pos, sb_sizes, dim=0)
            last_pos_rel_split = torch.split_with_sizes(last_pos_rel, sb_sizes, dim=0)

            for g, h, xy, dxdy, seq_s_e in zip(
                self.gs,
                dec_h_split,
                last_pos_split,
                last_pos_rel_split,
                sub_batches_split,
            ):
                state_tuple = (h, torch.zeros_like(h))
                seq_s_e = seq_s_e - seq_s_e[0, 0]
                pred_traj_fake_rel, _ = g(
                    xy,
                    dxdy,
                    state_tuple,
                    seq_s_e,
                )
                preds_rel.append(pred_traj_fake_rel)
                preds_abs.append(relative_to_abs(pred_traj_fake_rel, xy))
            preds_rel = torch.cat(preds_rel, 1)
            preds_abs = torch.cat(preds_abs, 1)
        return GeneratorOutput(preds_rel, preds_abs)


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        num_ds=1,
        unbound_output=False,
        embedding_dim=16,
        h_dim=48,
        mlp_dim=64,
        num_layers=1,
        activation="relu",
        batch_norm=False,
        dropout=0.0,
        d_type="local",
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.n_ds = num_ds
        self.unbound_output = unbound_output

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.discs = nn.ModuleList()
        for i in range(num_ds):
            real_classifier = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
            setattr(self, f"D_{i}", real_classifier)
            self.discs.append(real_classifier)

        self.discs_hist = deepcopy(self.discs)

        # Freeze weights since only used as storage for history
        for disc in self.discs_hist:
            for p in disc.parameters():
                p.requires_grad = False

        if d_type == "global":
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
            )

        self.len_hist = 1.0
        self.eps = 1e-3

    def encode(self, in_xy, in_dxdy, gt_xy, gt_dxdy, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        traj_rel = torch.cat([in_dxdy, gt_dxdy], dim=0)
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == "local":
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, in_xy[0])
        return classifier_input

    def forward(
        self, in_xy, in_dxdy, pred_xy, pred_dxdy, return_all=False, seq_start_end=None
    ):
        enc = self.encode(
            in_xy, in_dxdy, pred_xy, pred_dxdy, seq_start_end=seq_start_end
        )

        y = []
        for disc in self.discs:
            y.append(disc(enc))
        output = torch.cat(y, dim=1)

        if not self.unbound_output:
            output = torch.sigmoid(output)
            output = output * (1 - 2 * self.eps) + self.eps

        if return_all:
            return output

        output = output.mean(1)
        return output

    def forward_by_hist(self, in_xy, in_dxdy, pred_xy, pred_dxdy, seq_start_end=None):
        enc = self.encode(
            in_xy, in_dxdy, pred_xy, pred_dxdy, seq_start_end=seq_start_end
        )

        y = []
        for disc in self.discs_hist:
            y.append(disc(enc))
        output = torch.cat(y, dim=1)

        if not self.unbound_output:
            output = torch.sigmoid(output)
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

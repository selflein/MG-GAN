import torch
import torch.nn as nn
import numpy as np


def make_mlp(dim_list, activation_list, batch_norm=False, dropout=0):
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        activation = activation_list[index]
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))
        index += 1
    return nn.Sequential(*layers)


class VisualNetwork(nn.Module):
    """VisualNetwork is the parent class for the attention and goal networks"""

    def __init__(
        self,
        decoder_h_dim=128,
        dropout=0.0,
        batch_norm=False,
        mlp_dim=32,
        img_scaling=0.25,
        final_embedding_dim=4,
        margin_in=16,
        num_layers=1,
        batch_norm_cnn=True,
        non_lin_cnn="relu",
        img_type="tiny_image",
        skip_connection=False,
        channels_cnn=4,
        social_pooling=False,
        **kwargs
    ):
        super(VisualNetwork, self).__init__()
        self.__dict__.update(locals())

    def init_cnn(self):
        self.CNN = CNN(
            social_pooling=self.social_pooling,
            channels_cnn=self.channels_cnn,
            encoder_h_dim=self.decoder_h_dim,
            mlp=self.mlp_dim,
            insert_trajectory=False,
            PhysFeature=self.PhysFeature,
            margin_in=self.margin_in,
            dropout=self.dropout,
            batch_norm=self.batch_norm_cnn,
            non_lin_cnn=self.non_lin_cnn,
            num_layers=self.num_layers,
            in_channels=4,
            skip_connection=self.skip_connection,
        )


class AttentionNetwork(VisualNetwork):
    def __init__(self, noise_attention_dim=8, **kwargs):
        super(AttentionNetwork, self).__init__()
        VisualNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())
        self.PhysFeature = True
        self.skip_connection = False
        self.need_decoder = False

        self.init_cnn()
        self.final_embedding = self.CNN.bottleneck_dim + self.noise_attention_dim
        attention_dims = [
            self.CNN.bootleneck_channel,
            self.mlp_dim,
            self.CNN.bootleneck_channel,
        ]
        activation = ["leakyrelu", None]
        self.cnn_attention = make_mlp(
            attention_dims,
            activation_list=activation,
        )

    def get_noise(self, batch_size, type="gauss"):
        if type == "gauss":
            return torch.randn((1, batch_size, self.noise_attention_dim))
        elif type == "uniform":

            rand_num = torch.rand((1, batch_size, self.noise_attention_dim))
            return rand_num


class AttentionGlobal(AttentionNetwork):
    def __init__(self, **kwargs):
        super(AttentionNetwork, self).__init__()
        AttentionNetwork.__init__(self, **kwargs)
        self.__dict__.update(locals())

        self.init_cnn()

    def forward(self, features):
        visual_features = self.CNN(features)["Features"].permute(0, 2, 3, 1)
        batch_size, hh, w, c = visual_features.size()
        visual_features = visual_features.view(batch_size, -1, c)
        attention_scores = self.cnn_attention(visual_features)
        attention_vec = attention_scores.softmax(dim=2)
        attention_out = attention_vec * visual_features
        return attention_out.sum(-1)


class Conv_Blocks(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        filter_size=3,
        batch_norm=False,
        non_lin="tanh",
        dropout=0.0,
        first_block=False,
        last_block=False,
        skip_connection=False,
    ):
        super(Conv_Blocks, self).__init__()
        self.skip_connection = skip_connection
        self.last_block = last_block
        self.first_block = first_block
        self.Block = nn.Sequential()
        self.Block.add_module(
            "Conv_1", nn.Conv2d(input_dim, output_dim, filter_size, 1, 1)
        )
        if batch_norm:
            self.Block.add_module("BN_1", nn.BatchNorm2d(output_dim))
        if non_lin == "tanh":
            self.Block.add_module("NonLin_1", nn.Tanh())
        elif non_lin == "relu":
            self.Block.add_module("NonLin_1", nn.ReLU())
        elif non_lin == "leakyrelu":
            self.Block.add_module("NonLin_1", nn.LeakyReLU())
        else:
            assert False, "non_lin = {} not valid: 'tanh', 'relu', 'leakyrelu'".format(
                non_lin
            )

        self.Block.add_module(
            "Pool",
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
        )
        if dropout > 0:
            self.Block.add_module("Drop", nn.Dropout2d(dropout))

    def forward(self, x):
        if self.skip_connection:
            if not self.first_block:
                x, skip_con_list = x
            else:
                skip_con_list = []

        x = self.Block(x)
        if self.skip_connection:
            if not self.last_block:
                skip_con_list.append(x)
            x = [x, skip_con_list]

        return x


class CNN(nn.Module):
    def __init__(
        self,
        social_pooling=False,
        channels_cnn=4,
        mlp=32,
        encoder_h_dim=16,
        insert_trajectory=False,
        PhysFeature=False,
        margin_in=32,
        num_layers=3,
        dropout=0.0,
        batch_norm=False,
        non_lin_cnn="tanh",
        in_channels=3,
        skip_connection=False,
    ):
        super(CNN, self).__init__()

        self.social_pooling = social_pooling

        self.in_traj = insert_trajectory
        self.skip_connection = skip_connection
        self.PhysFeature = PhysFeature
        self.bottleneck_dim = int(margin_in / 2 ** (num_layers - 1)) ** 2

        self.non_lin = non_lin_cnn
        self.encoder = nn.Sequential()

        layer_out = channels_cnn
        self.encoder.add_module(
            "ConvBlock_1",
            Conv_Blocks(
                in_channels,
                channels_cnn,
                dropout=dropout,
                batch_norm=batch_norm,
                non_lin=self.non_lin,
                first_block=True,
                skip_connection=self.skip_connection,
            ),
        )
        layer_in = layer_out
        for layer in np.arange(2, num_layers + 1):

            if layer != num_layers:
                layer_out = layer_in * 2
                last_block = False
            else:
                layer_out = layer_in
                last_block = True
            self.encoder.add_module(
                "ConvBlock_%s" % layer,
                Conv_Blocks(
                    layer_in,
                    layer_out,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    non_lin=self.non_lin,
                    skip_connection=self.skip_connection,
                    last_block=last_block,
                ),
            )
            layer_in = layer_out

        self.bootleneck_channel = layer_out

        if self.in_traj:
            self.traj2cnn = make_mlp(
                dim_list=[encoder_h_dim, mlp, self.bottleneck_dim],
                activation_list=["tanh", "tanh"],
            )
        if self.social_pooling:
            self.social_states = make_mlp(
                dim_list=[encoder_h_dim, mlp, self.bottleneck_dim],
                activation_list=["tanh", "tanh"],
            )
        self.init_weights()

    def init_weights(self):
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in")
                m.bias.data.fill_(0.01)

        def init_xavier(m):
            if type(m) == [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        if self.non_lin in ["relu", "leakyrelu"]:
            self.apply(init_kaiming)
        elif self.non_lin == "tanh":
            self.apply(init_xavier)
        else:
            assert False, "non_lin not valid for initialisation"

    def forward(self, image):
        output = {}
        enc = self.encoder(image)

        if self.PhysFeature:
            output.update(Features=enc)

        return output

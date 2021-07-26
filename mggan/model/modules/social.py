""" This is from https://github.com/amiryanj/socialways"""

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(AttentionPooling, self).__init__()
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.W = nn.Linear(h_dim, f_dim, bias=True)

    def forward(self, f, h, sub_batches):
        Wh = self.W(h)
        S = torch.zeros_like(h)
        for sb in sub_batches:
            N = sb[1] - sb[0]
            if N == 1:
                continue

            for ii in range(sb[0], sb[1]):
                fi = f[ii, sb[0] : sb[1]]
                sigma_i = torch.bmm(fi.unsqueeze(1), Wh[sb[0] : sb[1]].unsqueeze(2))
                sigma_i[ii - sb[0]] = -1000

                attentions = torch.softmax(sigma_i.squeeze(), dim=0)
                S[ii] = torch.mm(attentions.view(1, N), h[sb[0] : sb[1]])

        return S


class EmbedSocialFeatures(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmbedSocialFeatures, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
        )

    def forward(self, ftr_list, sub_batches):
        embedded_features = self.fc(ftr_list)
        return embedded_features


def DCA(xA_4d, xB_4d):
    dp = xA_4d[:2] - xB_4d[:2]
    dv = xA_4d[2:] - xB_4d[2:]
    ttca = torch.dot(-dp, dv) / (torch.norm(dv) ** 2 + 1e-6)
    # ttca = torch.max(ttca, 0)
    dca = torch.norm(dp + ttca * dv)
    return dca


def Bearing(xA_4d, xB_4d):
    dp = xA_4d[:2] - xB_4d[:2]
    v = xA_4d[2:]
    cos_theta = torch.dot(dp, v) / (torch.norm(dp) * torch.norm(v) + 1e-6)
    return cos_theta


def DCA_MTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]
    Dv = D_4d[:, :, 2:]
    DOT_Dp_Dv = torch.mul(Dp[:, :, 0], Dv[:, :, 0]) + torch.mul(
        Dp[:, :, 1], Dv[:, :, 1]
    )
    Dv_sq = (
        torch.mul(Dv[:, :, 0], Dv[:, :, 0]) + torch.mul(Dv[:, :, 1], Dv[:, :, 1]) + 1e-6
    )
    TTCA = -torch.div(DOT_Dp_Dv, Dv_sq)
    DCA = torch.zeros_like(Dp)
    DCA[:, :, 0] = Dp[:, :, 0] + TTCA * Dv[:, :, 0]
    DCA[:, :, 1] = Dp[:, :, 1] + TTCA * Dv[:, :, 1]
    DCA = torch.norm(DCA, dim=2)
    return DCA


def BearingMTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]  # NxNx2
    v = x_4d[:, 2:].unsqueeze(1).repeat(1, x_4d.shape[0], 1)  # => NxNx2
    DOT_Dp_v = Dp[:, :, 0] * v[:, :, 0] + Dp[:, :, 1] * v[:, :, 1]
    COS_THETA = torch.div(DOT_Dp_v, torch.norm(Dp, dim=2) * torch.norm(v, dim=2) + 1e-6)
    return COS_THETA


def SocialFeatures(x):
    N = x.shape[0]  # x is NxTx4 tensor

    x_ver_repeat = x[:, -1].unsqueeze(0).repeat(N, 1, 1)
    x_hor_repeat = x[:, -1].unsqueeze(1).repeat(1, N, 1)
    Dx_mat = x_hor_repeat - x_ver_repeat

    l2_dist_MTX = Dx_mat[:, :, :2].norm(dim=2)
    bearings_MTX = BearingMTX(x[:, -1], Dx_mat)
    dcas_MTX = DCA_MTX(x[:, -1], Dx_mat)
    sFeatures_MTX = torch.stack([l2_dist_MTX, bearings_MTX, dcas_MTX], dim=2)

    return sFeatures_MTX  # directly return the Social Features Matrix


class SocialAttention(nn.Module):
    def __init__(self, social_feat_size, hidden_size):
        super(SocialAttention, self).__init__()
        self.feature_embedder = EmbedSocialFeatures(3, social_feat_size)
        self.attention = AttentionPooling(hidden_size, social_feat_size)

    def forward(self, in_xy, in_dxdy, enc_h, sub_batches):
        b = enc_h.size(0)
        # Shape: Nx1x4
        inp = torch.cat((in_xy[-1], in_dxdy[-1]), -1).unsqueeze(1)
        social_feats = SocialFeatures(inp)
        emb_social_feats = self.feature_embedder(social_feats, sub_batches)
        att_social_feats = self.attention(emb_social_feats, enc_h, sub_batches).view(
            b, -1
        )

        return att_social_feats

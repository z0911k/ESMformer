import torch
import torch.nn as nn
from einops import rearrange

from model.SMFE import SMFE, Mlp
from model.MILF import MVF, MILF
from model.MCLF import MCLF

from common.opt import opts

opt = opts().parse()


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Single-view Multi-level Feature Extraction
        self.ssfe1 = SMFE(args)
        self.ssfe2 = SMFE(args)
        self.ssfe3 = SMFE(args)
        self.ssfe4 = SMFE(args)

        ## multi_view Fusion
        self.mvf1 = MVF(args)
        self.mvf2 = MVF(args)
        self.mvf3 = MVF(args)

        self.norm1 = nn.LayerNorm(args.channel)
        self.norm2 = nn.LayerNorm(args.channel)
        self.norm3 = nn.LayerNorm(args.channel)

        ##  Multi-view Intra-level Fusion
        self.milf = MILF(depth=args.milf, embed_dim=args.channel, length=args.frames)

        ##Multi-view Cross-level Fusion
        self.mclf = MCLF(args.mclf, args.channel, args.d_hid, length=args.frames)

        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel * 3, momentum=0.1),
            nn.Conv1d(args.channel * 3, 3 * args.out_joints, kernel_size=1)
        )

        self.mlp = Mlp(in_features=args.frames, hidden_features=64, out_features=1, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        B, F, M, J, C = x.shape  # batch, frames, multi-view, joints, dimension
        x = rearrange(x, 'b f m j c -> b m (j c) f').contiguous()
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x4 = x[:, 3, :, :]

        x11, x12, x13 = self.ssfe1(x1)
        x21, x22, x23 = self.ssfe2(x2)
        x31, x32, x33 = self.ssfe3(x3)
        x41, x42, x43 = self.ssfe4(x4)

        # mvf & mife
        x1 = torch.cat((x11.unsqueeze(1), x21.unsqueeze(1), x31.unsqueeze(1), x41.unsqueeze(1)), dim=1)
        x2 = torch.cat((x12.unsqueeze(1), x22.unsqueeze(1), x32.unsqueeze(1), x42.unsqueeze(1)), dim=1)
        x3 = torch.cat((x13.unsqueeze(1), x23.unsqueeze(1), x33.unsqueeze(1), x43.unsqueeze(1)), dim=1)

        x1 = self.norm1(self.mvf1(x1).squeeze(1) + x11 + x21 + x31 + x41)
        x2 = self.norm2(self.mvf2(x2).squeeze(1) + x12 + x22 + x32 + x42)
        x3 = self.norm3(self.mvf3(x3).squeeze(1) + x13 + x23 + x33 + x43)

        x1, x2, x3 = self.milf(x1, x2, x3)

        ##  mcff
        x = self.mclf(x1, x2, x3)

        ## Regression
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)

        if opt.self_supervised:
            x = self.mlp(x)

        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()
        return x


opt = opts().parse()
model = Model(opt)
x = torch.rand((16, 27, 4, 17, 2))
print(model(x).shape)

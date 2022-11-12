import torch
import torch.nn as nn
from functools import partial

from model.SMFE import Transformer


## TODO: multi_view Fusion
class MVF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=args.mvf_kernel, stride=1, padding=int((args.mvf_kernel-1)/2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(args.channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)

        return x


## TODO: Multi-view intra-level feature enhancer
class MIFE_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth='222', drop_rate=[0.1,0.1,0.1], length=27):
        super().__init__()

        self.trans1 = Transformer(depth=int(depth[0]), embed_dim=dim, mlp_hidden_dim=dim * 2, h=num_heads, drop_rate=drop_rate[0],
                                  length=length)
        self.trans2 = Transformer(depth=int(depth[1]), embed_dim=dim, mlp_hidden_dim=dim * 2, h=num_heads, drop_rate=drop_rate[1],
                                  length=length)
        self.trans3 = Transformer(depth=int(depth[2]), embed_dim=dim, mlp_hidden_dim=dim * 2, h=num_heads, drop_rate=drop_rate[2],
                                  length=length)


    def forward(self, x_1, x_2, x_3):
        x_1 = x_1 + self.trans1(x_1)
        x_2 = x_2 + self.trans2(x_2)
        x_3 = x_3 + self.trans3(x_3)

        return x_1, x_2, x_3


class MIFE(nn.Module):
    def __init__(self, depth=[2,2,2], embed_dim=512, drop_rate=0.1, length=27):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, length, embed_dim))

        self.pos_drop_1 = nn.Dropout(p=drop_rate)
        self.pos_drop_2 = nn.Dropout(p=drop_rate)
        self.pos_drop_3 = nn.Dropout(p=drop_rate)

        self.MIFE_blocks = MIFE_Block(dim=embed_dim, num_heads=8, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                      depth=depth, mlp_hidden_dim=embed_dim * 2, length=length)

        self.norm = norm_layer(embed_dim * 3)

    def forward(self, x_1, x_2, x_3):
        x_1 += self.pos_embed_1
        x_2 += self.pos_embed_2
        x_3 += self.pos_embed_3

        x_1 = self.pos_drop_1(x_1)
        x_2 = self.pos_drop_2(x_2)
        x_3 = self.pos_drop_3(x_3)

        x_1, x_2, x_3 = self.MIFE_blocks(x_1, x_2, x_3)

        return x_1, x_2, x_3

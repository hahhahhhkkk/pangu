import torch
import torch.nn.functional as F
import torch.nn as nn
# 在最后一步反patch回原形状
#

class PatchRecover(torch.nn.Module):

    def __init__(self, up_channel=5, down_channel=4, out_channel=384, patch=(2, 2, 2), stride=(2, 2, 2)):
        super().__init__()
        # self.c3d = nn.ConvTranspose3d(in_channels=out_channel,out_channels=up_channel,kernel_size=patch,stride=stride)
        # self.c2d = nn.ConvTranspose2d(in_channels=out_channel,out_channels=down_channel,kernel_size=patch[1:],stride=stride[1:])
        self.pred_head_up = nn.Sequential(
            # 把 token 通道 D 先压到64，避免内存爆
            nn.Conv3d(out_channel, 64, kernel_size=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(8, 64),

            #
            nn.ConvTranspose3d(64, 64, kernel_size=patch, stride=patch, bias=False),
            nn.GELU(),
            nn.GroupNorm(8, 64),

            nn.Conv3d(64, up_channel, kernel_size=3, padding=1, bias=True),
        )
        # 网格域细化
        self.refine3d_up = nn.Sequential(
            nn.Conv3d(up_channel, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 16),
            nn.Conv3d(16, up_channel, kernel_size=1),
        )
        self.pred_head_down = nn.Sequential(
            # 把 token 通道 D 先压到64，避免内存爆
            nn.Conv2d(out_channel, 64, kernel_size=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(8, 64),

            # 关键：核=步长=patch_size，避免棋盘格
            nn.ConvTranspose2d(64, 64, kernel_size=patch[1:], stride=patch[1:], bias=False),
            nn.GELU(),
            nn.GroupNorm(8, 64),

            #
            nn.Conv2d(64, down_channel, kernel_size=3, padding=1, bias=True),
        )
        # 网格域细化（3D 卷积，稳定起见用 GroupNorm 而不是 BN）
        self.refine2d_down = nn.Sequential(
            nn.Conv2d(down_channel, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, down_channel, kernel_size=1),
        )

    def forward(self, x, z, h, w):
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], z, h, w)

        # up_out = self.c3d(x[:,:,1:,:,:])  # 区分高空和地面
        # down_out = self.c2d(x[:,:,0,:,:])
        up_out = self.pred_head_up(x[:, :, 1:, :, :])
        down_out = self.pred_head_down(x[:, :, 0, :, :])

        up_out = up_out + self.refine3d_up(up_out)
        down_out = down_out + self.refine2d_down(down_out)

        return up_out, down_out


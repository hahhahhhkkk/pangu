import torch
# 在数据输入到网络的准备 对hw减半和对zhw减半 并将特征维度映射到dim
# 地面数据也算是一个高度层 最后在高度层进行拼接
class PatchEmbedding:
    def __init__(self, patch_size, dim):
        self.conv = Conv3d(5, dim, kernel_size=patch_size, stride=patch_size)
        self.conv_surface = Conv2d(7, dim, kernel_size=patch_size[1:], stride=patch_size[1:])


        self.var_scale = nn.Parameter(torch.ones(5))

    def forward(self, x, surface):


        x = x * self.var_scale[None, :, None, None, None]

        x = self.conv(x)

        surface = self.conv_surface(surface)

        x = Concatenate(x, surface)

        x = x * self.lat_weight[None, None, :, None]

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x

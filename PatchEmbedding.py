import torch
# 在数据输入到网络的准备 对hw减半和对zhw减半 并将特征维度映射到dim
# 地面数据也算是一个高度层 最后在高度层进行拼接
class PatchEmbedding(torch.nn.Module):
    def __init__(self, up_channel=5, down_channel=7, out_channel=192, patch=(2, 2, 2), stride=(2, 2, 2)):
        super().__init__()
        self.c3d = torch.nn.Conv3d(in_channels=up_channel, out_channels=out_channel, kernel_size=patch, stride=stride)
        self.c2d = torch.nn.Conv2d(in_channels=down_channel, out_channels=out_channel, kernel_size=patch[1:],
                                   stride=stride[1:])

    def forward(self, up, down):
        # up = F.pad(up, (0, 0, 1, 0, 0, 0), mode='constant', value=0) 数据是奇数进行pad
        # down = F.pad(down, (1, 1, 1, 1), mode='constant', value=0)

        up = self.c3d(up)
        down = self.c2d(down)
        down = down.unsqueeze(2)  # 增加维度 匹配高度特征

        data = torch.cat([down, up], axis=2)
        data = data.permute(0, 2, 3, 4, 1)
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3], -1)  # 做成Transformer形式
        return data

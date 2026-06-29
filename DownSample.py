import torch
# 降采样 高宽减半 将减少的特征维度添加到特征维度 特征维度继续映射

class DownSample(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.liner = torch.nn.Linear(4 * dim, 2 * dim)
        self.norm = torch.nn.LayerNorm(4 * dim)

    def forward(self, x, z, h, w):
        x = x.reshape(x.shape[0], z, h, w, x.shape[2])
        _, _, h, w, _ = x.shape
        x = x.reshape(x.shape[0], z, h // 2, 2, w // 2, 2, x.shape[-1])

        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(x.shape[0], z * (w // 2) * (h // 2), 4 * x.shape[-1])

        x = self.norm(x)
        x = self.liner(x)

        return x
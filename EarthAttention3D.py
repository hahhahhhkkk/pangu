import torch
import numpy as np
# 自注意力模块

def mask_attention(attn, mask):
    if mask is None:
        return attn
    return attn + mask


class EarthAttention3D(torch.nn.Module):
    def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim * 3, bias=True)  # 三倍特征映射出qkv
        self.linear2 = torch.nn.Linear(dim, dim)  # 最后的注意力映射
        self.softmax = torch.nn.Softmax(dim=-1)  # 变为概率分布  谁更重要，权重就更大 当前像素/patch应该关注哪些位置
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.head_number = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.window_size = window_size
        self.mask_attention = mask_attention  # attention 与mask相加
        self.input_shape = input_shape
        self.type_window = (self.input_shape[0] // window_size[0]) * (
                    self.input_shape[1] // window_size[1])  # 在zh维度上区别每个窗口的位置
        self.earth_specific_bias = torch.empty(
            (2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0],
            # 附加特征(n*n的特殊映射,在zh的窗口数量,头数)
            self.type_window, self.head_number
        )
        self.earth_specific_bias = torch.nn.Parameter(self.earth_specific_bias)
        torch.nn.init.trunc_normal_(self.earth_specific_bias, std=0.02)

        self.position_index = self._construct_index()

    def _construct_index(self):
        coords_zi = torch.tensor(np.arange(self.window_size[0]))  # [0,1]
        coords_zj = -torch.tensor(np.arange(self.window_size[0])) * self.window_size[0]  # [0,-2]
        coords_hi = torch.tensor(np.arange(self.window_size[1]))
        coords_hj = -torch.tensor(np.arange(self.window_size[1])) * self.window_size[1]
        coords_w = torch.tensor(np.arange(self.window_size[2]))

        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))  # 记录hwz
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))  # 特殊编码的hwz

        coords_flatten_1 = torch.flatten(coords_1, 1)  # (3,n)
        coords_flatten_2 = torch.flatten(coords_2, 1)  # (3,n)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]  # (3,n,n)

        coords = coords.permute(1, 2, 0)

        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 1] *= 2 * self.window_size[2] - 1
        coords[:, :, 0] *= (2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1]

        self.position_index = torch.sum(coords, dim=-1)
        self.position_index = torch.flatten(self.position_index)
        return self.position_index.long()

    def forward(self, x, mask, z, h, w):  # 这里的zhw是进行window后每个维度的窗口数
        x = self.linear1(x)
        x_origin_shape = x.shape
        # (*b,窗口，特征）
        x = x.reshape(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)
        # (pkv,B,head,一个窗口的大小,*特征数)
        x = x.permute(2, 0, 3, 1, 4)
        q, k, v = torch.unbind(x, dim=0)

        q = q * self.scale
        attention = (q @ k.transpose(-2, -1))  # 会变成(B,head,n,n) # B 包含了batch *z *h *w
        EarthSpecificBias = self.earth_specific_bias[self.position_index]

        EarthSpecificBias = EarthSpecificBias.reshape(self.window_size[0] * self.window_size[1] * self.window_size[2],
                                                      self.window_size[0] * self.window_size[1] * self.window_size[2],
                                                      self.type_window, self.head_number)

        EarthSpecificBias = EarthSpecificBias.reshape(
            self.head_number,
            self.type_window,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2]
        )
        EarthSpecificBias = EarthSpecificBias.unsqueeze(0)
        EarthSpecificBias = EarthSpecificBias.permute(0, 2, 1, 3, 4)
        EarthSpecificBias = EarthSpecificBias.unsqueeze(2)

        # 拆出attention里的type_window 这个维度 与EarthSpecificBias的type_window相加
        attention = attention.reshape(-1, z * h, w, self.head_number,
                                      self.window_size[0] * self.window_size[1] * self.window_size[2],
                                      self.window_size[0] * self.window_size[1] * self.window_size[2])

        attention = attention + EarthSpecificBias
        attention = attention.reshape(attention.shape[0] * z * h * w, self.head_number,
                                      self.window_size[0] * self.window_size[1] * self.window_size[2],
                                      self.window_size[0] * self.window_size[1] * self.window_size[2])
        attention = self.mask_attention(attention, mask)  # mask要匹配的位置是B N N ，head数不需要匹配
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        x = attention @ v  # (B, heads, N, head_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

import torch
# 块 一层有很多块 每块选择实现ROLL操作

class EarthSpecificBlock(torch.nn.Module):
    def __init__(self, dim, dropout_rate, head, window_size, input_shape):
        super().__init__()
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.head = head
        self.window_size = window_size

        self.linear = MLP(dim, 0)
        self.drop = torch.nn.Dropout(dropout_rate)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.attention = EarthAttention3D(dim=dim, heads=head, dropout_rate=dropout_rate, window_size=window_size,
                                          input_shape=input_shape)

    def forward(self, x, z, h, w, roll):  # 这里的zhw是数据的长度 但是是patch后的
        shortcut = x
        x = x.reshape(x.shape[0], z, h, w, x.shape[2])
        origin_shape = x.shape

        # 窗口平移 并且将新的mask给attention 让它知道哪些属于窗口是边缘窗口
        if roll:
            x = torch.roll(x, shifts=(z // self.window_size[0], h // self.window_size[1], w // self.window_size[2]),
                           dims=(1, 2, 3))
            mask = gen_mask(Z=x.shape[1], H=x.shape[2], W=x.shape[3]).repeat(x.shape[0], 1, 1, 1, 1).reshape(-1, 1,
                                                                                                             self.window_size[
                                                                                                                 0] *
                                                                                                             self.window_size[
                                                                                                                 1] *
                                                                                                             self.window_size[
                                                                                                                 2],
                                                                                                             self.window_size[
                                                                                                                 0] *
                                                                                                             self.window_size[
                                                                                                                 1] *
                                                                                                             self.window_size[
                                                                                                                 2])
        else:
            mask = None

        x_window = x.reshape(x.shape[0], z // self.window_size[0], self.window_size[0], h // self.window_size[1],
                             self.window_size[1],
                             w // self.window_size[2], self.window_size[2], -1)

        x_window = x_window.permute(0, 1, 3, 5, 2, 4, 6, 7)
        # 做成(B,一个窗口的东西,特征维度) 这个形状
        x_window = x_window.reshape(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                    x_window.shape[-1])

        x_window = self.attention(x_window, mask, z // self.window_size[0], h // self.window_size[1],
                                  w // self.window_size[2])
        # 还原形状
        x_window = x_window.reshape(-1, z // self.window_size[0], h // self.window_size[1], w // self.window_size[2],
                                    self.window_size[0],
                                    self.window_size[1], self.window_size[2], x_window.shape[-1])

        x_window = x_window.permute(0, 1, 4, 2, 5, 3, 6, 7)

        x = x_window.reshape(origin_shape)  # 还原回（T,Z,H,W,dim）
        # 还原 ROLL操作
        if roll:
            x = torch.roll(x, shifts=(-z // self.window_size[0], -h // self.window_size[1], -w // self.window_size[2]),
                           dims=(1, 2, 3))

        # 做成Transformer形式输入
        x = x_window.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])

        x = shortcut + self.drop(self.norm1(x))
        x = x + self.drop(self.linear(self.norm2(x)))

        return x


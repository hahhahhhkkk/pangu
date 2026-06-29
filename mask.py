import torch
# 做出一个(B,一个窗口的东西,特征维度) 这个形状
def window_partition(x, window_size):
    B,Z,H,W,C = x.shape

    wz,wh,ww = window_size

    x = x.view(B,Z//wz,wz,H//wh,wh,W//ww,ww,C)

    x = x.permute( 0, 1, 3, 5,2, 4, 6, 7)

    windows = x.reshape(
        -1,
        wz*wh*ww,
        C
    )

    return windows


def gen_mask(
    Z=8,
    H=24,
    W=36,
    window_size=(2,2,2)
):

    wz,wh,ww = window_size

    img_mask = torch.zeros((1,Z,H,W,1),device=device)

    cnt = 0

    for z in range(0,Z,wz):    # 给每个窗口编号 0号窗口全是0
        for h in range(0,H,wh):
            for w in range(0,W,ww):

                img_mask[:, z:z+wz, h:h+wh,  w:w+ww, : ] = cnt
                cnt += 1

    img_mask = torch.roll(    # 同步窗口移动 ROLL
        img_mask,
        shifts=(wz//2,wh//2,ww//2),
        dims=(1,2,3)
    )

    mask_windows = window_partition(  # 把这个窗口标志 转化为patch形式
        img_mask,
        window_size
    )

    mask_windows = mask_windows.squeeze(-1) #

    attn_mask = (             # 变为两两对应的关系 是否为同一个窗口
        mask_windows.unsqueeze(1)
        -
        mask_windows.unsqueeze(2)
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask != 0,
        -100.0
    )

    attn_mask = attn_mask.masked_fill(
        attn_mask == 0,
        0.0
    )

    return attn_mask.unsqueeze(1)
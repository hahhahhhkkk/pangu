import torch
from torch import nn
import numpy as np


device = torch.device("cuda")

def normalize(up_file,down_file):
    input_up, input_down = np.load(up_file), np.load(down_file)
    test_up, test_down = input_up[1168:, :, :, :48, :72], input_down[1168:, :, :48, :72]
    input_up, input_down = input_up[:1168, :, :, :48, :72], input_down[:1168, :, :48, :72]
    # input_up = np.pad(input_up, pad_width=((0,0), (0,0), (0,1), (0,3), (0,3)), mode='constant')
    # input_down= np.pad(input_down, pad_width=((0,0), (0,0), (0,3), (0,3)), mode='constant')

    upper_mean = input_up.mean(
        axis=(0, 3, 4)
    )

    upper_std = input_up.std(
        axis=(0, 3, 4)
    )
    upper_std = np.maximum(upper_std, 1e-5)
    np.save("upper_mean.npy", upper_mean)
    np.save("upper_std.npy", upper_std)
    input_up = (
                       input_up
                       - upper_mean[None, :, :, None, None]
               ) / (
                       upper_std[None, :, :, None, None]
               )

    test_up = (
                      test_up
                      - upper_mean[None, :, :, None, None]
              ) / (
                      upper_std[None, :, :, None, None]
              )

    down_mean = input_down.mean(
        axis=(0, 2, 3)
    )

    down_std = input_down.std(
        axis=(0, 2, 3)
    )
    down_std = np.maximum(down_std, 1e-5)
    np.save("down_mean.npy", down_mean)
    np.save("down_std.npy", down_std)
    input_down = (
                         input_down
                         - down_mean[None, :, None, None]
                 ) / (
                         down_std[None, :, None, None]
                         + 1e-6
                 )

    test_down = (
                        test_down
                        - down_mean[None, :, None, None]
                ) / (
                        down_std[None, :, None, None]
                        + 1e-6
                )

    print(input_up.mean())
    print(input_up.std())
    print(input_down.mean())
    print(input_down.std())

    return input_up, input_down, test_up, test_down

def un_normalize(up_mean_file,up_std_file,down_mean_file,down_std_file,pred_up, pred_down,target_up,target_down):
    up_mean = np.load(up_mean_file)
    up_std = np.load(up_std_file)
    down_mean = np.load(down_mean_file)
    down_std = np.load(down_std_file)

    pred_up = (
            pred_up
                * up_std[None,:,:,None,None]
                + up_mean[None,:,:,None,None]
    )
    pred_down = (
        pred_down
        * down_std[None,:, None,None]
        + down_mean[None,:, None,None]
    )
    target_up = (
        target_up
        * up_std[None,:,:,None,None]
        + up_mean[None,:,:,None,None]
    )

    target_down = (
        target_down
        * down_std[None,:, None,None]
        + down_mean[None,:, None,None]
    )

    return pred_up, pred_down, target_up, target_down


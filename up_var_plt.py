import matplotlib.pyplot as plt
import numpy as np


def plot_upper_prediction(
    pred_up,
    target_up,
    sample_idx=0,
    level_idx=0,
    var_names=None,
    cmap="jet",
    figsize=(12, 18),
):


    if var_names is None:
        var_names = [
            "Z500",
            "Q700",
            "T850",
            "U850",
            "V850"
        ]

    n_var = len(var_names)

    fig, axes = plt.subplots(
        n_var,
        3,
        figsize=figsize
    )

    # 如果只有一个变量
    if n_var == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, var in enumerate(var_names):

        truth = (
            target_up[sample_idx, i, level_idx]
            .detach()
            .cpu()
            .numpy()
        )

        pred = (
            pred_up[sample_idx, i, level_idx]
            .detach()
            .cpu()
            .numpy()
        )

        error = pred - truth

        vmax = max(
            np.max(truth),
            np.max(pred)
        )

        vmin = min(
            np.min(truth),
            np.min(pred)
        )

        axes[i, 0].imshow(
            truth,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        axes[i, 0].set_title(f"{var} Truth")

        axes[i, 1].imshow(
            pred,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        axes[i, 1].set_title(f"{var} Prediction")

        axes[i, 2].imshow(
            error,
            cmap="RdBu_r"
        )
        axes[i, 2].set_title(f"{var} Error")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()
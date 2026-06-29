import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda")

def weighted_acc_simple(
    pred,
    target
):
    pred = pred.reshape(pred.shape[0],-1)
    target = target.reshape(target.shape[0],-1)

    pred = pred - pred.mean(dim=1,keepdim=True)
    target = target - target.mean(dim=1,keepdim=True)

    numerator = (pred*target).sum(dim=1)

    denominator = torch.sqrt(
        (pred**2).sum(dim=1)
        *
        (target**2).sum(dim=1)
    )

    acc = numerator/(denominator+1e-8)

    return acc.mean()

dicts = torch.load("best_model.pth")
model.load_state_dict(dicts)

def evaluate(
    model,
    test_up,
    test_down,
    upper_mean,
    upper_std,
    down_mean,
    down_std,
    max_lead=2,
    batch_size=48,
    device="cuda"
):

    model.eval()

    eval_vars = {
        "Z500": (0, 5),
        "Q700": (1, 3),
        "T850": (2, 2),
        "U850": (3, 2),
        "V850": (4, 2),
    }

    surface_vars = {
        "U10": 0,
        "V10": 1,
        "T2M": 2,
        "MSL": 3
    }

    metrics = {}

    for name in list(eval_vars.keys()) + list(surface_vars.keys()):
        metrics[name] = {
            "rmse": [],
            "mae": [],
            "acc": []
        }

    with torch.no_grad():

        for lead in range(1, max_lead + 1):

            print(f"Evaluating {lead * 6}h")

            dataset = loader(
                test_up,
                test_down,
                lead
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            )

            rmse_sum = {name: 0.0 for name in metrics}
            mae_sum = {name: 0.0 for name in metrics}
            acc_sum = {name: 0.0 for name in metrics}

            for in_up, in_down, target_up, target_down in loader:

                in_up = in_up.to(device)
                in_down = in_down.to(device)

                target_up = target_up.to(device)
                target_down = target_down.to(device)

                pred_up = in_up
                pred_down = in_down

                constant = in_down[:1, 4:].repeat(pred_down.shape[0], 1, 1, 1)

                # rollout
                for _ in range(lead):

                    pred_up, pred_down = model(
                        pred_up,
                        pred_down
                    )

                    pred_down = torch.cat(
                        [pred_down, constant],
                        dim=1
                    )


                pred_up = (
                    pred_up.cpu()
                    * upper_std[None, :, :, None, None]
                    + upper_mean[None, :, :, None, None]
                )

                target_up = (
                    target_up.cpu()
                    * upper_std[None, :, :, None, None]
                    + upper_mean[None, :, :, None, None]
                )

                pred_down = (
                    pred_down[:, :4].cpu()
                    * down_std[None, :4, None, None]
                    + down_mean[None, :4, None, None]
                )

                target_down = (
                    target_down[:, :4].cpu()
                    * down_std[None, :4, None, None]
                    + down_mean[None, :4, None, None]
                )

                for name, (var_idx, level_idx) in eval_vars.items():

                    pred = pred_up[:, var_idx, level_idx]
                    truth = target_up[:, var_idx, level_idx]

                    rmse = torch.sqrt(
                        torch.mean((pred - truth) ** 2)
                    )

                    mae = torch.mean(
                        torch.abs(pred - truth)
                    )

                    acc = weighted_acc_simple(
                        pred,
                        truth
                    )

                    rmse_sum[name] += rmse.item()
                    mae_sum[name] += mae.item()
                    acc_sum[name] += acc.item()


                for name, idx in surface_vars.items():

                    pred = pred_down[:, idx]
                    truth = target_down[:, idx]

                    rmse = torch.sqrt(
                        torch.mean((pred - truth) ** 2)
                    )

                    mae = torch.mean(
                        torch.abs(pred - truth)
                    )

                    acc = weighted_acc_simple(
                        pred,
                        truth
                    )

                    rmse_sum[name] += rmse.item()
                    mae_sum[name] += mae.item()
                    acc_sum[name] += acc.item()



            for name in metrics:

                metrics[name]["rmse"].append(
                    rmse_sum[name] / len(loader)
                )

                metrics[name]["mae"].append(
                    mae_sum[name] / len(loader)
                )

                metrics[name]["acc"].append(
                    acc_sum[name] / len(loader)
                )

    return metrics
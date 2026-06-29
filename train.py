import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda")

model = PanguModel()
optimier = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
loss_fun = torch.nn.MSELoss()
epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimier,T_max=epochs, eta_min=1e-6)
model = model.to(device)
train_loss = []
test_loss = []
best_score = 100.0

weights = {
    "Z500": 1,
    "Q700": 2,
    "T850": 1,
    "U850": 1,
    "V850": 1,
    "U10": 1,
    "V10": 1,
    "T2M": 1,
    "MSL": 1
}
for epoch in range(epochs):

    model.train()
    train_total_loss = 0.0
    test_total_loss = 0.0
    loss_up = 0.0
    loss_down = 0.0
    for in_up, in_down, target_up, target_down in dataloder:
        in_up = in_up.to(device)
        in_down = in_down.to(device)
        target_up = target_up.to(device)
        target_down = target_down.to(device)

        out_up, out_down = model(in_up, in_down)

        loss_up += loss_fun(out_up[:, 0], target_up[:, 0]) * weights["Z500"]
        loss_up += loss_fun(out_up[:, 1], target_up[:, 1]) * weights["Q700"]
        loss_up += loss_fun(out_up[:, 2], target_up[:, 2]) * weights["T850"]
        loss_up += loss_fun(out_up[:, 3], target_up[:, 3]) * weights["U850"]
        loss_up += loss_fun(out_up[:, 4], target_up[:, 4]) * weights["V850"]

        # loss_up /= (out_up.shape[2] * 5)

        loss_down += loss_fun(out_down[:, 0], target_down[:, 0]) * weights["U10"]
        loss_down += loss_fun(out_down[:, 1], target_down[:, 1]) * weights["V10"]
        loss_down += loss_fun(out_down[:, 2], target_down[:, 2]) * weights["T2M"]
        loss_down += loss_fun(out_down[:, 3], target_down[:, 3]) * weights["MSL"]
        # loss_down /= 4
        loss = loss_up * 0.6 + loss_down * 0.4

        optimier.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimier.step()
        train_total_loss += loss.item()
        loss_up = 0
        loss_down = 0
    train_loss.append(train_total_loss / len(dataloder))
    model.eval()
    with torch.no_grad():
        for in_up, in_down, target_up, target_down in test_dataloader:
            in_up = in_up.to(device)
            in_down = in_down.to(device)
            target_up = target_up.to(device)
            target_down = target_down.to(device)

            out_up, out_down = model(in_up, in_down)

            loss_up += loss_fun(out_up[:, 0], target_up[:, 0]) * weights["Z500"]
            loss_up += loss_fun(out_up[:, 1], target_up[:, 1]) * weights["Q700"]
            loss_up += loss_fun(out_up[:, 2], target_up[:, 2]) * weights["T850"]
            loss_up += loss_fun(out_up[:, 3], target_up[:, 3]) * weights["U850"]
            loss_up += loss_fun(out_up[:, 4], target_up[:, 4]) * weights["V850"]

            # loss_up /= (out_up.shape[2] * 5)

            loss_down += loss_fun(out_down[:, 0], target_down[:, 0]) * weights["U10"]
            loss_down += loss_fun(out_down[:, 1], target_down[:, 1]) * weights["V10"]
            loss_down += loss_fun(out_down[:, 2], target_down[:, 2]) * weights["T2M"]
            loss_down += loss_fun(out_down[:, 3], target_down[:, 3]) * weights["MSL"]
            # loss_down /= 4
            loss = loss_up * 0.6 + loss_down * 0.4
            test_total_loss += loss.item()
            loss_up = 0.0
            loss_down = 0.0
        test_loss.append(test_total_loss / len(test_dataloader))
        scheduler.step()
        if best_score > test_loss[-1]:
            torch.save(model.state_dict(), "best_model.pth")
            best_score = test_loss[-1]
        current_lr = optimier.param_groups[0]["lr"]
        print(f"epoch:{epoch + 1}的train_loss,test_loss：{train_loss[epoch], test_loss[epoch]}，lr：{current_lr}")
plt.close('all')
plt.plot(train_loss, color='blue', label='Train Loss')
plt.plot(test_loss, color='red', label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(1, epochs)
plt.title("Training And Testing Loss Curve")
plt.show()


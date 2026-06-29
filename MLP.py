import torch
# 简单的双层MLP用来非线性操作
class MLP(torch.nn.Module):
    def __init__(self,dim,drop_rate):
        super().__init__()
        self.l1 = torch.nn.Linear(dim,4*dim)
        self.l2 = torch.nn.Linear(4*dim,dim)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self,x):
        x = self.l1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        return x

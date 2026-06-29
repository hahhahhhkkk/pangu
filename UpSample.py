import torch
# 上采样 先映射特征的4倍 然后将特征分别加入到hw
class UpSample(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.liner1 = torch.nn.Linear(in_channel,4*out_channel)
        self.liner2 = torch.nn.Linear(out_channel,out_channel)
        self.norm1 = torch.nn.LayerNorm(out_channel)
    def forward(self,x,z,h,w):
        x = self.liner1(x)
        x = x.reshape(x.shape[0],z,h,w,2,2,x.shape[-1]//4)
        x = x.permute(0,1,2,4,3,5,6)
        x = x.reshape(x.shape[0],z,h*2,w*2,-1)
        x = self.norm1(x)
        x = self.liner2(x)
        x = x.reshape(x.shape[0],-1,x.shape[-1])
        return x

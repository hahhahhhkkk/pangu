import torch
class EarthSpecificLayer(torch.nn.Module):
    def __init__(self,depth,dim,dropout_rate,head,window_size,input_shape):
        super().__init__()
        self.depth = depth
        self.block = []

        for i in range(depth):
            self.block.append(EarthSpecificBlock(dim,dropout_rate[i],head,window_size,input_shape))
        self.block = nn.ModuleList(self.block)
    def forward(self,x,z,h,w):
        origin_shape = x.shape
        for i in range(self.depth):
            if i%2==0:
                x = self.block[i](x,z,h,w,True)
            else:
                x = self.block[i](x,z,h,w,False)
            x = x.reshape(origin_shape)
        return x
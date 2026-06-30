class PanguModel(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.drop_list  = np.linspace(0, 0.2, 8)
      self.window_size = (2,2,2)
      self.patchembedding = PatchEmbedding(192，self.window_size)

      self.layer1 = EarthSpecificLayer(2,192,self.drop_list[:2],4,self.window_size,(8,24,36))
      self.layer2 = EarthSpecificLayer(2,384,self.drop_list[2:],6,self.window_size,(8,12,18))
      self.layer3 = EarthSpecificLayer(2 ,384,self.drop_list[2:],6,self.window_size,(8,12,18))
      self.layer4 = EarthSpecificLayer(2,192,self.drop_list[:2],4,self.window_size,(8,24,36))

      self.downsample = DownSample(192)
      self.upsample = UpSample(384,192)

      self.patchrecover = PatchRecover()

   def forward(self,input_up,input_down):
      # print(f"原始形状up and down:{input_up.shape,input_down.shape}")
      x = self.patchembedding(input_up,input_down)
      # print(f"ptachembedding后形状:{x.shape}")
      x = self.layer1(x,8,24,36)
      # print(f"layer1后形状:{x.shape}")
      skip_x = x

      x = self.downsample(x,8,24,36)
      # print(f"downsample后形状:{x.shape}")
      x = self.layer2(x,8,12,18)
      # print(f"layer2后形状:{x.shape}")
      x = self.layer3(x,8,12,18)
      # print(f"layer3后形状:{x.shape}")
      x = self.upsample(x,8,12,18)
      # print(f"upsample后形状:{x.shape}")
      x = self.layer4(x,8,24,36)
      # print(f"layer4后形状:{x.shape}")
      x = torch.cat([skip_x,x],dim=-1)
      # print(f"torch.cat后形状:{x.shape}")
      input_up,input_down = self.patchrecover(x,8,24,36)
      # print(f"patchrecover后形状:{input_up.shape,input_down.shape}")
      return input_up,input_down

import torch
import torch.nn as nn
from cs336_basics.Transformer_utils import Linear_Transform,SiLU_Activation

class Feed_Forward_Network(nn.Module):
    def __init__(self,
                 d_model:int,
                 d_ff=None,
                 device=None,
                 dtype=None):
        super(Feed_Forward_Network,self).__init__()
        self.d_model=d_model
        if d_ff is not None:
            self.d_ff=d_ff
        else:
            self.d_ff=int(8/3*d_model)
        self.linear_w1=Linear_Transform(d_model,d_ff,device=device,dtype=dtype)
        self.linear_w3=Linear_Transform(d_model,d_ff,device=device,dtype=dtype)
        self.linear_w2=Linear_Transform(d_ff,d_model,device=device,dtype=dtype)
        self.activator=SiLU_Activation()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        enhanced=self.linear_w1(x)
        activated=self.activator(enhanced)
        gate=self.linear_w3(x)
        gated=activated*gate
        output=self.linear_w2(gated)
        return output
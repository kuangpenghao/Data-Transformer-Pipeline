import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super(RoPE,self).__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device

        d_half=d_k//2
        positions=torch.arange(max_seq_len, device=device).unsqueeze(1)#[max_seq_len,1]
        dims=torch.arange(d_half,device=device).unsqueeze(0)#[1,d_half]
        angles=positions/(theta**(2*dims/d_k))#[max_seq_len,d_half]

        cos_values=torch.cos(angles).unsqueeze(0)
        self.register_buffer("cos_values",cos_values)#(1,max_seq_len,d_half)
        sin_values=torch.sin(angles).unsqueeze(0)
        self.register_buffer("sin_values",sin_values)#(1,max_seq_len,d_half)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        x_splited=x.reshape(*x.shape[:-1],self.d_k//2,2)
        cos_chunk=self.cos_values[:,token_positions,:]
        sin_chunk=self.sin_values[:,token_positions,:]
        
        even_transform=torch.stack([cos_chunk,-sin_chunk],dim=-1)
        odd_transform=torch.stack([sin_chunk,cos_chunk],dim=-1)

        x_rotated_odd=torch.sum(x_splited*even_transform,dim=-1)#(bsz,seq_len,d_k//2)
        x_rotated_even=torch.sum(x_splited*odd_transform,dim=-1)#(bsz,seq_len,d_k//2)
        stacked_x=torch.stack([x_rotated_odd,x_rotated_even],dim=-1)
        x_rotated=stacked_x.reshape(*stacked_x.shape[:-2],self.d_k)
        
        return x_rotated
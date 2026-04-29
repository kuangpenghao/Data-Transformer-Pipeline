import torch
import torch.nn as nn
from cs336_basics.Transformer_utils import *
from cs336_basics.Attention import Multihead_Attention
from cs336_basics.Feed_Forward import Feed_Forward_Network

class Transformer_Block(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 max_seq_length:int=None,
                 theta:int=None,
                 dtype=None,
                 device=None):
        super(Transformer_Block,self).__init__()
        self.RMSNorm_Attn=RMSNorm(d_model,dtype=dtype,device=device)
        self.RMSNorm_FF=RMSNorm(d_model,dtype=dtype,device=device)
        self.Multihead_Attn=Multihead_Attention(d_model,num_heads,max_seq_length,theta,device=device)
        self.Feed_Forward=Feed_Forward_Network(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        residual_attn=x
        x_normed_attn=self.RMSNorm_Attn(x)
        attn_output=self.Multihead_Attn(x_normed_attn,token_positions)
        x=residual_attn+attn_output
        
        residual_ff=x
        x_normed_ff=self.RMSNorm_FF(x)
        ff_output=self.Feed_Forward(x_normed_ff)
        x=residual_ff+ff_output

        return x

class Transformer_LM(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 vocab_size:int,
                 num_layers:int,
                 max_seq_length:int=None,
                 theta:int=None,
                 dtype=None,
                 device=None):
        super(Transformer_LM,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.max_seq_length=max_seq_length
        self.theta=theta
        self.dtype=dtype
        self.device=device
        self.embeddings=Generate_Embeddings(vocab_size,d_model,device=device,dtype=dtype)
        self.transformer_blocks=nn.ModuleList([
            Transformer_Block(d_model=d_model,
                              num_heads=num_heads,
                              d_ff=d_ff,
                              max_seq_length=max_seq_length,
                              theta=theta,
                              dtype=dtype,
                              device=device)
            for _ in range(num_layers)
        ])
        self.final_norm=RMSNorm(d_model,device=device,dtype=dtype)
        self.final_layer=Linear_Transform(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,token_ids:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        x=self.embeddings(token_ids)
        for block in self.transformer_blocks:
            x=block(x,token_positions)
        x=self.final_norm(x)
        linear_score=self.final_layer(x)
        return linear_score
    
if __name__=="__main__":
    lm=Transformer_LM(d_model=512,
                      num_heads=8,
                      d_ff=2048,
                      vocab_size=10000,
                      num_layers=2,
                      max_seq_length=128,
                      theta=100000,
                      dtype=torch.float32,
                      device="cpu")
    states=lm.state_dict()
    for state_key in states:
        print(state_key,states[state_key].shape)
import torch
import torch.nn as nn
import math
from cs336_basics.Transformer_utils import *
from cs336_basics.RoPE import RoPE

class Scaled_dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_dot_Product_Attention,self).__init__()

    def forward(self,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        d_k=Q.shape[-1]

        attn_score=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)#[bsz,*,Qseq_len,Kseq_len]
        if mask is not None:
            attn_score=attn_score.masked_fill(mask==0,float('-inf'))
        softmax=Softmax_Activation(dim=-1)
        attn_weight=softmax(attn_score)#[bsz,*,Qseq_len,Kseq_len]

        #V:[bsz,*,Kseq_len,d_v]
        attn_output=torch.matmul(attn_weight,V)
        return attn_output#[bsz,*,Qseq_len,d_v]

class Causal_Mask:
    def __init__(self,seq_len,device=None):
        self.seq_len=seq_len
        self.device=device

    def generate(self)->torch.Tensor:
        ones=torch.ones(self.seq_len,self.seq_len,device=self.device)
        mask=torch.triu(ones,diagonal=1)
        mask=(mask==0)
        return mask

class Multihead_Attention(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 max_seq_length:int=None,
                 theta:int=None,
                 device=None):
        super(Multihead_Attention,self).__init__()
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.d_v=d_model//num_heads

        self.max_seq_length=max_seq_length
        self.theta=theta
        self.token_positions=None

        self.q_proj=Linear_Transform(d_model,num_heads*self.d_k,device=device)
        self.k_proj=Linear_Transform(d_model,num_heads*self.d_k,device=device)
        self.v_proj=Linear_Transform(d_model,num_heads*self.d_v,device=device)
        self.o_proj=Linear_Transform(num_heads*self.d_v,d_model,device=device)

        self.sdpa=Scaled_dot_Product_Attention()

        if max_seq_length is not None and theta is not None:
            self.rope=RoPE(theta,self.d_k,max_seq_length,device=device)
        else:
            self.rope=None

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        bsz=x.shape[0]
        seq_len=x.shape[1]

        #qk:[bsz,n_heads,seq_len,d_k]
        #v:[bsz,n_heads,seq_len,d_v]
        Q=self.q_proj(x)
        Q=Q.reshape(bsz,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K=self.k_proj(x)
        K=K.reshape(bsz,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V=self.v_proj(x)
        V=V.reshape(bsz,seq_len,self.num_heads,self.d_v).transpose(1,2)

        #apply RoPE on QK
        if self.rope is not None:
            self.token_positions=token_positions
            Q=self.rope(Q,self.token_positions)
            K=self.rope(K,self.token_positions)

        mask=Causal_Mask(seq_len,device=x.device).generate()
        mask=mask.unsqueeze(0).unsqueeze(1)

        attn_output=self.sdpa(Q,K,V,mask)
        attn_output=attn_output.transpose(1,2).reshape(bsz,seq_len,self.num_heads*self.d_v)
        attn_output=self.o_proj(attn_output)

        return attn_output
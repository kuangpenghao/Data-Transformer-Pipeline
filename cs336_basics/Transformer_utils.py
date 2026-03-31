import torch
import torch.nn as nn

# 重点：行主序规则
class Linear_Transform(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super(Linear_Transform,self).__init__()
        self.linear_matrix=torch.empty(out_features,
                                       in_features,
                                       device=device,
                                       dtype=torch.float32)
        self.linear_matrix=self.linear_matrix.transpose(-2,-1)
        nn.init.trunc_normal_(self.linear_matrix,mean=0,std=0.02)
        self.linear_matrix=nn.Parameter(self.linear_matrix)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return torch.matmul(x,self.linear_matrix)

# vocab_size*embedding_dim矩阵中取样
class Generate_Embeddings(nn.Module):
    def __init__(self,number_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super(Generate_Embeddings,self).__init__()
        self.embedding_matrix=torch.empty(number_embeddings,
                                          embedding_dim,
                                          device=device,
                                          dtype=torch.float32)
        nn.init.trunc_normal_(self.embedding_matrix,mean=0,std=0.02)
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.embedding_matrix[token_ids]

# 运算法则：归一化*可学习缩放倍数
class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super(RMSNorm,self).__init__()
        self.eps=eps
        self.g=nn.Parameter(torch.ones(d_model,device=device,dtype=torch.float32))

    def _get_rms(self,x:torch.Tensor)->torch.Tensor:
        sum_square=torch.sum(x**2,dim=-1,keepdim=True)
        mean_square=sum_square/x.shape[-1]
        return torch.sqrt(mean_square+self.eps)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        ori_dtype=x.dtype
        x=x.to(torch.float32) # 张量元素类型转换
        rms=self._get_rms(x)
        x_normed=x/rms
        x_normed=x_normed.to(ori_dtype)
        return x_normed*self.g

class Sigmoid_Activation(nn.Module):
    def __init__(self):
        super(Sigmoid_Activation,self).__init__()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        denominator=1+torch.exp(-x)
        return 1/denominator

class SiLU_Activation(nn.Module):
    def __init__(self):
        super(SiLU_Activation,self).__init__()
        self.sigmoid_activator=Sigmoid_Activation()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        sigmoid_x=self.sigmoid_activator(x)
        return x*sigmoid_x

# 需要公式推导
class Softmax_Activation(nn.Module):
    def __init__(self,dim:int=-1):
        super(Softmax_Activation,self).__init__()
        self.dim=dim

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #shape of x:(bsz,seq_len,d_k)
        x_max=torch.max(x,dim=self.dim,keepdim=True).values
        x_exp=torch.exp(x-x_max)
        x_exp_sum=torch.sum(x_exp,dim=self.dim,keepdim=True)
        return x_exp/x_exp_sum

# 需要公式推导
class Log_Softmax():
    def __init__(self,dim:int=-1):
        self.dim=dim

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x_max=torch.max(x,dim=self.dim,keepdim=True).values
        x=x-x_max
        x_exp=torch.exp(x)
        x_exp_sum=torch.sum(x_exp,dim=self.dim,keepdim=True)
        return x-torch.log(x_exp_sum)
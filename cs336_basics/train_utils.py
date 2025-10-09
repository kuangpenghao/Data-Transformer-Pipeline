from cs336_basics.Transformer_utils import Log_Softmax
from cs336_basics.text_chunker import Memmap_Manager
import torch
import math
import numpy as np

class Cross_Entropy_Calculator:
    def __init__(self):
        self.log_softmax=Log_Softmax(dim=-1)

    def forward(self,inputs:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
        inputs=inputs.reshape(-1,inputs.shape[-1])#[bsz*seq_len,vocab_size]
        inputs=-self.log_softmax.forward(inputs)#[bsz*seq_len,vocab_size]
        targets=targets.reshape(-1)#[bsz*seq_len]
 
        selected=inputs[torch.arange(inputs.shape[0]),targets]#[bsz*seq_len]
        loss=torch.mean(selected,dim=0)
        return loss

class AdamW_Optimizer(torch.optim.Optimizer):
    def __init__(self,parameters,lr:float,weight_decay:float,betas,eps:float):
        param_groups=[
            {
                "params":parameters,
                "lr":lr
            }
        ]
        super(AdamW_Optimizer,self).__init__(param_groups,{})
        self.weight_decay=weight_decay
        self.beta1=betas[0]
        self.beta2=betas[1]
        self.eps=eps
        
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]={
                    "m":torch.zeros_like(p.data),
                    "v":torch.zeros_like(p.data),
                    "step": torch.tensor(0.0, device=p.device)
                }

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad=p.grad.data
                state=self.state[p]
                m,v,step=state["m"],state["v"],state["step"]

                if not isinstance(step,torch.Tensor):
                    step=torch.tensor(float(step),device=p.device)
                    state["step"]=step

                current_lr=group.get("lr")

                m=m*self.beta1+(1-self.beta1)*grad
                v=v*self.beta2+(1-self.beta2)*(grad**2)
                step=step+1

                alpha_t=current_lr*(math.sqrt(1-self.beta2**step)/(1-self.beta1**step))
                p.data=p.data-alpha_t*(m/(torch.sqrt(v)+self.eps))-current_lr*self.weight_decay*p.data

                self.state[p]["m"]=m
                self.state[p]["v"]=v
                self.state[p]["step"]=step

class Learning_Rate_Scheduler:
    def __init__(self):
        pass
    def get_lr(self,step,lr_max,lr_min,Tw,Tc)->float:
        if step<Tw:
            lr=lr_max*step/Tw
        elif step>Tc:
            lr=lr_min
        else:
            lr=lr_min+0.5*(1+math.cos(math.pi*(step-Tw)/(Tc-Tw) ) )*(lr_max-lr_min)
        return lr

class Gradient_Clipper:
    def __init__(self,max_norm:float):
        self.max_norm=max_norm
    def clip(self,parameters):
        total_norm=torch.sqrt(sum(p.grad.data.norm(2)**2 for p in parameters if p.grad is not None))
        for p in parameters:
            if p.grad is None:
                continue
            if total_norm<=self.max_norm:
                continue
            clip_factor=self.max_norm/(total_norm+1e-6)
            p.grad.data=p.grad.data*clip_factor

class Batch_Getter:
    def __init__(self):
        pass

    def get_batch(self,dataset,bsz,seq_len,device=None):
        max_start_idx=len(dataset)-seq_len
        start_indices=np.random.randint(0,max_start_idx,bsz)

        x=np.array([dataset[i:i+seq_len] for i in start_indices],dtype=np.int64)
        y=np.array([dataset[i+1:i+seq_len+1] for i in start_indices],dtype=np.int64)

        x=torch.tensor(x,dtype=torch.long,device=device)
        y=torch.tensor(y,dtype=torch.long,device=device)
        return (x,y)

class Batch_By_Memmap:
    def __init__(self,memmep_manager:Memmap_Manager):
        self.memmap_manager=memmep_manager

    def get_batch(self,bsz,seq_len,dataset_length,device=None):
        max_start_idx=dataset_length-seq_len
        start_indices=np.random.randint(0,max_start_idx,bsz)

        x=np.array([self.memmap_manager.load_by_range(i,i+seq_len) for i in start_indices],dtype=np.int64)
        y=np.array([self.memmap_manager.load_by_range(i+1,i+seq_len+1) for i in start_indices],dtype=np.int64)

        x=torch.tensor(x,dtype=torch.long,device=device)
        y=torch.tensor(y,dtype=torch.long,device=device)
        return (x,y)

class Checkpoint_Manager:
    def __init__(self):
        pass

    def save(self,model,optimizer,iteration,save_path):
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_model=model.state_dict()
        state_optimizer=optimizer.state_dict()
        checkpoint={
            "model":state_model,
            "optimizer":state_optimizer,
            "iteration":iteration
        }
        torch.save(checkpoint,save_path)

    def load(self,src_path,model,optimizer=None):
        checkpoint=torch.load(src_path)
        state_model=checkpoint["model"]
        if optimizer is not None:
            print(f"optimizer is not none")
            state_optimizer=checkpoint["optimizer"]
        iteration=checkpoint["iteration"]

        model.load_state_dict(state_model)
        if optimizer is not None:
            optimizer.load_state_dict(state_optimizer)
        return iteration
    
if __name__=="__main__":
    model=torch.nn.Linear(10,10)
    optimizer=AdamW_Optimizer(model.parameters(),lr=0.001,weight_decay=0.01,betas=(0.9,0.95),eps=1e-8)
    states=optimizer.state_dict()
    print(states)
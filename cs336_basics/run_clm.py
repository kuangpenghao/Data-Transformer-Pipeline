from cs336_basics.train_utils import *
from cs336_basics.text_chunker import Memmap_Manager
from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
from cs336_basics.Transformer_LM import Transformer_LM
from cs336_basics.Transformer_utils import Softmax_Activation
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import argparse
import time


def parse_bash_args():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--d_model",type=int,default=512)
    parser.add_argument("--num_heads",type=int,default=8)
    parser.add_argument("--d_ff",type=int,default=1344)
    parser.add_argument("--vocab_size",type=int,default=32000)
    parser.add_argument("--num_layers",type=int,default=6)
    parser.add_argument("--max_seq_length",type=int,default=512)
    parser.add_argument("--seq_length",type=int,default=256)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--theta",type=int,default=100000)
    parser.add_argument("--device",type=str,default="cuda")

    parser.add_argument("--num_epochs",type=float,default=10)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--lr_min",type=float,default=1e-5)
    parser.add_argument("--warmup_ratio",type=float,default=0.1)
    parser.add_argument("--warmfix_ratio",type=float,default=0.9)
    
    parser.add_argument("--chunk_size",type=int,default=500000)
    parser.add_argument("--vocab_path",type=str,default="data/vocab_32000.txt")
    parser.add_argument("--merges_path",type=str,default="data/merges_32000.txt")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])

    parser.add_argument("--corpus_size",type=str)

    parser.add_argument("--log_interval",type=int)    
    parser.add_argument("--save_interval",type=int)

    parser.add_argument("--weight_decay",type=float,default=0.01)
    parser.add_argument("--betas",type=float, nargs="*", default=(0.9,0.95))
    parser.add_argument("--eps",type=float,default=1e-8)

    parser.add_argument("--max_norm",type=float,default=1.0)
    
    args=parser.parse_args()
    return args


def training_loop(batch_getter,batch_size,seq_length,dataset_length,device,
                  transformer_lm,token_positions,
                  loss_fn,optimizer,grad_clipper,
                  should_output):
    
    tensor_input=batch_getter.get_batch(batch_size,seq_length,dataset_length,device=device)
    token_ids,labels=tensor_input

    if should_output==True:
        tokenizer=BPE_Tokenizer.from_files("/home/kuangph/CS336-Assignment1/data/vocab_32000.txt",
                                           "/home/kuangph/CS336-Assignment1/data/merges_32000.txt",
                                           special_tokens=["<|endoftext|>"])
        #just choose first batch of token ids to output
        sample_ids=token_ids[0].cpu().numpy().tolist()
        sample_text=tokenizer.decode(sample_ids)
        print(f"sample text:\n{sample_text}\n")


    output=transformer_lm(token_ids,token_positions)

    if should_output==True:
        # output shape (batch_size,seq_length,vocab_size)
        sample_output=output[0].detach().cpu().numpy()
        sample_pred_ids=np.argmax(sample_output,axis=-1).tolist()
        sample_pred_text=tokenizer.decode(sample_pred_ids)
        print(f"sample prediction:\n{sample_pred_text}\n")
        #input("press enter to continue")

    loss=loss_fn.forward(output,labels)
    optimizer.zero_grad()
    loss.backward()
    grad_clipper.clip(transformer_lm.parameters())
    optimizer.step()

    return loss.item()

# decode function implementation
def decode_function(transformer_lm, tokenizer, config, device):

    corpus_path="/home/kuangph/CS336-Assignment1/data/validation/decode_1.txt"
    
    with open(corpus_path) as f:
        encoded_list=tokenizer.encode(f.read())
    
    decoded_list=tokenizer.decode(encoded_list)
    print("=================================================\n")
    print(f"Text to decode:\n{decoded_list}")
    print("\n=================================================\n")
    print(f"Generated text:")

    input("press Enter to decode...")

    ite=0
    max_gen_length=256
    
    softmax_activation=Softmax_Activation(-1)
    temperature=0.8
    repetition_penalty=1.2
    recent_tokens=[]

    while ite < max_gen_length:
        if len(encoded_list) < config["max_seq_length"]:
            input_ids=encoded_list
        else:
            input_ids=encoded_list[-config["max_seq_length"]:]
        
        input_tensor=torch.tensor([input_ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            token_positions=torch.arange(len(input_ids), device=device)
        
            output_scores=transformer_lm(input_tensor, token_positions)
            last_token_scores=output_scores[0,-1,:]
            
            for token_id in recent_tokens[-10:]:
                last_token_scores[token_id] /= repetition_penalty
            
            last_token_scores=last_token_scores / temperature
            last_token_weights=softmax_activation(last_token_scores)
            sampled_id=torch.multinomial(last_token_weights, num_samples=1).item()
            encoded_list.append(sampled_id)
            sampled_token=tokenizer.decode([sampled_id])
            print(sampled_token, end="")
            
            recent_tokens.append(sampled_id)
            if len(recent_tokens)>15:
                recent_tokens.pop(0)
            
            ite+=1

            if sampled_id==tokenizer.encode("<|endoftext|>")[0]:
                break
    print("\n")


def train_manage():

    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    args=parse_bash_args()

    d_model=args.d_model
    num_heads=args.num_heads
    d_ff=args.d_ff
    vocab_size=args.vocab_size
    num_layers=args.num_layers
    max_seq_length=args.max_seq_length
    seq_length=args.seq_length
    batch_size=args.batch_size
    theta=args.theta
    dtype=torch.float32
    device=args.device

    num_epochs=args.num_epochs
    lr_max=args.lr
    lr_min=args.lr_min
    warmup_ratio=args.warmup_ratio
    warmfix_ratio=args.warmfix_ratio

    chunk_size=args.chunk_size
    vocab_path=args.vocab_path
    merge_path=args.merges_path
    special_tokens=args.special_tokens
    corpus_size=args.corpus_size
    corpus_path="/home/kuangph/CS336-Assignment1/data/"+corpus_size+".txt"

    if corpus_path=="/home/kuangph/CS336-Assignment1/data/2K.txt":
        dataset_length=306
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/5M.txt":
        dataset_length=1310528
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/21M.txt":
        dataset_length=5621617
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/276M.txt":
        dataset_length=66296750
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/2G.txt":
        dataset_length=556539005
    if corpus_path=="/home/kuangph/CS336-Assignment1/data/11G.txt":
        dataset_length=2731070566

    save_path="/home/kuangph/CS336-Assignment1/outputs/"+corpus_size+"_checkpoints.pt"
    log_interval=args.log_interval
    save_interval=args.save_interval
    
    weight_decay=args.weight_decay
    betas=tuple(args.betas)
    eps=args.eps
    max_norm=args.max_norm

    token_positions=torch.arange(seq_length,device=device)

    transformer_lm=Transformer_LM(d_model=d_model,
                                  num_heads=num_heads,
                                  d_ff=d_ff,
                                  vocab_size=vocab_size,
                                  num_layers=num_layers,
                                  max_seq_length=max_seq_length,
                                  theta=theta,
                                  dtype=dtype,
                                  device=device)
                                  #token_positions=token_positions)

    loss_fn=Cross_Entropy_Calculator()
    optimizer=AdamW_Optimizer(transformer_lm.parameters(),lr=0,weight_decay=weight_decay,betas=betas,eps=eps)
    lr_scheduler=Learning_Rate_Scheduler()
    grad_clipper=Gradient_Clipper(max_norm=max_norm)
    memmap_manager=Memmap_Manager(chunk_size,vocab_path,merge_path,special_tokens,corpus_path,corpus_size)
    batch_getter=Batch_By_Memmap(memmap_manager)
    checkpoint_manager=Checkpoint_Manager()

    token_per_tensor=batch_size*seq_length
    total_iterations=dataset_length//token_per_tensor*int(num_epochs)
    warmup_iterations=int(total_iterations*warmup_ratio)
    warmfix_iterations=int(total_iterations*warmfix_ratio)

    print(f"total_iterations:{total_iterations},warmup_iterations:{warmup_iterations},warmfix_iterations:{warmfix_iterations}")

    ite=0

    # check if save_path exists
    try:
        ite=checkpoint_manager.load(save_path,transformer_lm,optimizer)
        print(f"checkpoint detected, training resumes from iteration {ite}")
    except:
        print("no checkpoint detected, training starts from iteration 0")

    last_time=time.time()
    last_ite=ite
    
    while ite<total_iterations:

        lr=lr_scheduler.get_lr(ite,lr_max,lr_min,warmup_iterations,warmfix_iterations)
        
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr

        if ite>=int(total_iterations*0.025) and (ite+1)%250==0:
            should_output=True
        else:
            should_output=False
        
        loss=training_loop(batch_getter,batch_size,seq_length,dataset_length,device,
                      transformer_lm,token_positions,
                      loss_fn,optimizer,grad_clipper,
                      should_output)
        
        if (ite+1)%25==0:
            current_time=time.time()
            time_spent=current_time-last_time
            ites_spent=ite-last_ite
            time_remaining=time_spent/ites_spent*(total_iterations-ite)
            hours=int(time_remaining)//3600
            minutes=(int(time_remaining)%3600)//60
            seconds=int(time_remaining)%60
            print(f"iteration/total:{ite+1}/{total_iterations}, loss:{loss}, lr:{lr}. Time remaining:{hours}h{minutes}m{seconds}s")

        if (ite+1)%save_interval==0:
            try:
                checkpoint_manager.save(transformer_lm,optimizer,ite+1,save_path)
                print(f"model saved at iteration {ite+1} to {save_path}")
            except:
                print("model saving failed")

        ite+=1

    # final save
    try:
        checkpoint_manager.save(transformer_lm,optimizer,ite,save_path)
        print(f"The model is trained for {num_epochs} epochs, final model saved to {save_path}")
    except:
        print("model saving failed")

    config_dict={
        "max_seq_length": max_seq_length,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "theta": theta
    }

    tokenizer=BPE_Tokenizer.from_files(vocab_path,merge_path,special_tokens)
    decode_function(transformer_lm,tokenizer,config_dict,device)

if __name__=="__main__":
    train_manage()
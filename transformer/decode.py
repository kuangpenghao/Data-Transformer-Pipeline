from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
from cs336_basics.Transformer_LM import Transformer_LM
from cs336_basics.train_utils import *
from cs336_basics.Transformer_utils import *

class Config:
    def __init__(self):
        self.corpus_path="/home/kuangph/CS336-Assignment1/data/validation/decode_1.txt"
        self.checkpoint_path="/home/kuangph/CS336-Assignment1/outputs/21M_checkpoints.pt"

        self.vocab_path="/home/kuangph/CS336-Assignment1/data/vocab_32000.txt"
        self.merge_path="/home/kuangph/CS336-Assignment1/data/merges_32000.txt"
        self.special_tokens=["<|endoftext|>"]

        self.d_model=512
        self.num_heads=8
        self.d_ff=1344
        self.vocab_size=32000
        self.num_layers=8
        self.max_seq_length=256
        self.seq_length=256
        self.theta=100000
        self.device="cuda"
        self.dtype=torch.float32

        self.max_generate_length=256

        self.temperature=0.8
        self.repetition_penalty=1.2
        self.recent_token_num=10

if __name__=="__main__":
    config=Config()

    tokenizer=BPE_Tokenizer.from_files(config.vocab_path, config.merge_path, config.special_tokens)

    transformer_lm=Transformer_LM(d_model=config.d_model,
                                  num_heads=config.num_heads,
                                  d_ff=config.d_ff,
                                  vocab_size=config.vocab_size,
                                  num_layers=config.num_layers,
                                  max_seq_length=config.max_seq_length,
                                  theta=config.theta,
                                  dtype=config.dtype,
                                  device=config.device)
    try:
        checkpoint_manager=Checkpoint_Manager()
        checkpoint_manager.load(config.checkpoint_path, transformer_lm)
        print(f"Successfully load model parameters. Path:{config.checkpoint_path}")
    except Exception as e:
        print(f"Failed to load model parameters. Path:{config.checkpoint_path}")
        print(f"Exception:{e}")
        exit(-1)

    corpus_path=config.corpus_path
    with open(corpus_path) as f:
        text=f.read()
        encoded_list=tokenizer.encode(text)

    decoded_list=tokenizer.decode(encoded_list)
    print(f"Text to decode:\n{decoded_list}\n\ngenerated text:")

    input("press Enter to decode...")

    ite=0
    max_gen_length=config.max_generate_length

    softmax_activator=Softmax_Activation(-1)
    temperature=config.temperature
    repetition_penalty=config.repetition_penalty
    recent_token_num=config.recent_token_num
    recent_tokens=[]

    while ite<max_gen_length:
        if len(encoded_list)<config.max_seq_length:
            input_ids=encoded_list
        else:
            input_ids=encoded_list[-config.max_seq_length:]
        
        input_tensor=torch.tensor([input_ids],dtype=torch.long,device=config.device)

        with torch.no_grad():
            token_positions=torch.arange(len(input_ids),device=config.device)

            output_scores=transformer_lm(input_tensor,token_positions)
            last_token_scores=output_scores[0,-1,:]

            for token_id in recent_tokens[-recent_token_num:]:
                last_token_scores[token_id]/=repetition_penalty

            last_token_scores=last_token_scores/temperature
            last_token_weights=softmax_activator(last_token_scores)
            sampled_id=torch.multinomial(last_token_weights,num_samples=1).item()
            encoded_list.append(sampled_id)
            
            sampled_token=tokenizer.decode([sampled_id])
            print(sampled_token,end="")

            recent_tokens.append(sampled_id)
            if len(recent_tokens) > recent_token_num:
                recent_tokens.pop(0)

            ite+=1

            if sampled_id==tokenizer.encode("<|endoftext|>")[0]:
                break
    print("\n")
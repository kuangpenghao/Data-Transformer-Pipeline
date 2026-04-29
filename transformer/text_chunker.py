from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
import numpy as np

class Memmap_Manager:
    def __init__(self,
                 chunk_size,
                 vocab_path,
                 merge_path,
                 special_tokens,
                 corpus_path,
                 corpus_size=None):
        self.chunk_size=chunk_size
        self.vocab_path=vocab_path
        self.merge_path=merge_path
        self.special_tokens=special_tokens
        self.corpus_path=corpus_path
        self.corpus_size=corpus_size
        
    def save_by_chunks(self,token_ids,buffer_len,chunk_num):
        fname="/home/kuangph/CS336-Assignment1/data/"+self.corpus_size+f"_chunks/encoded_tokens_chunk_{chunk_num}.dat"
        dtype=np.int32
        shape=(buffer_len,)
        memmap_arr = np.memmap(fname, dtype=dtype, mode="w+", shape=shape)
        memmap_arr[:] = token_ids[:]
        memmap_arr.flush()

    def save_as_memmap(self):
        tokenizer=BPE_Tokenizer.from_files(self.vocab_path,self.merge_path,self.special_tokens)
        buffer=[]
        chunk_num=0
        length=0
        with open(self.corpus_path) as f:
            encoder=tokenizer.encode_iterable(f)
            for id in encoder:
                length+=1
                buffer.append(id)
                if len(buffer)>=self.chunk_size:
                    self.save_by_chunks(buffer,self.chunk_size,chunk_num)
                    chunk_num+=1
                    buffer=[]
            if len(buffer)>0:
                self.save_by_chunks(buffer,len(buffer),chunk_num)
                buffer=[]
        print(f"length of corpus in tokens:{length}")

    def load_by_range(self,start_idx,end_idx):
        chunk_size=self.chunk_size
        start_chunk=start_idx//chunk_size
        end_chunk=end_idx//chunk_size
        idx_in_start=start_idx%chunk_size
        idx_in_end=end_idx%chunk_size

        token_ids=[]
        for chunk in range(start_chunk,end_chunk+1):
            fname=f"/home/kuangph/CS336-Assignment1/data/"+self.corpus_size+f"_chunks/encoded_tokens_chunk_{chunk}.dat"
            dtype=np.int32
            memmap_arr=np.memmap(fname,dtype=dtype,mode="r")
            if start_chunk==end_chunk:
                token_ids.extend(memmap_arr[idx_in_start:idx_in_end])
            else:
                if chunk==start_chunk:
                    token_ids.extend(memmap_arr[idx_in_start:])
                elif chunk>start_chunk and chunk<end_chunk:
                    token_ids.extend(memmap_arr[:])
                else:
                    token_ids.extend(memmap_arr[:idx_in_end])
        return token_ids

if __name__=="__main__":
    chunk_size=500000
    vocab_path="/home/kuangph/CS336-Assignment1/data/vocab_32000.txt"
    merge_path="/home/kuangph/CS336-Assignment1/data/merges_32000.txt"
    special_tokens=["<|endoftext|>"]
    corpus_path="/home/kuangph/CS336-Assignment1/data/11G.txt"
    memmap_manager=Memmap_Manager(chunk_size,vocab_path,merge_path,special_tokens,corpus_path,"11G")

    print(f"chunk begin")
    memmap_manager.save_as_memmap()

    token_ids=memmap_manager.load_by_range(0,150)
    print(f"loaded {len(token_ids)} tokens")
    input("press enter to continue")
    tokenizer=BPE_Tokenizer.from_files(vocab_path,merge_path,special_tokens)
    decoded_text=tokenizer.decode(token_ids)
    print(f"decoded text:{decoded_text}")
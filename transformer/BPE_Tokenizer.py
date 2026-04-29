from collections import defaultdict
import regex
import tracemalloc

class BPE_Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merge_list:list[tuple[bytes,bytes]],special_tokens=None):
        self.vocab=vocab
        self.vocab_reverse={v:k for k,v in vocab.items()}
        self.special_tokens=special_tokens
        if self.special_tokens is not None:
            if len(self.special_tokens)==2:
                if self.special_tokens[0]*2==self.special_tokens[1]:
                    temp=self.special_tokens[0]
                    self.special_tokens[0]=self.special_tokens[1]
                    self.special_tokens[1]=temp
        
        merge_dict={pair:i for i,pair in enumerate(merge_list)}
        self.merge_dict=merge_dict
    
    def from_files(vocab_path:str,merge_path:str,special_tokens=None):
        vocab={}
        #read vocab file,this file is a txt file,the file format is like {"!": 0, "\"": 1, "#": 2, "$": 3, ……,only one line
        with open(vocab_path,"r",encoding="utf-8") as f:
            import ast
            vocab_str = f.read()
            vocab_raw = ast.literal_eval(vocab_str)
            vocab = {k:v for k, v in vocab_raw.items()}

        #read merge file,this file is like:[(b'h', b'e'), (b' ', b't'), (b' ', b'a'), (b' ', b's'), (b' ', b'w'), (b'n', b'd'), (b' t', b'he'),……,one of this two parts can be ' '.only one line
        merge_list=[]
        with open(merge_path,"r",encoding="utf-8") as f:
            import ast
            merge_str = f.read()
            merge_list = ast.literal_eval(merge_str)

        return BPE_Tokenizer(vocab,merge_list,special_tokens)

    def _chunk_text(self,text:str):
        if self.special_tokens is None or len(self.special_tokens)==0:
            yield text
            return
        pattern = "(" + "|".join([regex.escape(token) for token in self.special_tokens]) + ")"
        regex_chunk = regex.compile(pattern)
        chunks = regex_chunk.split(text)
        for chunk in chunks:
            if chunk:
                yield chunk

    def _encode_merge(self,token_list:list[bytes]):
        dict_idx=defaultdict(int)

        # 列出该token的所有pair，并检查是否有pair在merge_dict中
        # merge_dict即为从pair到合并序号的映射
        for i in range(len(token_list)-1):
            pair=(token_list[i],token_list[i+1])
            if pair in self.merge_dict:
                dict_idx[pair]=self.merge_dict[pair]
        # 不可继续合并
        if len(dict_idx)==0:
            return token_list,False
        
        min_number=998244353
        min_pair=None
        # 找出合并序号最小的pair
        for pair,number in dict_idx.items():
            #print(f">>>>>pair:{pair},number:{number}")
            if number<min_number:
                min_number=number
                min_pair=pair

        # 进行合并
        new_token=min_pair[0]+min_pair[1]
        for i in range(len(token_list)-1):
            pair=(token_list[i],token_list[i+1])
            if pair==min_pair:
                # token_list改为：合并后的token,前后拼接上原token_list
                token_list=token_list[:i]+[new_token]+token_list[i+2:]
                break
        return token_list,True

    def _process_char(self,char):
        return bytes([char])
    
    def _process_encode(self,ori_text:str):
        # 对待编码文本进行Pre-tokenization
        texts=self._chunk_text(ori_text)

        GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens is None:
            self.special_tokens=[]
        special_tokens_pattern = "|".join([regex.escape(token) for token in self.special_tokens])
        pattern_str = f"({special_tokens_pattern})|{GPT2_SPLIT_PATTERN}"
        pattern = regex.compile(pattern_str, flags=regex.UNICODE)

        encoded_text_list=[]

        for text in texts:
            # text即为单个文本
            text_split=pattern.finditer(text)
            for token in text_split:
                token_str=token.group()
                # token_str即为单个token

                # 如果是special token，直接编码
                if token_str in self.special_tokens:
                    token_bytes=token_str.encode("utf-8")
                    encoded_text_list.append(self.vocab_reverse[token_bytes])
                    continue
                # 否则转成utf-8字节，进行合并操作
                token_list=[self._process_char(b) for b in token_str.encode("utf-8")]

                can_merge=True
                while can_merge:
                    token_list,can_merge=self._encode_merge(token_list)
                    #print(f"finish a merge,token_list:{token_list}")
                    #input("press enter to continue")

                encoded_list=[self.vocab_reverse[token] for token in token_list if token in self.vocab_reverse]
                encoded_list=[self.vocab[token] for token in encoded_list]

                encoded_text_list.extend([self.vocab_reverse[token] for token in encoded_list])
                #print(f"\nnew encoded text:{encoded_list}")
                #print(f"encoded_list:{encoded_text_list}")
                #print()

        return encoded_text_list  
    
    def encode(self,ori_text:str)->list[int]:
        encoded_text_list=[]
        encoded_text_list=self._process_encode(ori_text)
        return encoded_text_list


    def encode_iterable(self,iterable):
        for line in iterable:
            #input("press enter to continue")
            #print(f"text:{text}")
            encoded_line=self.encode(line)
            for id in encoded_line:
                yield id

    def decode(self,ids:list[int])->str:
        bytes_list=[self.vocab[id] for id in ids]
        bytes_string=b''.join(bytes_list)
        decoded_string=bytes_string.decode("utf-8",errors="ignore")
        return decoded_string
    
if __name__=="__main__":
    #tracemalloc.start()
    tokenizer=BPE_Tokenizer.from_files("/home/kuangph/CS336-Assignment1/HIDE/data/vocab_32000.txt","/home/kuangph/CS336-Assignment1/HIDE/data/merges_32000.txt",special_tokens=["<|endoftext|>"])
    vocab_rev=tokenizer.vocab_reverse
    merges=tokenizer.merge_dict

    with open("/home/kuangph/CS336-Assignment1/HIDE/data/simple.txt") as f:
        text=f.read()
        print(f"text:")
        print(text)
        encoded_ids=tokenizer.encode(text)
        #for id in encoded_ids:
        #    print(f"id:{id},vocab:{tokenizer.vocab[id]}")
        
        print(f"\nencoded ids:{encoded_ids}\n")
        decoded_text=tokenizer.decode(encoded_ids)
        print(f"decoded text:")
        print(decoded_text)
        ''''''

    #peak=tracemalloc.get_traced_memory()[1]
    #print(f"peak memory usage:{peak/1024/1024} MB")
''''''
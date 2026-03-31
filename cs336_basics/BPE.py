from collections import defaultdict
import os
from typing import BinaryIO
import multiprocessing
import re
import heapq
import time
import regex
import tracemalloc

utf2int={}
int2utf={}

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunk_text(chunk_block,special_tokens):
    # 正则表达式：按照special_tokens切分文本块
    pattern=b"("+b"|".join([regex.escape(token.encode("utf-8")) for token in special_tokens])+b")"
    regex_chunk=regex.compile(pattern)
    chunks=regex.split(regex_chunk,chunk_block)
    for chunk in chunks:
        if chunk:
            yield chunk

def process_chunk(start,end,special_tokens,input_path):
    #print(f"Processing chunk from {start} to {end}")
    pair_positions=defaultdict(list)
    token_dict=defaultdict(int)
    pair_counter=defaultdict(int)
    next=defaultdict(int)
    prev=defaultdict(int)

    # 读入该区间的分块语料库文本
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk_block=f.read(end-start)
    
    # 按special tokens（如<|endoftext|>）切分文本
    texts=chunk_text(chunk_block,special_tokens)

    # 制定粗略整词分割的规则
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens_pattern = "|".join([re.escape(token) for token in special_tokens])
    pattern = f"({special_tokens_pattern})|{GPT2_SPLIT_PATTERN}"
    pattern = regex.compile(pattern, flags=regex.UNICODE)
    ''''''
    
    token_idx=start
    for text in texts:# 逐个文档遍历
        #print("chunk a text:",text)

        # 按照正则表达式切分文本为token
        text_split=pattern.finditer(text.decode("utf-8",errors='ignore'))

        for token in text_split:# 逐个token遍历
            token_str=token.group()
            token=token_str.encode("utf-8")
            #print(">>>>>split a token:",token)

            if token_str in special_tokens:
                token_idx+=len(token) # 整个跳过，同时保证位置编号也跳过
                continue

            prev_token=None
            for c in token:
                token_dict[token_idx]=c # 为该token进行位置编号
                next[token_idx]=None # 在整词的第一个token，默认prev_token等全都为None
                prev[token_idx]=None
                if prev_token is not None: # 
                    pair_positions[(prev_token,c)].append(token_idx-1) # 记录该pair出现的位置
                    pair_counter[(prev_token,c)]+=1 # 更新该pair出现的次数
                    next[token_idx-1]=token_idx # 更新该token的next与prev
                    prev[token_idx]=token_idx-1
                token_idx+=1
                prev_token=c
    '''
    print("pair_positions:")
    for pair in pair_positions:
        char_of_pairs=(int2utf[pair[0]],int2utf[pair[1]])
        print(f"pair:{pair},char:{char_of_pairs},positions:{pair_positions[pair]}")
    '''
    '''
    print("token_dict:")
    for idx in token_dict:
        print(f"idx:{idx},char:{int2utf[token_dict[idx]]}")
    '''
    '''
    print("pair_counter:")
    for pair in pair_counter:
        char_of_pairs=(int2utf[pair[0]],int2utf[pair[1]])
        print(f"pair:{pair},char:{char_of_pairs},count:{pair_counter[pair]}")
    '''
    '''
    print("next:")
    for idx in next:
        if next[idx] is not None:
            print(f"idx:{idx}, char:{int2utf[token_dict[idx]]}, next_idx:{next[idx]}, next_char:{int2utf[token_dict[next[idx]]]}")
        else:
            print(f"idx:{idx}, char:{int2utf[token_dict[idx]]}, next_idx:None, next_char:None")

    print("prev:")
    for idx in prev:
        if prev[idx] is not None:
            print(f"idx:{idx}, char:{int2utf[token_dict[idx]]}, prev_idx:{prev[idx]}, prev_char:{int2utf[token_dict[prev[idx]]]}")
        else:
            print(f"idx:{idx}, char:{int2utf[token_dict[idx]]}, prev_idx:None, prev_char:None")
    '''
    
    return pair_positions,token_dict,pair_counter,next,prev

def BPE_init(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str],
        vocab_tot
    ):
    # 建立初始词表：256对token与整数编码的映射，外加特殊token，如<|endoftext|>
    for i in range(256):
        utf2int[bytes([i])]=i
        int2utf[i]=bytes([i])
    for i in special_tokens:
        utf2int[i.encode("utf-8")]=vocab_tot
        int2utf[vocab_tot]=i.encode("utf-8")
        vocab_tot+=1

    with open(input_path,"rb") as f:
        # 这里运用了一个多进程分割的技巧来并行处理大语料库文件
        # 最终的期望：以<|endoftext|>为界限将文件划分为多个块，每个块由一个进程处理
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        tasks=[]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((start,end,special_tokens,input_path))
        
        global_pair_positions=defaultdict(list)
        global_token_dict=defaultdict(int)
        global_pair_counter=defaultdict(int)
        global_next=defaultdict(int)
        global_prev=defaultdict(int)

        # 将各个语料库分割块进行process_chunk预处理
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, tasks)

            # 将各个进程的预处理结果合并
            for result in results:
                pair_positions,token_dict,pair_counter,next,prev=result
                # Merge pair_positions
                for pair, positions in pair_positions.items():
                    if pair in global_pair_positions:
                        global_pair_positions[pair].extend(positions)
                    else:
                        global_pair_positions[pair] = positions

                # Merge token_dict
                global_token_dict.update(token_dict)

                # Merge pair_counter
                for pair, count in pair_counter.items():
                    global_pair_counter[pair] += count

                # Merge next and prev dictionaries
                global_next.update(next)
                global_prev.update(prev)
    # 合并后的各全局数据结构
    return global_pair_positions,global_token_dict,global_pair_counter,global_next,global_prev
    
def BPE_merge(pair_positions,token_dict,pair_counter,next,prev,heap,vocab_tot):

    # 取出出现频率最高的pair
    pair_freq,pair_chosen=heapq.heappop(heap)
    pair_freq=-pair_freq
    # lazy update的做法：如果与计数器不符，说明该pair其实已被删除
    while pair_freq!=pair_counter[pair_chosen]:
        if not heap:
            return
        pair_freq,pair_chosen=heapq.heappop(heap)
        pair_freq=-pair_freq

    # 将pair存储以便后续比较字典序
    pair_name=(int2utf[pair_chosen[0]],int2utf[pair_chosen[1]])
    pair_name_set=set()
    pair_name_set.add(pair_name)
    pair_set=set()
    pair_set.add(pair_chosen)

    # 取出所有频率并列最高的pairs，并存储pair以便后续比较字典序
    while heap[0][0]==-pair_freq:
        pair=heapq.heappop(heap)
        if pair_freq!=pair_counter[pair[1]]:
            continue
        pair_name=(int2utf[pair[1][0]],int2utf[pair[1][1]])
        pair_name_set.add(pair_name)
        pair_set.add(pair[1])
    
    # 选出字典序最大的pair作为最终合并的pair
    max_pair_name=max(pair_name_set)
    for pair in pair_set:
        pair_name=(int2utf[pair[0]],int2utf[pair[1]])
        if pair_name==max_pair_name:
            pair_chosen=pair
            pair_set.remove(pair)
            break
    
    # 将其他pair重新放回优先队列
    for pair in pair_set:
        heapq.heappush(heap,(-pair_counter[pair],pair))
    
    # 开始合并操作
    pair0=pair_chosen[0]
    pair1=pair_chosen[1]
    
    # 生成新的token，更新至词表
    new_token=int2utf[pair_chosen[0]]+int2utf[pair_chosen[1]]#bytes
    utf2int[new_token]=vocab_tot
    int2utf[vocab_tot]=new_token

    # update set of previous tokens and next tokens
    prev_set=set()
    next_set=set()

    for index in pair_positions[pair_chosen]:
        # pair0和pair1的idx
        idx0=index
        idx1=next[idx0]

        # 检验该位置的pair是否被修改过
        if token_dict[idx0]!=pair0 or token_dict[idx1]!=pair1:
            continue

        # 获取prev_token及其位置下标
        if prev[idx0] is not None:
            prev_idx=prev[idx0]
            prev_token=token_dict[prev_idx]
            prev_set.add(prev_token)
        else:
            prev_token=None
            prev_idx=None

        # 获取next_token及其位置下标
        if next[idx1] is not None:
            next_idx=next[idx1]
            next_token=token_dict[next_idx]
            next_set.add(next_token)
        else:
            next_token=None
            next_idx=None

        # 计数器更新
        if prev_idx is not None:
            pair_counter[(prev_token,pair0)]-=1
            pair_counter[(prev_token,vocab_tot)]+=1

        if next_idx is not None:
            pair_counter[(pair1,next_token)]-=1
            pair_counter[(vocab_tot,next_token)]+=1
        
        # pair_positions更新
        if prev_idx is not None:
            pair_positions[(prev_token,vocab_tot)].append(prev_idx)

        if next_idx is not None:
            pair_positions[(vocab_tot,next_token)].append(idx0)
        
        # next prev双向链表更新
        next[idx0]=next_idx
        if next_idx is not None:
            prev[next_idx]=idx0

        # token_dict（位置下标更新）
        token_dict[idx0]=vocab_tot
        token_dict[idx1]=-1

    # 被合并的pair，各种信息清零
    pair_counter[pair_chosen]=0
    pair_positions[pair_chosen]=[]

    # 将涉及到修改的所有pairs加入优先队列
    # 各个(prev_token,new) (prev_token,pair0) (new,next_token) (pair1,next_token)
    for prev_token_id in prev_set:
        pair=(prev_token_id,vocab_tot)
        count=pair_counter[pair]
        heapq.heappush(heap,(-count,pair))

        pair=(prev_token_id,pair0)
        count=pair_counter[pair]
        heapq.heappush(heap,(-count,pair))

    for next_token_id in next_set:
        pair=(vocab_tot,next_token_id)
        count=pair_counter[pair]
        heapq.heappush(heap,(-count,pair))

        pair=(pair1,next_token_id)
        count=pair_counter[pair]
        heapq.heappush(heap,(-count,pair))

    return pair_chosen

def BPE(input_path:str,vocab_size:int,special_tokens:list[str]):
    vocab_tot=256
    #print("Memory before BPE_init:", tracemalloc.get_traced_memory()[0] / 1024 / 1024, "MB")
    result=BPE_init(input_path,vocab_size,special_tokens,vocab_tot)
    #print("Initialization done")
    #print("Memory after BPE_init:", tracemalloc.get_traced_memory()[0] / 1024 / 1024, "MB")
    pair_positions=result[0]
    token_dict=result[1]
    pair_counter=result[2]
    next=result[3]
    prev=result[4]
    heap=[] # 建立一个优先队列，使得每次能够快速查找出现频率最高的pair
    merge_list=[] # 每次合并操作的具体合并pair记录在此
    for pair in pair_counter:
        count=pair_counter[pair]
        heapq.heappush(heap,(-count,pair))
    #print("Heap initialization done")
    #print("Memory after heap initialization:", tracemalloc.get_traced_memory()[0] / 1024 / 1024, "MB")

    while vocab_tot<vocab_size-1:
        if not heap:
            break
        vocab_tot+=1
        pair_merged=BPE_merge(pair_positions,token_dict,pair_counter,next,prev,heap,vocab_tot)
        merge_list.append(pair_merged)
        #if vocab_tot%100==0 or vocab_tot<300:
        #    print(f"Current vocab size:{vocab_tot},last merged pair:{pair_merged},total pairs in heap:{len(heap)}")

    #change all items in merge list from int to bytes
    merge_list_bytes=[]
    for pair in merge_list:
        merge_list_bytes.append((int2utf[pair[0]],int2utf[pair[1]]))
    #print("Memory after BPE merge:", tracemalloc.get_traced_memory()[0] / 1024 / 1024, "MB")
    #print("Memory peak:", tracemalloc.get_traced_memory()[1] / 1024 / 1024, "MB")
    return int2utf,merge_list_bytes

def export2file(vocabulary,bytes_merge_list):
    vocab_path = "/home/kuangph/CS336-Assignment1/HIDE/data/vocab_1000.txt"
    merges_path = "/home/kuangph/CS336-Assignment1/HIDE/data/merges_1000.txt"
    import os
    os.makedirs("data", exist_ok=True)

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(str(vocabulary) + "\n")

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write(str(bytes_merge_list) + "\n")

if __name__=="__main__":
    input_path="/home/kuangph/CS336-Assignment1/HIDE/data/simple.txt"
    vocab_size=1000
    special_tokens=["<|endoftext|>"]
    vocabulary,merge_list=BPE(input_path,vocab_size,special_tokens)

    #export2file(vocabulary,merge_list)

'''

if __name__ == "__main__":
    #tracemalloc.start()

    input_path="data/owt_valid.txt"#5M.txt
    vocab_size=32000
    special_tokens=["<|endoftext|>"]
    time_start=time.time()
    #print("Training begin")
    result=BPE(input_path,vocab_size,special_tokens)
    vocabulary,bytes_merge_list=result
    time_end=time.time()
    #print(f"Total time: {time_end-time_start} seconds")
    
    # export vocabulary and bytes_merge_list to json and txt file
    export2file(vocabulary,bytes_merge_list)

'''
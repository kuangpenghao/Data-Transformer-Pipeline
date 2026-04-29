import hashlib
import os
from collections import defaultdict
import unicodedata
import regex
import networkx
from pathlib import Path

def get_hash(line:str):
    return hashlib.md5(line.encode('utf-8')).hexdigest()

def exact_line_ded(input_dirs:list[str],output_dir:str):
    os.makedirs(output_dir,exist_ok=True)

    line_count=defaultdict(int)
    
    for input_dir in input_dirs:
        input_dir=Path(input_dir)
        with input_dir.open('r') as f:
            text=f.read()
            lines=text.split('\n')
        
        for line in lines:
            line=get_hash(line)
            line_count[line]+=1

    for input_dir in input_dirs:
        input_dir=Path(input_dir)
        with input_dir.open('r') as f:
            lines=f.read().split('\n')

        new_lines=[]
        for i in range(len(lines)):
            line=lines[i]
            if line_count[get_hash(line)]==1 or line.strip()=='':
                new_lines.append(line)

        # prevent over 3 empty lines appearing continuously
        line_desert=[False]*len(new_lines)
        last_empty=(new_lines[0].strip()=='')
        for i in range(1,len(new_lines)):
            line_empty=(new_lines[i].strip()=='')
            if line_empty and last_empty:
                line_desert[i]=True
            last_empty=line_empty

        final_lines=[]
        for i in range(len(new_lines)):
            if line_desert[i] is False:
                final_lines.append(new_lines[i])

        new_text='\n'.join(final_lines)

        output_dir=Path(output_dir)
        output_path=os.path.join(output_dir,input_dir.name)
        with open(output_path,'w') as f:
            f.write(new_text)

def text_normalization(text:str):
    text=text.lower()
    text=unicodedata.normalize("NFD", text)
    text=regex.sub(r"\p{Mn}+", "", text)
    text=regex.sub(r"[^a-z0-9]+", " ", text)
    text=text.strip()
    return text

def text_to_minhash(text:str,num_hashes:int,n_grams:int):
    k_mins=[-1]*num_hashes
    tokens=text.split(' ')
    n_grams_set=set()

    for i in range(len(tokens)-n_grams+1):
        n_gram='|'.join(tokens[i:i+n_grams])
        if n_gram in n_grams_set:
            continue
        n_grams_set.add(n_gram)

        for k in range(num_hashes):
            hash_key=n_gram+'|||'+str(k)
            hash_value=get_hash(hash_key)
            hash_value=int(hash_value,16)
            if k_mins[k]==-1 or hash_value<k_mins[k]:
                k_mins[k]=hash_value

    return k_mins

def pre_LSH(min_hashes:list,num_bands:int):
    b=num_bands
    r=len(min_hashes)//b
    assert r*b==len(min_hashes)
    min_hashes=[min_hashes[start:start+r] for start in range(0,b*r,r)]
    return min_hashes
    
def make_buckets(min_hashes:list,num_bands:int,num_texts:int):
    buckets=[]
    for k in range(num_bands):
        band_dict=defaultdict(list)
        for i in range(num_texts):
            band_key='|'.join([str(x) for x in min_hashes[i][k]])
            band_dict[band_key].append(i)

        for key in band_dict:
            text_idxs=band_dict[key]
            if len(text_idxs)<=1:
                continue
            for i in range(len(text_idxs)-1):
                buckets.append((text_idxs[i],text_idxs[i+1]))
    return buckets

def get_ngrams(text_dir:str,n_grams:int):
    text_dir=Path(text_dir)
    with text_dir.open('r') as f:
        text=f.read()
        text=text_normalization(text)
        tokens=text.split(' ')
        n_grams_set=set()
        for i in range(len(tokens)-n_grams+1):
            n_gram='|'.join(tokens[i:i+n_grams])
            n_grams_set.add(n_gram)
    return n_grams_set

def calc_jaccard(set1:set,set2:set):
    intersection=set1&set2
    union=set1|set2
    jaccard=len(intersection)/len(union) if len(union)>0 else 0.0
    return jaccard

def Minhash_and_LSH_deduplication(input_dirs:list[str],
                                  output_dir:str,
                                  num_hashes:int,
                                  num_bands:int,
                                  n_grams:int,
                                  jaccard_threshold:float):
    
    min_hashes=[]
    num_texts=len(input_dirs)
    
    for input_dir in input_dirs:
        input_dir=Path(input_dir)
        with input_dir.open('r') as f:
            text=f.read()
            text=text_normalization(text)
            hashes=text_to_minhash(text,num_hashes,n_grams)
            min_hashes.append(pre_LSH(hashes,num_bands))

    print("Minhashes calculated.")

    buckets=[]
    buckets=make_buckets(min_hashes,num_bands,num_texts)

    same_texts=[]
    for bucket in buckets:
        text1,text2=bucket
        set1=get_ngrams(input_dirs[text1],n_grams)
        set2=get_ngrams(input_dirs[text2],n_grams)
        jaccard=calc_jaccard(set1,set2)
        if jaccard>=jaccard_threshold:
            same_texts.append((text1,text2))

    print(f"Found {len(same_texts)} similar text pairs.")

    G=networkx.Graph()
    G.add_edges_from(same_texts)
    clusters=list(networkx.connected_components(G))

    still_accept=[True]*num_texts
    for cluster in clusters:
        for i in range(1,len(cluster)):
            still_accept[list(cluster)[i]]=False
    
    output_dir=Path(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    for i in range(num_texts):
        if still_accept[i] is False:
            continue
        input_dir=input_dirs[i]
        input_dir=Path(input_dir)
        with input_dir.open('r') as f:
            text=f.read()

        output_path=os.path.join(output_dir,input_dir.name)
        with open(output_path,"w") as f:
            f.write(text)
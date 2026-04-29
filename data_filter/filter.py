import nltk
import fasttext

def one_alphabet(word:str):
    num_alpha=0
    for char in word:
        num_alpha+=char.isalpha()
    return num_alpha==1

def naive_filter(text:str):
    
    words=nltk.word_tokenize(text)
    words_len=[len(word) for word in words]

    if len(words)<50 or len(words)>100000:
        return False

    avg_word_len=sum(words_len)/len(words)
    if avg_word_len<3 or avg_word_len>10:
        return False

    total_lines=0
    total_dots=0
    for line in text.split('\n'):
        line=line.strip()
        if not line:
            continue

        total_lines+=1
        if line.endswith('...') or line.endswith('…'):
            total_dots+=1
    if total_lines>0 and total_dots/total_lines>0.3:
        return False

    is_single=[one_alphabet(word) for word in words]
    if sum(is_single)/len(words)>0.2:
        return False
    
    return True

def filter_model_training(input_path:str,
                          save_path:str,
                          lr:float=0.1,
                          epoch:int=30,
                          word_ngrams:int=4,
                          verbose:int=2,
                          min_count:int=1,
                          loss:str="hs"):
    model=fasttext.train_supervised(
        input=input_path,
        lr=lr,
        epoch=epoch,
        wordNgrams=word_ngrams,
        verbose=verbose,
        minCount=min_count,
        loss=loss
    )

    model.save_model(save_path)
    print(f"Model saved to {save_path}")

def filter_by_model(text:str,
                    model_path:str='/home/kuangph/Data/data/models/quality30.bin'):
    model=fasttext.load_model(model_path)

    text=' '.join(text.split())

    labels,probs=model.predict(text,k=1)
    label=labels[0]
    prob=probs[0]

    return label,prob

if __name__=="__main__":
    '''
    input_path="data/high_low_texts.txt"
    save_path="data/models/quality30.bin"

    filter_model_training(input_path,
                          save_path,
                          lr=0.75,
                          epoch=30,
                          word_ngrams=4,
                          verbose=2,
                          min_count=1,
                          loss="hs")
    '''
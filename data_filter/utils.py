import langid
import math
import regex
import fasttext

def detect_language(text:str):
    detection=langid.rank(text)
    
    languages=[lang for lang,_ in detection]
    probabilities=[prob for _,prob in detection]
    max_prob=max(probabilities)
    probabilities=[prob-max_prob for prob in probabilities]
    probabilities=[math.exp(prob) for prob in probabilities]
    sum_prob=sum(probabilities)
    probabilities=[prob/sum_prob for prob in probabilities]

    lang=languages[0]
    prob=probabilities[0]

    return lang,prob

def process_emails(text:str):
    email_re=r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    email_pattern=regex.compile(email_re)

    emails_found=email_pattern.findall(text)
    num_masked=len(emails_found)

    text_processed=email_pattern.sub("|||EMAIL_ADDRESS|||",text)

    return text_processed,num_masked

def process_phone_numbers(text:str):
    number_re=r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
    number_pattern=regex.compile(number_re)

    numbers_found=number_pattern.findall(text)
    num_masked=len(numbers_found)
    text_processed=number_pattern.sub("|||PHONE_NUMBER|||",text)

    return text_processed,num_masked

def process_ips(text:str):
    ip_re=r"(?:(?:25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[01]?\d{1,2})\.){3}(?:25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[01]?\d{1,2})"
    ip_pattern=regex.compile(ip_re)

    ips_found=ip_pattern.findall(text)
    num_masked=len(ips_found)

    text_processed=ip_pattern.sub("|||IP_ADDRESS|||",text)

    return text_processed,num_masked

def process_nsfw(text:str):
    model_path='/home/kuangph/Data/data/models/nsfw.bin'
    model=fasttext.load_model(model_path)

    lines=text.split('\n')
    probs=[]
    lengths=[max(20,len(line)) for line in lines]
    total_length=sum(lengths)
    prediction=[model.predict(line,k=1) for line in lines]

    for pred,line_len in zip(prediction,lengths):
        if pred[0][0]=='__label__nsfw':
            probs.append(pred[1][0]*max(20,line_len))
        else:
            probs.append((1.0-pred[1][0])*max(20,line_len))
    avg_prob=sum(probs)/total_length if total_length>0 else 0.0

    label="__label__nsfw" if avg_prob>0.05 else "__label__non-nsfw"
    prob=avg_prob if label=="__label__nsfw" else 1.0-avg_prob
    return label,prob

def process_toxic(text:str):
    model_path='/home/kuangph/Data/data/models/toxic.bin'
    model=fasttext.load_model(model_path)

    lines=text.split('\n')
    probs=[]
    lengths=[max(len(line),20) for line in lines]
    total_length=sum(lengths)
    prediction=[model.predict(line,k=1) for line in lines]

    for pred,line_len in zip(prediction,lengths):
        if pred[0][0]=='__label__toxic':
            probs.append(pred[1][0]*max(20,line_len))
        else:
            probs.append((1.0-pred[1][0])*max(20,line_len))
    avg_prob=sum(probs)/total_length if total_length>0 else 0.0

    label="__label__toxic" if avg_prob>0.05 else "__label__non-toxic"
    prob=avg_prob if label=="__label__toxic" else 1.0-avg_prob
    return label,prob
    

if __name__=="__main__":
    text="The discussion quickly turned toxic and abusive.\n You lazy, incompetent moron!\n Your pathetic ideas are pure garbage, absolute BULLSHIT!\n Go F*CK yourself and take that STUPID website with you.\n No one wants to hear your CRAP, you absolute waste of space.\n This whole thread is a JOKE.\n After the explosion, the user quickly calmed down and signed off with their email: support-team@example.com."
    nsfw=process_nsfw(text)
    print(f"NSFW={nsfw[0]}, PROB={nsfw[1]}")
    toxic=process_toxic(text)
    print(f"TOXIC={toxic[0]}, PROB={toxic[1]}")
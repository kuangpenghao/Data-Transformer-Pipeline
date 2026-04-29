import os
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from cs336_data.utils import *
from cs336_data.filter import naive_filter,filter_by_model
from cs336_data.deduplication import Minhash_and_LSH_deduplication,exact_line_ded

def filter_text(processed_text:str):
    # filter out some known phrases
    filter_list=["This website is using a security service to protect itself"]
    for filter_item in filter_list:
        if filter_item in processed_text:
            return None
    
    # filter out emails, phone numbers, IPs
    processed_text,_=process_emails(processed_text)
    processed_text,_=process_phone_numbers(processed_text)
    processed_text,_=process_ips(processed_text)
 
    # filter out NSFW and toxic text
    nsfw_result=process_nsfw(processed_text)
    toxicity_result=process_toxic(processed_text)
    if nsfw_result[0]=='__label__nsfw' or toxicity_result[0]=='__label__toxic':
        return None
        
    # delete short lines
    remove_short=processed_text.split('\n')
    if not remove_short:
        return None
    remove_short=[line.strip() for line in remove_short if len(line)>=50]
    remove_short='\n'.join(remove_short)
    
    # filter again after line deletion
    if naive_filter(remove_short) is False:
        return None
    if filter_by_model(remove_short)[0]=="__label__low_quality":
        return None
    
    return processed_text

def write_to_file(content:str,quality_label="__label__high_quality"):

    if quality_label=="__label__high_quality":
        file_path="data/wiki_texts-justfortest.txt"
    else:
        file_path="data/low_quality_texts.txt"

    with open(file_path,"a") as f:
        content=content.split('\n')

        def valid_line(line:str):
            line=line.strip()
            if len(line)<50:
                return None
            # if the line starts with numbers or special characters, remove it
            if line[0].isdigit() or not line[0].isalpha():
                return None
            return line
        
        if quality_label=="__label__high_quality":
            content=[valid_line(line) for line in content if valid_line(line) is not None]
        else:
            content=[line.strip() for line in content if line.strip()]
        content=' '.join(content)
        content=quality_label+' '+content+"\n"
        f.write(content)

def export_to_file(content:str,export_path:str,files_count:int):
    os.makedirs(export_path,exist_ok=True)
    export_file_path=os.path.join(export_path,f"extracted_{files_count}.txt")

    content=content.strip()
    content=content.split('\n')

    def valid_line(line:str):
        if len(line.strip())==0:
            return line
        line=line.strip()
        if len(line)<30:
            return None
        # if the line starts with numbers or special characters, remove it
        if line[0].isdigit() or not line[0].isalpha():
            return None
        return line

    with open(export_file_path,"w") as f:
        # write line by lines
        for line in content:
            v_line=valid_line(line)
            if v_line is not None:
                f.write(v_line+'\n')

    return export_file_path


def parse_html(content_bytes:bytes,should_filter:bool=True):
        decode_method=detect_encoding(content_bytes)

        try:
            html_content=content_bytes.decode(decode_method,errors='ignore')
            processed_text=extract_plain_text(html_content)

            # filter out non-English text
            language_result=detect_language(processed_text)
            if language_result[0] != 'en':
                return None

            if should_filter is True:
                processed_text=filter_text(processed_text)
                if processed_text is None:
                    return None

            return processed_text
        
        except Exception as e:
            print(f"{'>'*25}{decode_method} decode error{'<'*25}")
            print(f"Error details: {e}")
            return None

def parse_warc(file_path:str,
               should_filter:bool=True,
               should_write:bool=False,
               quality_label="__label__high_quality",
               export_path=None,
               deduplication_path=None):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    lines_now=0
    total_lines=9000
    files_count=0
    total_files=2000

    with open(file_path,"rb") as stream:
        for i,record in enumerate(ArchiveIterator(stream)):

            if lines_now==total_lines or files_count==total_files:
                break
            
            warc_type=record.record_type
            if not warc_type==WarcRecordType.response:
                continue

            content_type=record.http_headers.get('Content-Type','')
            if not 'text/html' in content_type:
                continue

            content_bytes=record.reader.read()

            print(f"read a text in record {i}")

            text=parse_html(content_bytes,should_filter)

            if text is not None and len(text.strip())>0:
                print(f"valid text.")

                # write to file, for training quality classification model
                if should_write is True:
                    write_to_file(text,quality_label)
                    lines_now+=1
                    print(f"lines_now: {lines_now}/{total_lines}")
                
                # export texts without deduplication to files
                if export_path is not None:
                    export_file_path=export_to_file(text,export_path,files_count)
                    files_count+=1
                    print(f"Exported to {export_file_path}, total files: {files_count}")
            else:
                print(f"text filtered.")

    # deduplication after all texts are exported
    if deduplication_path is not None and export_path is not None:
        os.makedirs(deduplication_path,exist_ok=True)
        file_names=os.listdir(export_path)
        input_files=[os.path.join(export_path,filename) for filename in file_names if filename.endswith('.txt')]
        
        exact_line_ded(input_files,deduplication_path)

        print("Exact line deduplication done.")

        # remove all files in export_path, and move all files in deduplication_path to export_path
        for filename in os.listdir(export_path):
            file_path=os.path.join(export_path,filename)
            os.remove(file_path)
        for filename in os.listdir(deduplication_path):
            src_path=os.path.join(deduplication_path,filename)
            dst_path=os.path.join(export_path,filename)
            os.rename(src_path,dst_path)

        print(f"Moved deduplicated files to {export_path}")

        Minhash_and_LSH_deduplication(input_files,
                                      deduplication_path,
                                      num_hashes=150,
                                      num_bands=150,
                                      n_grams=5,
                                      jaccard_threshold=0.8)
        
        print(f"Deduplication done, results are in {deduplication_path}")


if __name__=="__main__":
    file_path="/home/kuangph/Data/data/example.warc.gz"
    export_path="/home/kuangph/Data/data/extracted_texts"
    deduplication_path="/home/kuangph/Data/data/deduplicated_texts"
    parse_warc(file_path,
               should_filter=True,
               should_write=False,
               quality_label="__label__high_quality",
               export_path=export_path,
               deduplication_path=deduplication_path)
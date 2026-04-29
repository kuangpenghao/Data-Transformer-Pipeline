if __name__=="__main__":
    high_path="data/wiki_texts.txt"
    low_path="data/low_quality_texts.txt"
    combined_path="data/high_low_texts.txt"

    # for data in high_path,repeat it locally
    with open(high_path,"r") as f:
        high_texts=f.read()
    print(f"read original high quality texts")
    input("Press Enter to continue...")

    with open(high_path,"a") as f:
        f.write(high_texts)
    print(f"duplicated high quality texts")
    input("Press Enter to continue...")
    
    with open(high_path,"r") as f_high, open(low_path,"r") as f_low, open(combined_path,"w") as f_combined:
        high_lines=f_high.readlines()
        low_lines=f_low.readlines()

        min_len=min(len(high_lines),len(low_lines))
        for i in range(min_len):
            f_combined.write(high_lines[i].strip()+'\n')
            f_combined.write(low_lines[i].strip()+'\n')
        
        # write remaining lines
        if len(high_lines)>min_len:
            for i in range(min_len,len(high_lines)):
                f_combined.write(high_lines[i].strip()+'\n')
        if len(low_lines)>min_len:
            for i in range(min_len,len(low_lines)):
                f_combined.write(low_lines[i].strip()+'\n')
        
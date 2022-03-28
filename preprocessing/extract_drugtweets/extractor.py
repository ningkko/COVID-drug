#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ning
"""

import os
import re
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
import glob
import warnings
import tqdm
from pandarallel import pandarallel
pandarallel.initialize()
warnings.filterwarnings("ignore", category=FutureWarning)


def get_raw_data(data_path):
    """
    given a data path, iteratively find .csv files
    """
    folders = sorted([f for f in glob.glob(data_path+"*") if "-" in f])
    files = []
    for folder in folders:
        files.extend(glob.glob(folder+"/*.csv"))
    files = sorted(files)
    return files

def extract_screenname_from_user(x):
    if not x:
        return ""
    result = re.search("screen_name\': \'(.*?)\'", x)
    if result:
        return result.group(1).strip()
    return ""

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def main():
    """
    Extract tweets with keywords. 
    """
    data_path = "../../../Dropbox/"
    out_dir = "../../data2/" ## where to store outputs?
    os.makedirs(out_dir, exist_ok=True)

    altered_input_dir = out_dir + "../../data2/raw/"  ## we'll do a filtering out of columns we don't need, and store the filtered data in a new dir 
    # os.makedirs(altered_input_dir, exist_ok=True)

    ## Set patterns to extract from tweets
    keywords_dict = {"hcq":"ydroxych| hcq |plaqu |plaquenil|hydroquin|axemal",
                    "ivermectin": "ivermectin|stromectol|soolantra|sklice",
                    "remdesivir": "remdesivir|veklury",
                    "molnupiravir": "molnupiravir|merck's drug|merck's pill|merck's antiviral"}

    for folder in keywords_dict.keys():
        os.makedirs(out_dir+folder, exist_ok=True)

    files = get_raw_data(data_path)
    if not files:
        print("No input file found.")
        return

    print("Extracting...")
    with open("tweet_amount.log","w") as amount_log:
        for file in tqdm.tqdm(files[34:]):
            print("On file "+file)
            df = pd.read_csv(file, lineterminator='\n', low_memory=False).replace(np.nan, "")
            df = df[['created_at', 'id', 'full_text', 'user', 'coordinates', 'place','quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'geo']]
            df["screen_name"] = df.user.parallel_apply(lambda x: extract_screenname_from_user(x))  
            amount_log.write("%i tweets written to %s\n"%(len(df), altered_input_dir+file.split("/")[-1]))
            # df.to_csv(altered_input_dir+file.split("/")[-1], index=False)
            
            for drug, keywords in keywords_dict.items():
                drug_df = df[df["full_text"].str.contains(keywords, case=False)].drop_duplicates()
                # other_keys = keywords_dict.copy()
                # other_keys.pop(drug)
                # keys_for_other_drugs = "|".join(list(other_keys.values()))
                # drug_df = drug_df[~drug_df.full_text.str.contains(keys_for_other_drugs, case=False)]

                file_num = file.split("/")[-1]               
                drug_filename = f"{out_dir}{drug}/{file_num}"
                if not drug_df.empty:
                    drug_df.to_csv(drug_filename, index=False)
                message = "%i tweets written to %s\n"%(len(drug_df), drug_filename)
                print(message)
                amount_log.write(message)
    
    print("Merging...")
    for drug in keywords_dict.keys():
        print("On %s"%drug)
        l = [pd.read_csv(f, lineterminator="\n") for f in glob.glob(f"{out_dir}{drug}/*.csv")]
        df = pd.concat(l)
        df = df[df.full_text!=np.nan].drop_duplicates(subset=["id"])
        print("%i total tweets found"%len(df))
        df.to_csv(f"{out_dir}{drug}_all.csv",index=False)

main()
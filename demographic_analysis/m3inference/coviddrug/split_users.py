"""
Author: hjian42@icloud.com

This script splits users into two parts: users with both texts and images and users with only text descriptions
    because the former is fed into the regular M3 module 
    and the latter users will use a M3 model variant that does not require iamges
"""
import pandas as pd
import os, json
import glob

df = pd.read_csv("./all_users_info.csv")
print(df.shape)
df.head(10)

from os import path
from tqdm import tqdm
import json

with open('users_with_images.jsonl', 'w') as outfile, open('users_only_texts.jsonl', 'w') as outfile2:
    for row in tqdm(df.iterrows()):
        row = row[1]
        # img_path = "./coviddrug/pic_resize_400x400/{}.jpg".format(row.screen_name)
        if row.screen_name in users_with_images:
            jsonobj = {"id": str(row.user_id),
                    "name": row.name,
                    "screen_name": row.screen_name, 
                    "description": row.description, 
                    "lang": "en",
                    "img_path": users_with_images[row.screen_name]}
            json.dump(jsonobj, outfile)
            outfile.write('\n')
        else:
            jsonobj = {"id": str(row.user_id),
                    "name": row.name,
                    "screen_name": row.screen_name, 
                    "description": row.description, 
                    "lang": "en"}
            json.dump(jsonobj, outfile2)
            outfile2.write('\n')

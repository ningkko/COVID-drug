"""
Author: hjian42@icloud.com

This script converts user json file to csv format
"""
import json
import pandas as pd
json_list = []
with open("./all_users_info.jsonl") as f:
    for line in f:
        json_list.append(json.loads(line))

df_2022 = pd.DataFrame([[d['id'], d['name'], d['screen_name'], 
                         d['description'].replace('\n',' ').replace('\r',' ').replace('\t',' '), 
                         d['img_url']] for d in json_list],
                       columns=['user_id', 'name', 'screen_name', 'description', 'profile_image_url'])
df_2022.to_csv("all_users_info.csv", index=False, sep="\t")
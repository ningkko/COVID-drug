"""
Author: hjian42@icloud.com

This script crawls the user information given a file containing user ids
"""
import time
import tweepy
import csv
import os
from requests.exceptions import Timeout, ConnectionError
import tweepy
import pandas as pd
from tqdm import tqdm
import json

# read data 
with open("all_users.txt") as f:
    user_ids = [line.strip() for line in f]
# print(user_ids)

# Enter your own Consumer Key, Consumer Secret, Access Token, and Access Token Secret here.
auth = tweepy.OAuthHandler('xxx', 'xxx')
api = tweepy.API(auth, retry_count=3, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=2000, wait_on_rate_limit=True)

i = 0
total = len(user_ids)
with open('all_users_info.jsonl', 'w') as outfile:
    while i < (total / 100) + 1:
        try:
            user_jsons = api.lookup_users(user_id=user_ids[i*100:min((i+1)*100, total)])
            print(i, len(user_jsons))
            for jsonobj in user_jsons:
                # print(jsonobj.id)
                try:
                    jsonobj = {"id": jsonobj.id_str,
                                "name": jsonobj.name,
                                "screen_name": jsonobj.screen_name, 
                                "description": jsonobj.description, 
                                "lang": "en",
                                "img_url": jsonobj.profile_image_url_https} 
                    json.dump(jsonobj, outfile)
                    outfile.write('\n')
                except NameError:
                    print(NameError) 
            i += 1
        except NameError:
            print(NameError) 

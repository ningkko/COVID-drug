"""
Author: hjian42@icloud.com

This script crawls the friends (people that someone follows) of each user and saves them inside ./friends_info/
"""
import time
import tweepy
import csv
import os
from requests.exceptions import Timeout, ConnectionError
import tweepy
import pandas as pd
from tqdm import tqdm

# read data 
df = pd.read_csv("all_users.csv")
user_ids = df.user_id.values

# Enter your own Consumer Key, Consumer Secret, Access Token, and Access Token Secret here.
auth = tweepy.OAuthHandler('xxx', 'xxx')
api = tweepy.API(auth, retry_count=3, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=2000, wait_on_rate_limit=True)

for user_id in tqdm(user_ids):
    try:
        friends = api.get_friend_ids(user_id=user_id)
        friend_line = ",".join([str(friend) for friend in friends])
        line = "\t".join([str(user_id), friend_line])
        with open("./friends_info/{}.txt".format(user_id), "w") as out:
            out.write(line)
    except Exception as err:
        print("user_id", user_id, "error", err)

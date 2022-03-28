#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hang Jiang
"""
import pandas as pd
from litecoder.usa import USCityIndex, USStateIndex
idx_city = USCityIndex()
idx_state = USStateIndex()

idx_city.load()
idx_state.load()

def extract_city_state(L):
    """
    returns city, state, country
    """
    if L:
        L = str(L)
    
        if L.lower() in set(["united states", "the united states", "u.s.", "us", "usa", "u.s.a", "u.s.a."]):
            return "", "", "US"

        r = idx_city[L]
        if r:
            return r[0]['name'], r[0]['name_a1'], r[0]['country_iso']
        r = idx_state[L]
        if r:
            return "", r[0]['name'], r[0]['country_iso']
    
    return "", "", ""


def extract_location_from_user(user):
    user = ast.literal_eval(user) if user==user else ""
    location = user['location'] if "location" in user else ""
    return extract_city_state(location)


from glob import glob
import ast
from pandarallel import pandarallel

pandarallel.initialize()

for filename in glob("../../data/*_all.csv"):
    print(filename)
    df = pd.read_csv(filename)
    df['triple_loc'] = df.location.parallel_apply(extract_city_state)
    df['city'] = df.triple_loc.parallel_apply(lambda x: x[0])
    df['state'] = df.triple_loc.parallel_apply(lambda x: x[1])
    df['country'] = df.triple_loc.parallel_apply(lambda x: x[2])
    df = df.drop(["triple_loc", 'Unnamed: 0'], axis=1)
    df = df[df.state == ""]
#     print(df)
    out_filename = "output/2/{}".format(filename[2:])
    df.to_csv(out_filename, index=False)
#     break

from glob import glob
import ast

for filename in glob("output/2/*.csv"):
    print(filename)
    df = pd.read_csv(filename)
#     df = df.head(1000)
    df['triple_loc'] = df.user.apply(extract_location_from_user)
#     df['triple_loc'] = df.user.apply(lambda x: extract_city_state(ast.literal_eval(x)['location']))
    df['city'] = df.triple_loc.apply(lambda x: x[0])
    df['state'] = df.triple_loc.apply(lambda x: x[1])
    df['country'] = df.triple_loc.apply(lambda x: x[2])
    df = df.drop(["triple_loc"], axis=1)
    df.to_csv("output/2/{}".format(filename[2:]))
#     break
# df_12months['triple_loc'] = df_12months.location.apply(extract_city_state)
#     break

for filename in glob("output/2/*.csv"):
    print(filename)
    df = pd.read_csv(filename)
    remain_num, total_num = df[df.country == "US"].shape[0], df.shape[0]
    print("before de-dup:", remain_num, total_num, remain_num/total_num)
    df['name_tweet'] = df.screen_name + df.full_text
    df = df.drop_duplicates(subset='name_tweet', keep='first')
    remain_num, total_num = df[df.country == "US"].shape[0], df.shape[0]
    print("after de-dup:", remain_num, total_num, remain_num/total_num)


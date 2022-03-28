#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Minghui Li
"""
import pandas as pd
import numpy as np
import re
import tqdm
import os
input_dir = "../../data/"
output_dir = "output/1/"
os.makedirs(output_dir, exist_ok=True)

def main(drug):
    print("On %s..."%drug)
    save_path = f"{output_dir}{drug}.csv"
    data = pd.read_csv(f"{input_dir}{drug}_all.csv", low_memory=False, lineterminator="\n")
    data['is_place'] = data['place'].str.contains("country_code")
    I = range(data.index.size)
    location = []
    data['is_user'] = data['user'].str.contains("location.*description")

    print("Extracting location...")
    for i in tqdm.tqdm(I):
        try:
            if data["is_place"][i] == True:
                a = data['place'][i]
                b = eval(a)
                location.append(b.get('full_name'))
            elif data['is_user'][i] == True:
                a = data['user'][i]
                b = eval(a)
                location.append(b.get('location'))
            else:
                location.append(" ")
        except SyntaxError:
            pass
        continue
    data['location'] = location

    state = "Alaska| AK|Alabama| AL|Arkansas| AR|Arizona| AZ|California| CA|Colorado| CO|Connecticut| CT|Delaware| DE|Florida| FL|Georgia| GA|Hawaii| HI|Iowa| IA|Idaho| ID|Illinois| IL|Indiana| IN|Kansas| KS|Kentucky| KY|Louisiana| LA| ME|Maine| MD|Maryland|Massachusetts| MA|Michigan| MI|Minnesota| MN|Missouri| MO|Mississippi| MS|Montana| MT|North Carolina| NC|North Dakota| ND|Nebraska| NE|New Hampshire| NH|New Jersey| NJ|New Mexico| NM|Nevada| NY|New York| NY|Ohio| OH|Oklahoma| OK|Oregon| OR|Pennsylvania| PA|Rhode Island| RL|South Carolina| SC|South Dakota| SD|Tennessee| TN|Texas| TX|Utah| UT|Virginia| VA|Vermont| VT|Washington| WA|Wisconsin| WI|West Virginia| WVWyoming| WY"
    data['state'] = "nan"
    print("Mapping location...")
    s = []
    for m in data.index:
        a = data.loc[m,'location']
        b = re.findall(state, a)
        c = '_'.join(b)
        s.append(c.strip())
    data['state'] = s
    data['state'].replace('', np.nan, inplace=True)
    data = data[-data['state'].isna()]
    data = data.drop(['is_user'], axis=1).drop_duplicates()
    print("%i locations extracted."%len(data))
    
    print("Writing to %s"%save_path)
    data.to_csv(save_path, index=False)

drugs = ["hcq", "ivermectin", "remdesivir", "molnupiravir"]

for drug in drugs:
    main(drug)
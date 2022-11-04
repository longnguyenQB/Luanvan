from datetime import timedelta , datetime
import json
import pandas as pd
import itertools
from collections import Counter
import unidecode
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from typing import Dict, List
import unidecode

def remove_accent(sentence):
    return unidecode.unidecode(sentence)

def filter_by_days(data, column_name: str, days: int):
    range_time = data[column_name].max()- data[column_name]
    data = data.loc[(timedelta(days=0) <= range_time) & (range_time <= timedelta(days=days))]
    return data

def filter_by_startday(data, column_name ,start):
    data = data.loc[data[column_name] >= start]
    return data

def remove_prefix(input_string):
    """Xóa ký tự + và 84 ở đầu

    Args:
        input_string (_type_): string
    """    
    def remove_prefix1(input_string):
        prefix = '+'
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string
    def remove_prefix2(input_string):
        prefix = '84'
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string
    input_string = remove_prefix1(input_string)
    input_string = remove_prefix2(input_string)
    return input_string 

def spam_counts_report(df):
    try:
        df['spam'] = df.type.value_counts().spam
    except:
        df['spam'] = 0
    try:
        df['ham'] = df.type.value_counts().ham
    except:
        df['ham'] = 0
    df['num_member_phone'] = len(df.member_phone.unique())
    return df

def spam_counts_predict(df):
    try:
        df['spam_predict'] = df.status.value_counts()['1']
    except:
        df['spam_predict'] = 0
    try:
        df['ham_predict'] = df.status.value_counts()['0']
    except:
        df['ham_predict'] = 0
    return df
def replace_type(df):
    df = df.replace({'type': {'1': 'ham', '2': 'spam', '3': 'spam', '4': 'spam', '5': 'spam', '6':'spam', '7':'spam', '8':'ham', '9': 'ham', '10': 'ham'}})
    return df

def js2df(link: str):
    f = open(link, encoding="utf-8")
    js = json.load(f)
    return pd.DataFrame(js)
def df2js(df, path_save, orient):
    df = df.to_json(orient=orient)
    with open(path_save, 'w', encoding="utf-8") as f:
        json.dump(df, f)
    return "Save done!"
    
def sort_by_another_list(df, column_name, sorter):
    sorterIndex = dict(zip(sorter, range(len(sorter))))
    df['Tm_Rank'] = df[column_name].map(sorterIndex)
    df.sort_values(['Tm_Rank'], ascending = [True], inplace = True)
    df.drop('Tm_Rank', 1, inplace = True)
    df.reset_index(inplace=True, drop=True)
    return df
def convert_time(time):
    return datetime.fromisoformat(str(time))

def zero_div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0
    
def plot_name(df, num, type):
    list_word = df.name.dropna().unique().flatten()
    if type == 'remove':
        list_word = [remove_accent(i) for i in list_word]
        list_word = [i.split() for i in list_word]
        list_word = list(itertools.chain.from_iterable(list_word))
        begin = 0
    elif type == 'notremove':
        list_word = [i.split() for i in list_word]
        list_word = list(itertools.chain.from_iterable(list_word))
        begin = 0
    elif type == 'notsplit':
        list_word = [remove_accent(i) for i in list_word]
        begin = 10
    name = np.array(sorted(Counter(list_word).items(), key=lambda x: x[1]))
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xticklabels(name[-num + begin:][:,0], rotation=90, ha='right', fontsize = 10)
    ax.bar(name[-num + begin:][:,0], name[-num + begin:][:,1])
    return name[-num + begin:][:,0]

def list_contains(List1, List2): 
  
    set1 = set(List1) 
    set2 = set(List2) 
    if set1.intersection(set2): 
        return True 
    else: 
        return False
    
def check_name(name, name_vn, name_dx, name_job, name_confused, name_spam):
    def check_sentence(name, list_name):
        for i in list_name:
            if len(i.split()) > 1:
                if i in name:
                    return True
            else:
                if list_contains([i], name.split()):
                    return True
    if name.startswith(tuple(name_dx)) | any(ext in name for ext in name_job):
        return 'notspam_name'
    elif set([name]).intersection(set(name_confused)):
        return "confused_name"
    elif check_sentence(name, name_spam):
        return "spam_name"
    elif any(ext in name for ext in name_vn):
        return "notspam_name"
    else:
        return "unknow_name"
def normalize_str(string):
    return normalize("NFKC", string.__str__())
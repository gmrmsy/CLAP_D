import pandas as pd
import numpy as np

def index_extractor(no=None):
    df = pd.read_csv("/home/inter/CLAP_D/data/csv/d_talk_clean_.csv")

    if no == None:
        temp_index_1 = df.loc[df['Target'] == 1].index
        temp_index_2 = df.loc[df['Target'] != 1].index
    elif isinstance(no,float) or isinstance(no,int):
        temp_index_1 = df.loc[(df['QUESTION_NO']==no) & (df['Target'] == 1)].index
        temp_index_2 = df.loc[(df['QUESTION_NO']==no) & (df['Target'] != 1)].index
        
    return temp_index_1, temp_index_2

def train_valid_test_ratio(npy, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    npy_len = len(npy)
    ran_np = np.arange(npy_len)
    np.random.shuffle(ran_np)
    train_ratio_num = npy_len*train_ratio
    valid_ratio_num = int(npy_len*valid_ratio)+1
    test_ratio_num = int(npy_len*test_ratio)+1
    
    train_npy = npy[test_ratio_num+valid_ratio_num:]
    valid_npy = npy[test_ratio_num:test_ratio_num+valid_ratio_num]
    test_npy = npy[:test_ratio_num]

    return train_npy, valid_npy, test_npy


def train_valid_test_split(npy, num=None, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    is1idx, no1idx = index_extractor(no=num)

    is1_train_npy, is1_valid_npy, is1_test_npy = train_valid_test_ratio(npy[is1idx], train_ratio, valid_ratio, test_ratio)
    no1_train_npy, no1_valid_npy, no1_test_npy = train_valid_test_ratio(npy[no1idx], train_ratio, valid_ratio, test_ratio)

    temp_train_npy = np.concatenate((is1_train_npy,no1_train_npy))
    temp_valid_npy = np.concatenate((is1_valid_npy,no1_valid_npy))
    temp_test_npy = np.concatenate((is1_test_npy,no1_test_npy))

    np.random.shuffle(temp_train_npy)
    np.random.shuffle(temp_valid_npy)
    np.random.shuffle(temp_test_npy)

    return temp_train_npy, temp_valid_npy, temp_test_npy
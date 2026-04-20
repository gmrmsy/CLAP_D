from scr.data.split import train_valid_test_split
from scr.utils.augment_utils import make_aug_dataset_pitch_speed
import tensorflow as tf
import pandas as pd
import numpy as np


class data_load():
    def __init__(self):
        self.x1_data_npy = np.load("/home/inter/CLAP_D/data/npy/x1_data.npy")
        self.x1_data_length_npy = np.load("/home/inter/CLAP_D/data/npy/x1_data_length.npy")
        self.x2_data_npy = np.load("/home/inter/CLAP_D/data/npy/x2_data.npy")
        self.x2_data_length_npy = np.load("/home/inter/CLAP_D/data/npy/x2_data_length.npy")
        self.y_data_npy = np.load("/home/inter/CLAP_D/data/npy/y_data.npy")

        self.x1_train_list, self.x1_valid_list, self.x1_test_list = [], [], []
        self.x1_length_train_list, self.x1_length_valid_list, self.x1_length_test_list = [], [], []
        self.x2_train_list, self.x2_valid_list, self.x2_test_list = [], [], []
        self.x2_length_train_list, self.x2_length_valid_list, self.x2_length_test_list = [], [], []
        self.y_train_list, self.y_valid_list, self.y_test_list = [], [], []
    
    def make_list(self, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        for i in range(1,26,1):
            temp_x1_train, temp_x1_valid, temp_x1_test = train_valid_test_split(self.x1_data_npy, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, num=i)
            temp_x1_length_train, temp_x1_length_valid, temp_x1_length_test = train_valid_test_split(self.x1_data_length_npy, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, num=i)
            temp_x2_train, temp_x2_valid, temp_x2_test = train_valid_test_split(self.x2_data_npy, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, num=i)
            temp_x2_length_train, temp_x2_length_valid, temp_x2_length_test = train_valid_test_split(self.x2_data_length_npy, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, num=i)
            temp_y_train, temp_y_valid, temp_y_test = train_valid_test_split(self.y_data_npy, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, num=i)

            self.x1_train_list.append(temp_x1_train), self.x1_valid_list.append(temp_x1_valid), self.x1_test_list.append(temp_x1_test)
            self.x1_length_train_list.append(temp_x1_length_train), self.x1_length_valid_list.append(temp_x1_length_valid), self.x1_length_test_list.append(temp_x1_length_test)
            self.x2_train_list.append(temp_x2_train), self.x2_valid_list.append(temp_x2_valid), self.x2_test_list.append(temp_x2_test)
            self.x2_length_train_list.append(temp_x2_length_train), self.x2_length_valid_list.append(temp_x2_length_valid), self.x2_length_test_list.append(temp_x2_length_test)
            self.y_train_list.append(temp_y_train), self.y_valid_list.append(temp_y_valid), self.y_test_list.append(temp_y_test)

        return (self.x1_train_list, self.x1_valid_list, self.x1_test_list,
                self.x1_length_train_list, self.x1_length_valid_list, self.x1_length_test_list,
                self.x2_train_list, self.x2_valid_list, self.x2_test_list,
                self.x2_length_train_list, self.x2_length_valid_list, self.x2_length_test_list,
                self.y_train_list, self.y_valid_list, self.y_test_list
        )


    def total_concat(self, data):
        (x1_train_list, x1_valid_list, x1_test_list,
         x1_length_train_list, x1_length_valid_list, x1_length_test_list,
         x2_train_list, x2_valid_list, x2_test_list,
         x2_length_train_list, x2_length_valid_list, x2_length_test_list,
         y_train_list, y_valid_list, y_test_list) = data

        x1_train, x1_valid, x1_test = np.concatenate((x1_train_list)), np.concatenate((x1_valid_list)), np.concatenate((x1_test_list))
        x1_length_train, x1_length_valid, x1_length_test = np.concatenate((x1_length_train_list)), np.concatenate((x1_length_valid_list)), np.concatenate((x1_length_test_list))
        x2_train, x2_valid, x2_test = np.concatenate((x2_train_list)), np.concatenate((x2_valid_list)), np.concatenate((x2_test_list))
        x2_length_train, x2_length_valid, x2_length_test = np.concatenate((x2_length_train_list)), np.concatenate((x2_length_valid_list)), np.concatenate((x2_length_test_list))
        y_train, y_valid, y_test = np.concatenate((y_train_list)), np.concatenate((y_valid_list)), np.concatenate((y_test_list))

        return (x1_train, x1_valid, x1_test,
                x1_length_train, x1_length_valid, x1_length_test,
                x2_train, x2_valid, x2_test,
                x2_length_train, x2_length_valid, x2_length_test,
                y_train, y_valid, y_test
        )


    def partial_concat(self, data, part_list):
        (x1_train_list, x1_valid_list, x1_test_list,
         x1_length_train_list, x1_length_valid_list, x1_length_test_list,
         x2_train_list, x2_valid_list, x2_test_list,
         x2_length_train_list, x2_length_valid_list, x2_length_test_list,
         y_train_list, y_valid_list, y_test_list) = data
        
        part_x1_train_list = [x1_train_list[i-1] for i in part_list]
        part_x1_valid_list = [x1_valid_list[i-1] for i in part_list]
        part_x1_test_list = [x1_test_list[i-1] for i in part_list]
        
        part_x1_length_train_list = [x1_length_train_list[i-1] for i in part_list]
        part_x1_length_valid_list = [x1_length_valid_list[i-1] for i in part_list]
        part_x1_length_test_list = [x1_length_test_list[i-1] for i in part_list]        

        part_x2_train_list = [x2_train_list[i-1] for i in part_list]
        part_x2_valid_list = [x2_valid_list[i-1] for i in part_list]
        part_x2_test_list = [x2_test_list[i-1] for i in part_list]
        
        part_x2_length_train_list = [x2_length_train_list[i-1] for i in part_list]
        part_x2_length_valid_list = [x2_length_valid_list[i-1] for i in part_list]
        part_x2_length_test_list = [x2_length_test_list[i-1] for i in part_list] 

        part_y_train_list = [y_train_list[i-1] for i in part_list]
        part_y_valid_list = [y_valid_list[i-1] for i in part_list]
        part_y_test_list = [y_test_list[i-1] for i in part_list]

        x1_train, x1_valid, x1_test = np.concatenate((part_x1_train_list)), np.concatenate((part_x1_valid_list)), np.concatenate((part_x1_test_list))
        x1_length_train, x1_length_valid, x1_length_test = np.concatenate((part_x1_length_train_list)), np.concatenate((part_x1_length_valid_list)), np.concatenate((part_x1_length_test_list))
        x2_train, x2_valid, x2_test = np.concatenate((part_x2_train_list)), np.concatenate((part_x2_valid_list)), np.concatenate((part_x2_test_list))
        x2_length_train, x2_length_valid, x2_length_test = np.concatenate((part_x2_length_train_list)), np.concatenate((part_x2_length_valid_list)), np.concatenate((part_x2_length_test_list))
        y_train, y_valid, y_test = np.concatenate((part_y_train_list)), np.concatenate((part_y_valid_list)), np.concatenate((part_y_test_list))

        return (x1_train, x1_valid, x1_test,
                x1_length_train, x1_length_valid, x1_length_test,
                x2_train, x2_valid, x2_test,
                x2_length_train, x2_length_valid, x2_length_test,
                y_train, y_valid, y_test
        )        
        
    def augment(self, 
                x1, x1_len, x2, x2_len, y, num_aug,
                min_speed_1=0.875, min_speed_2=0.925,
                max_speed_1=1.1, max_speed_2=1.25,
                min_pitch_bins=5, max_pitch_bins=10,
                replace_val=-80.0,
                merge_with_original=True,
                return_lengths=True,
                seed=None
                ):
        
        idx_0 = np.where(y != 1)[0]
        idx_1 = np.where(y == 1)[0]

        temp_0_x1, temp_0_x1_len, temp_0_x2, temp_0_x2_len, temp_0_y = x1[idx_0], x1_len[idx_0], x2[idx_0], x2_len[idx_0], y[idx_0]
        temp_1_x1, temp_1_x1_len, temp_1_x2, temp_1_x2_len, temp_1_y = x1[idx_1], x1_len[idx_1], x2[idx_1], x2_len[idx_1], y[idx_1]

        if return_lengths == True:
            aug_x1, aug_x1_len, aug_x2, aug_x2_len, aug_y =  make_aug_dataset_pitch_speed(temp_0_x1, temp_0_x1_len, temp_0_x2, temp_0_x2_len, temp_0_y, num_aug=num_aug,
                                                                                          min_speed_1=min_speed_1, min_speed_2=min_speed_2,
                                                                                          max_speed_1=max_speed_1, max_speed_2=max_speed_2,
                                                                                          min_pitch_bins=min_pitch_bins, max_pitch_bins=max_pitch_bins,
                                                                                          replace_val=replace_val,
                                                                                          merge_with_original=merge_with_original,
                                                                                          return_lengths=return_lengths,
                                                                                          seed=seed)
            
            rng = np.random.default_rng(seed)
        
            x1_train = np.concatenate([temp_1_x1, aug_x1], axis=0)
            x1_len_train = np.concatenate([temp_1_x1_len, aug_x1_len], axis=0)
            x2_train = np.concatenate([temp_1_x2, aug_x2], axis=0)
            x2_len_train = np.concatenate([temp_1_x2_len, aug_x2_len], axis=0)
            y_train = np.concatenate([temp_1_y, aug_y], axis=0)

            perm = rng.permutation(x1_train.shape[0])
            x1_train = x1_train[perm]
            x1_len_train = x1_len_train[perm]
            x2_train = x2_train[perm]
            x2_len_train = x2_len_train[perm]
            y_train = y_train[perm]

            return x1_train, x1_len_train, x2_train, x2_len_train, y_train
        
        else:
            aug_x1, aug_x2, aug_y =  make_aug_dataset_pitch_speed(temp_0_x1, temp_0_x1_len, temp_0_x2, temp_0_x2_len, temp_0_y, num_aug=num_aug,
                                                                                          min_speed_1=min_speed_1, min_speed_2=min_speed_2,
                                                                                          max_speed_1=max_speed_1, max_speed_2=max_speed_2,
                                                                                          min_pitch_bins=min_pitch_bins, max_pitch_bins=max_pitch_bins,
                                                                                          replace_val=replace_val,
                                                                                          merge_with_original=merge_with_original,
                                                                                          return_lengths=return_lengths,
                                                                                          seed=seed)

            rng = np.random.default_rng(seed)
        
            x1_train = np.concatenate([temp_1_x1, aug_x1], axis=0)
            x2_train = np.concatenate([temp_1_x2, aug_x2], axis=0)
            y_train = np.concatenate([temp_1_y, aug_y], axis=0)

            perm = rng.permutation(x1_train.shape[0])
            x1_train = x1_train[perm]
            x2_train = x2_train[perm]
            y_train = y_train[perm]

            return x1_train, x2_train, y_train
        
    
    def select_reliable_data(self, data):
        d_2_csv = pd.read_csv("/home/inter/CLAP_D/data/csv/d_talk_clean_.csv")
        temp_list = d_2_csv.groupby('QUESTION_NO')['Target'].describe()[['std']].sort_values('std', ascending=False).index[:3]
        select_data = self.partial_concat(data, temp_list)

        return select_data
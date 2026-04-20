from scr.utils.jamo_utils import jamo_to_index, index_to_jamo, decompose_hangul, char_to_index, seq_padding, text_to_ctc_indices
from scr.utils.wav_utils import audio_preprocess, wav_padding, x_data_preprocess

import pandas as pd
import numpy as np

# D_2관련 CSV 가져오기
d_2_csv = pd.read_csv("/home/inter/CLAP_D/data/csv/d_talk_clean_.csv")

x1_data_list = []
x1_data_length_list = []
x2_data_list = []
x2_data_length_list = []
y_data_list = []

for row in d_2_csv.itertuples():
    temp_path = row.Path
    temp_note = row.Note
    temp_x1_data, temp_x1_data_length = x_data_preprocess(temp_path)
    temp_x2_data, temp_x2_data_length = text_to_ctc_indices(temp_note)
    temp_y_data = row.Target

    x1_data_list.append(temp_x1_data)
    x1_data_length_list.append(temp_x1_data_length)
    x2_data_list.append(temp_x2_data)
    x2_data_length_list.append(temp_x2_data_length)
    y_data_list.append(temp_y_data)

x1_data = np.array(x1_data_list)
x1_data_length = np.array(x1_data_length_list)
x2_data = np.array(x2_data_list)
x2_data_length = np.array(x2_data_length_list)
y_data = np.array(y_data_list)

print("===== shape 확인 =====")
print(f"x1_data.shape: {x1_data.shape}, x1_data_length.shape: {x1_data_length.shape}, x2_data.shape: {x2_data.shape}, x2_data_length.shape: {x2_data_length.shape}, y_data.shape: {y_data.shape}")

np.save("/home/inter/CLAP_D/data/npy/x1_data.npy",x1_data)
np.save("/home/inter/CLAP_D/data/npy/x1_data_length.npy",x1_data_length)
np.save("/home/inter/CLAP_D/data/npy/x2_data.npy",x2_data)
np.save("/home/inter/CLAP_D/data/npy/x2_data_length.npy",x2_data_length)
np.save("/home/inter/CLAP_D/data/npy/y_data.npy",y_data)
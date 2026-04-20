import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

from scr.models.model import hardtanh, SinusoidalPositionalEncoding, SequenceMask, MakeAttnMask
from scr.utils.data_utils import data_load
from glob import glob
import tensorflow as tf
import pandas as pd
import numpy as np

# 데이터 로드
data_loader = data_load()

x1_data_npy = data_loader.x1_data_npy
x1_data_length_npy = data_loader.x1_data_length_npy
x2_data_npy = data_loader.x2_data_npy
x2_data_length_npy = data_loader.x2_data_length_npy
y_data_npy = data_loader.y_data_npy

df = pd.read_csv("/home/inter/CLAP_D/data/csv/d_talk_clean_.csv")
correct_y = df['Score(Refer)'].to_numpy()
alloc_npy = df['Score(Alloc)'].to_numpy()


# evaluation을 위한 csv준비(전체 예측점수표, accuracy/corr 비교표)
idx_list = []
for i in range(1,26,1):
    idx_list.append(f'no{i}')
idx_list.append('total')
idx_list.append('reliable')

temp_list = glob('/home/inter/CLAP_D/checkpoints/*/')
temp_list.sort()

for path in temp_list:
    print(f"==================== {path} ====================")
    compare_df = pd.DataFrame(index=idx_list)
    score_df = pd.DataFrame()

    model_list = glob(f"{path}/*.keras")
    model_list.sort()

    for model_path in model_list:
        print(f"==================== {model_path} ====================")
        pred_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "hardtanh": hardtanh,
                "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
                "SequenceMask": SequenceMask,
                "MakeAttnMask": MakeAttnMask,
                }
            )

        pred_score = pred_model.predict([x1_data_npy, x2_data_npy]).reshape(-1)
        post_process = pred_score * alloc_npy
        process_score = np.round(post_process,0)

        name = model_path.split('.')[0].split('_D_')[1]

        score_df[f'{name}_pred'] = pred_score
        score_df[f'{name}_score'] = process_score
        
        accuracy = np.mean(process_score == correct_y)
        corr = np.corrcoef(process_score,correct_y)[0, 1]
        compare_df.loc[f'{name}','accuracy'] = accuracy
        compare_df.loc[f'{name}','corr'] = corr

        del(pred_model)
    
    score_df[f'y'] = correct_y
    score_df.to_csv(f"/home/inter/CLAP_D/data/csv/{path.split('/')[5]}_score_df.csv", index=False)
    compare_df.to_csv(f"/home/inter/CLAP_D/data/csv/{path.split('/')[5]}_compare_df.csv")


# # 모델별 예측값, accuracy, corr 추출
# for i in range(1,26,1):
#     print(f"==================== no{i} ====================")
#     pred_model = tf.keras.models.load_model(
#         f'/home/inter/CLAP_D/checkpoints/tf_CLAP_D_no{i}.keras',
#         custom_objects={
#             "hardtanh": hardtanh,
#             "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
#             "SequenceMask": SequenceMask,
#             "MakeAttnMask": MakeAttnMask,
#             }
#         )
#     pred_score = pred_model.predict([x1_data_npy, x2_data_npy]).reshape(-1)
#     post_process = pred_score * alloc_npy
#     process_score = np.round(post_process,0)

#     score_df[f'no{i}_pred'] = pred_score
#     score_df[f'no{i}_score'] = process_score

#     accuracy = np.mean(process_score == correct_y)
#     corr = np.corrcoef(process_score,correct_y)[0, 1]
#     compare_df.loc[f'no{i}','accuracy'] = accuracy
#     compare_df.loc[f'no{i}','corr'] = corr

#     del(pred_model)


# print(f"==================== total ====================")
# pred_model = tf.keras.models.load_model(
#     f'/home/inter/CLAP_D/checkpoints/tf_CLAP_D_total.keras',
#     custom_objects={
#         "hardtanh": hardtanh,
#         "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
#         "SequenceMask": SequenceMask,
#         "MakeAttnMask": MakeAttnMask,
#         }
#     )

# pred_score = pred_model.predict([x1_data_npy, x2_data_npy]).reshape(-1)
# post_process = pred_score * alloc_npy
# process_score = np.round(post_process,0)

# score_df[f'total_pred'] = pred_score
# score_df[f'total_score'] = process_score

# accuracy = np.mean(process_score == correct_y)
# corr = np.corrcoef(process_score,correct_y)[0, 1]
# compare_df.loc['total','accuracy'] = accuracy
# compare_df.loc['total','corr'] = corr

# score_df[f'y'] = correct_y

# score_df.to_csv("/home/inter/CLAP_D/data/csv/score_df.csv", index=False)
# compare_df.to_csv("/home/inter/CLAP_D/data/csv/compare_df.csv")
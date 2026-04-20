import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

from scr.utils.data_utils import data_load
from scr.models.model import make_talk_clean_model
from scr.utils.train_utils import model_train

from itertools import product
import pandas as pd

data_loader = data_load()
d2_data = data_loader.make_list()

data_dict = {}
idx_list = []
for i in range(1,26,1):
    idx_list.append(f'no{i}')

for idx, i in enumerate(zip(*d2_data)):
    data_dict[idx_list[idx]] = i

data_dict['total'] = data_loader.total_concat(d2_data)
data_dict['reliable'] = data_loader.select_reliable_data(d2_data)

# 데이터(type, aug)
data_key_list = list(data_dict.keys())
# aug_list = [False, True]
aug_list = [True]

# 모델(dimension, out)
dimension_list = [1, 2]
# out_list = ['relu','linear']
out_list = ['linear']

# 학습(loss_weight)
loss_weight_list = [False, True]

comb_list = list(product(dimension_list, out_list, loss_weight_list, aug_list, data_key_list))
dimension, out, loss_weight, aug, data_key = comb_list[0]

cnt = 0
for comb in comb_list:
    dimension, out, loss_weight, aug, data_key = comb
    save_path = f"/home/inter/CLAP_D/checkpoints/{dimension}D_{out}_loss{'O' if loss_weight else 'X'}"

    print(f'{"="*10} cnt: {cnt} / {save_path}/CLAP_D_{data_key}{"_aug" if aug else ""} {"="*10}')

    if aug:
        (x1_train, x1_valid, x1_test,
        x1_length_train, x1_length_valid, x1_length_test,
        x2_train, x2_valid, x2_test,
        x2_length_train, x2_length_valid, x2_length_test,
        y_train, y_valid, y_test) = data_dict[data_key]

        aug_x1_train, aug_x1_length_train, aug_x2_train, aug_x2_length_train, aug_y_train = data_loader.augment(x1_train, x1_length_train,
                                                                    x2_train, x2_length_train,
                                                                    y_train, num_aug=(len(x1_train)*2))

        aug_data = (aug_x1_train, x1_valid, x1_test,
                    aug_x1_length_train, x1_length_valid, x1_length_test,
                    aug_x2_train, x2_valid, x2_test,
                    aug_x2_length_train, x2_length_valid, x2_length_test,
                    aug_y_train, y_valid, y_test)        

        # aug_x1_valid, aug_x1_length_valid, aug_x2_valid, aug_x2_length_valid, aug_y_valid = data_loader.augment(x1_train, x1_length_train,
        #                                                             x2_train, x2_length_train,
        #                                                             y_train, num_aug=(len(aug_x1_valid)*2))
        # aug_data = (aug_x1_train, aug_x1_valid, x1_test,
        #             aug_x1_length_train, aug_x1_length_valid, x1_length_test,
        #             aug_x2_train, aug_x2_valid, x2_test,
        #             aug_x2_length_train, aug_x2_length_valid, x2_length_test,
        #             aug_y_train, aug_y_valid, y_test)

        model = make_talk_clean_model(model_name=f"{data_key}{'_aug' if aug else ''}", dimension= dimension, out=out)
        model_history = model_train(aug_data, model, save_path=save_path, save_name=f"{data_key}{'_train_aug' if aug else ''}", loss_weight=loss_weight)
       
    else:
        model = make_talk_clean_model(model_name=f"{data_key}", dimension= dimension, out=out)
        model_history = model_train(data_dict[data_key], model, save_path=save_path, save_name=f"{data_key}", loss_weight=loss_weight)

    hist_df = pd.DataFrame(model_history.history)
    hist_df.to_csv(save_path+f"/CLAP_D_{data_key}{'_train_aug' if aug else ''}_history.csv", index=False)

    cnt += 1
    del(model)
    
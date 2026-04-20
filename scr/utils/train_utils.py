from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

def model_train(data, model, save_name='',
                loss_weight = False,
                save_path = "/home/inter/CLAP_D/checkpoints"
                ):

    (x1_train, x1_valid, x1_test,
    x1_length_train, x1_length_valid, x1_length_test,
    x2_train, x2_valid, x2_test,
    x2_length_train, x2_length_valid, x2_length_test,
    y_train, y_valid, y_test) = data

    def weight_return(y, loss_weight=loss_weight):

        if loss_weight is True:
            temp_weight = {}
            temp = np.unique(y, return_counts=True)
            ratios, counts = temp[0], temp[1]
            total = counts.sum()

            for i in range(len(ratios)):
               temp_weight[ratios[i]] = np.sqrt(total/(len(ratios)*counts[i]))
            
        else:
            temp_weight = {}
            temp = np.unique(y, return_counts=True)
            ratios, counts = temp[0], temp[1]
            total = counts.sum()

            for i in range(len(ratios)):
                temp_weight[ratios[i]] = 1
        
        sample_weight = np.array([temp_weight[val] for val in y], dtype=np.float32)

        return sample_weight
    
    train_weight = weight_return(y_train, loss_weight=loss_weight)
    valid_weight = weight_return(y_valid, loss_weight=loss_weight)

    trained_model = model.fit(x=[x1_train,x2_train],
            y=y_train,
            batch_size=64,
            validation_data=([x1_valid,x2_valid], y_valid, valid_weight),
            sample_weight=train_weight,
            epochs=100,
            callbacks=[EarlyStopping(monitor='val_loss',patience=10),
                        ModelCheckpoint(filepath=save_path+f'/CLAP_D_{save_name}.keras', monitor='val_loss', save_best_only=True, verbose=3)]
            )

    return trained_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


@tf.keras.utils.register_keras_serializable(package="custom")
def hardtanh(x, min_val=-20.0, max_val=20.0):
    return tf.clip_by_value(x, min_val, max_val)

@tf.keras.utils.register_keras_serializable(package="P.E")
class SinusoidalPositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.supports_masking = True   # 추가

        # (1, max_len, d_model) 고정 위치 인코딩 미리 계산
        pos = np.arange(max_len)[:, np.newaxis]         # (max_len, 1)
        i = np.arange(d_model)[np.newaxis, :]           # (1, d_model)

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates                      # (max_len, d_model)

        pe = np.zeros_like(angles)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])

        self.pos_encoding = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)  # (1, max_len, d_model)

    def call(self, x):
        # x: (B, T, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def compute_mask(self, inputs, mask=None):
        return mask

@tf.keras.utils.register_keras_serializable(package="mask")
class SequenceMask(layers.Layer):
    def call(self, inputs):
        is_padding = tf.reduce_all(tf.equal(inputs, -80.0), axis=[1,3])
        is_valid = tf.logical_not(is_padding)
        lengths = tf.reduce_sum(tf.cast(is_valid, tf.int32), axis=-1)
        lengths = tf.cast(tf.math.ceil(tf.cast(lengths, tf.float32) / 2.0), tf.int32)
        return tf.sequence_mask(lengths, maxlen=156)

@tf.keras.utils.register_keras_serializable(package="atten_mask")
class MakeAttnMask(layers.Layer):
    def call(self, inputs):
        qmask, kmask = inputs
        qmask = tf.expand_dims(qmask, 2)  # (b, 156, 1)
        kmask = tf.expand_dims(kmask, 1)  # (b, 1,   12)
        return tf.logical_and(qmask, kmask)  # (b, 156, 12)

def make_talk_clean_model(input_shape=(128,312), model_name=None, learning_rate=0.001,
                           dimension=1, out='relu'):

    x1_input = tf.keras.Input(shape=input_shape, name="x1_input")
    x2_input = tf.keras.Input(shape=(12,), dtype='int32', name="x2_input")

    x1_4d = layers.Reshape((*input_shape, 1))(x1_input)
    x1_data_mask = SequenceMask()(x1_4d)

    # 1. 음성데이터 -> CNN -> GRU

    if dimension == 1:
        conv_1 = layers.Conv1D(512, kernel_size=11, strides=2, padding='same', use_bias=False)
        conv_2 = layers.Conv1D(512, kernel_size=11, strides=1, padding='same', use_bias=False)

        Reshape_1 = layers.Permute((2,1,3))
        Reshape_1_out = Reshape_1(x1_4d)
        b, t, f, c = Reshape_1_out.shape
        Reshape_2 = layers.Reshape((t, -1))
        x1_cnn_input = Reshape_2(Reshape_1_out)

    elif dimension == 2:
        conv_1 = layers.Conv2D(32, kernel_size=(41, 11), strides=(2, 2), padding='same', use_bias=False)
        conv_2 = layers.Conv2D(32, kernel_size=(21, 11), strides=(2, 1), padding='same', use_bias=False)
        x1_cnn_input = x1_4d

    conv_1_out = conv_1(x1_cnn_input)
    BatNor_1 = layers.BatchNormalization()
    BatNor_1_out = BatNor_1(conv_1_out)
    hardtanh_1 = layers.Activation(hardtanh)
    hardtanh_1_out = hardtanh_1(BatNor_1_out)

    conv_2_out = conv_2(hardtanh_1_out)
    BatNor_2 = layers.BatchNormalization()
    BatNor_2_out = BatNor_2(conv_2_out)
    hardtanh_2 = layers.Activation(hardtanh)
    hardtanh_2_out = hardtanh_2(BatNor_2_out)

    if dimension == 1:
        pass
    elif dimension == 2:
        Reshape_1 = layers.Permute((2,1,3))
        Reshape_1_out = Reshape_1(hardtanh_2_out)
        b, t, f, c = Reshape_1_out.shape
        Reshape_2 = layers.Reshape((t, -1))
        hardtanh_2_out = Reshape_2(Reshape_1_out)        

    LayNor_1 = layers.LayerNormalization()
    LayNor_1_out = LayNor_1(hardtanh_2_out)

    GRU_1 = layers.Bidirectional(layers.GRU(64, dropout=0.1, return_sequences=True), merge_mode='concat')
    GRU_1_out = GRU_1(LayNor_1_out, mask=x1_data_mask)


    # 2. Embedding
    Embed_1 = layers.Embedding(input_dim=55, output_dim=16, mask_zero=True)
    Embed_1_out = Embed_1(x2_input)
    key_mask  = Embed_1.compute_mask(x2_input)

    attn_mask = MakeAttnMask()([x1_data_mask, key_mask])

    cycle = SinusoidalPositionalEncoding(max_len=156, d_model=128)
    cycle_out = cycle(GRU_1_out)

    # 3. Attention
    for i in range(6):
        MHA_1 = layers.MultiHeadAttention(num_heads=16, key_dim=32, dropout=0.1)
        MHA_1_out = MHA_1(cycle_out, Embed_1_out, attention_mask=attn_mask)

        SUM_1 = tf.keras.layers.add([cycle_out, MHA_1_out])
        LayNor_2 = layers.LayerNormalization()
        LayNor_2_out = LayNor_2(SUM_1)

        Dense_01 = tf.keras.layers.Dense(512, activation='relu')
        Dense_01_out = Dense_01(LayNor_2_out)

        Dense_02 = tf.keras.layers.Dense(128)
        Dense_02_out = Dense_02(Dense_01_out)

        SUM_2 = tf.keras.layers.add([LayNor_2_out, Dense_02_out])
        LayNor_3 = layers.LayerNormalization()
        cycle_out = LayNor_3(SUM_2)

    GloAvg_1 = layers.GlobalAveragePooling1D()
    GloAvg_1_out = GloAvg_1(cycle_out, mask=x1_data_mask)
    GloMax_1 = layers.GlobalMaxPooling1D()
    GloMax_1_out = GloMax_1(cycle_out)
    pooled = layers.Concatenate()([GloAvg_1_out, GloMax_1_out])

    # 4. Dense
    Dense_2 = layers.Dense(512,activation='relu')
    Dense_2_out = Dense_2(pooled)

    Dense_3 = layers.Dense(1,activation=out)
    Dense_3_out = Dense_3(Dense_2_out)

    if model_name == None:
        model = Model(inputs=[x1_input,x2_input], outputs=Dense_3_out)
    else:
        model = Model(inputs=[x1_input,x2_input], outputs=Dense_3_out, name=f'{model_name}')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    return model
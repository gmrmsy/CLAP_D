import numpy as np
import librosa


# 원본데이터 -> Mel_Spectrogram 변환
def audio_preprocess(wav, sr=8000, n_mels=128):
  """input: 음성데이터 경로(str)
     output: Mel_spectrogram 변환데이터(np.array), Mel_spectrogram 길이(int)"""
  ori_y1, sr1 = librosa.load(wav, sr=sr)
  mel_spec1 = librosa.feature.melspectrogram(y=ori_y1, sr=sr1, n_mels=n_mels)
  mel_db1 = librosa.power_to_db(mel_spec1, ref=np.max)
  return mel_db1, mel_db1.shape[1]


# 음성데이터 padding처리
def wav_padding(wav, wav_max_len=312):
  """input: wav 데이터(np.array)
     output: padding된 wav 데이터(np.arrray)"""
  pad_width = wav_max_len - wav.shape[1]  # 얼마나 채워야 하는지
  if pad_width > 0:
      # 오른쪽(열 끝)에 0을 채움: ((행 시작, 행 끝), (열 시작, 열 끝))
      padded = np.pad(wav, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-80)
  elif pad_width == 0:
      padded = wav  # 이미 가장 김
  elif pad_width < 0:
      padded = wav[:, :wav_max_len]
  return padded


# 음성데이터 전처리
def x_data_preprocess(x, sr=8000, n_mels=128):
  """음성데이터 -> Mel_Spectrogram, npy 시간단위 길이"""
  temp_wav_data, temp_x_data_length = audio_preprocess(x,sr,n_mels)
  temp_x_data = wav_padding(temp_wav_data)
  return temp_x_data, temp_x_data_length
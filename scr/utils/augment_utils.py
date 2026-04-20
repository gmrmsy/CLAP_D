import numpy as np

def speed_aug(mel_tm, L, speed, replace_val=-80.0):
    """
    mel_tm: (T,M)
    L: 유효 길이 (<=T)
    speed: 속도 배속 (0.8~1.25 등)
    return: 속도 조정된 Mel, 조정 후 길이
    """
    M, T = mel_tm.shape
    L = int(min(max(L, 0), T))

    # 원본 유효구간 [0..L-1]을 L_new 길이로 리샘플
    L_new = int(np.clip(round(L * speed), 1, T))
    out = np.full_like(mel_tm, replace_val)

    # 보간용 좌표 (선형보간)
    src_x = np.linspace(0.0, max(L - 1, 1), num=L, dtype=np.float32)
    dst_x = np.linspace(0.0, max(L - 1, 1), num=L_new, dtype=np.float32)

    # 각 멜 bin에 대해 1D 보간
    for m in range(M):
        out[m, :L_new] = np.interp(dst_x, src_x, mel_tm[m, :L])

    return out, L_new

def pitch_aug(mel_tm, shift_bins, replace_val=-80.0):
    """
    mel_tm: (M, T)
    shift_bins: 0~4 등
    """
    if shift_bins == 0:
      return mel_tm.copy()

    M, T = mel_tm.shape
    out = np.full_like(mel_tm, replace_val)

    if shift_bins > 0:
      k = min(shift_bins, M - 1)
      out[k:, :] = mel_tm[:M - k, :]
    else:
      k = min(-shift_bins, M - 1)
      out[:M - k, :] = mel_tm[k:, :]
    return out

def spec_pitch_speed_np(
    mel, true_lengths, sub_x, sub_x_len,
    min_speed_1=0.75, min_speed_2=0.85,
    max_speed_1=1.2, max_speed_2=1.5,
    min_pitch_bins=5, max_pitch_bins=10,
    replace_val=-80.0,
    rng=None
):
    """
    mel: (B,M,T) or (M,T)
    true_length: 실제 음성 데이터 길이
    sub_x: 패딩처리된 제시 단어
    min_speed_1, min_speed_2: 감속 범위
    max_speed_1, max_speed_2: 가속 범위
    max_pitch_bins: 음정 조절 단위
    replace_val: 패딩값
    rng: np.random모듈
    """

    slow_1 = 1+((1-min_speed_2)*2)
    slow_2 = 1+((1-min_speed_1)*2)
    fast_1 = 1-((max_speed_2-1)*0.5)
    fast_2 = 1-((max_speed_1-1)*0.5)

    if rng is None:
        rng = np.random.default_rng()

    # speed_aug, pitch_aug 사용 함수
    def _apply(x, L, x_sub, new_sub_x_len):
        M, T = x.shape
        out = x.copy()
        L = int(min(max(L, 0), T))

        # Speed
        if rng.random() < 0.5:
            speed = rng.uniform(slow_1, slow_2)
        else:
            speed = rng.uniform(fast_1, fast_2)
        out, new_L = speed_aug(out, L, speed, replace_val=replace_val)
        if new_L < T:
            out[:, new_L:] = replace_val

        # Pitch
        if max_pitch_bins > 0:
            shift = rng.integers(-max_pitch_bins, max_pitch_bins + 1)
            while shift < min_pitch_bins:
              shift = rng.integers(-max_pitch_bins, max_pitch_bins + 1)
            out = pitch_aug(out, int(shift), replace_val=replace_val)

        return out, new_L, x_sub, new_sub_x_len

    # 배치 단위
    if mel.ndim == 3:
        B, M, T = mel.shape
        true_lengths = np.asarray(true_lengths, dtype=np.int32)
        sub_x = np.asarray(sub_x, dtype=np.int32)
        sub_x_len = np.asarray(sub_x_len, dtype=np.int32)
        x_out = np.empty_like(mel)
        new_lengths = np.empty((B,), dtype=np.int32)
        new_sub_x = np.empty((B,), dtype=np.int32)
        new_sub_x_len = np.empty((B,), dtype=np.int32)
        for i in range(B):
            x_out[i], new_lengths[i], new_sub_x[i], new_sub_x_len[i] = _apply(mel[i], int(true_lengths[i]), sub_x[i], int(sub_x_len[i]))
        return x_out, new_lengths, new_sub_x, new_sub_x_len
    # 단일 단위
    else:
        x_out, new_L, new_sub_x, new_sub_x_len = _apply(mel, int(true_lengths), sub_x, int(sub_x_len))
        return x_out, new_L, new_sub_x, new_sub_x_len


def make_aug_dataset_pitch_speed(
    x, x_len, sub_x, sub_x_len, y, num_aug,
    min_speed_1=0.75, min_speed_2=0.85,
    max_speed_1=1.2, max_speed_2=1.5,
    min_pitch_bins=5, max_pitch_bins=10,
    replace_val=-80.0,
    merge_with_original=True,
    return_lengths=True,
    seed=None
):
    """
    x: (B,M,T)
    x_len: 실제 음성데이터 길이
    sub_x: 패딩처리된 제시 단어
    y: 타겟값
    num_aug: 증강 데이터 개수
    min_speed_1, min_speed_2: 감속 범위
    max_speed_1, max_speed_2: 가속 범위
    max_pitch_bins: 음정 조절 단위
    replace_val: 패딩값
    merge_with_original: 원본데이터 병합 여부
    return_lengths: 길이 변환 여부
    seed: np.random: 시드값
    """

    rng = np.random.default_rng(seed)
    B, M, T = x.shape
    L = y.shape[0]

    if B < num_aug:
        idx = rng.choice(B, size=num_aug, replace=True)
    elif B >= num_aug:
        idx = rng.choice(B, size=num_aug, replace=False)

    # 증강된 데이터 담을 빈 배열 생성
    x_aug = np.empty((num_aug, M, T), dtype=x.dtype)
    x_len_aug = np.empty((num_aug,), dtype=np.int32)
    sub_x_aug = np.empty((num_aug, 12), dtype=np.int32)
    sub_x_len_aug = np.empty((num_aug,), dtype=np.int32)
    y_aug = y[idx].copy()
    for i, j in enumerate(idx):
        x_aug[i], x_len_aug[i], sub_x_aug[i], sub_x_len_aug[i] = spec_pitch_speed_np(
            x[j], int(x_len[j]), sub_x[j], int(sub_x_len[j]),
            min_speed_1=min_speed_1, min_speed_2=min_speed_2,
            max_speed_1=max_speed_1, max_speed_2=max_speed_2,
            min_pitch_bins=min_pitch_bins, max_pitch_bins=max_pitch_bins,
            replace_val=replace_val,
            rng=rng
        )

    # 원본데이터 병합 여부에 따라 변환
    if merge_with_original:
        x_train = np.concatenate([x, x_aug], axis=0)
        y_train = np.concatenate([y, y_aug], axis=0)
        x_len_train = np.concatenate([x_len, x_len_aug], axis=0)
        sub_x_train = np.concatenate([sub_x, sub_x_aug], axis=0)
        sub_x_len_train = np.concatenate([sub_x_len, sub_x_len_aug], axis=0)

        perm = rng.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]
        x_len_train = x_len_train[perm]
        sub_x_train = sub_x_train[perm]
        sub_x_len_train = sub_x_len_train[perm]

        if return_lengths:
            return x_train, x_len_train, sub_x_train, sub_x_len_train, y_train
        else:
            return x_train, sub_x_train, y_train
    else:
        if return_lengths:
            return x_aug, x_len_aug, sub_x_aug, sub_x_len_aug, y_aug
        else:
            return x_aug, sub_x_aug, y_aug
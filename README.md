# CLAP-D : 마비말장애 언어 기능 자동 평가 모델 개발

> 뇌졸중 후 언어장애 진단을 위한 딥러닝 기반 언어 기능 평가 서비스 — **또박또박 말하기** 항목 담당

---

## 프로젝트 개요

실어증(Aphasia)·마비말장애(Dysarthria)는 뇌졸중, 외상성 뇌손상, 신경퇴행성 질환으로 인해 환자의 의사소통 능력과 삶의 질에 중대한 영향을 미칩니다.  
현재 임상 평가는 언어치료사·신경과 전문의의 청지각적 판단에 의존하며, 주관성 개입, 시간/인력 소모, 정량적 추적 불가 등의 한계가 있습니다.

본 프로젝트는 **음성 데이터 기반의 딥러닝 모델**로 이 한계를 보완하는 자동 평가 시스템을 구축하는 것을 목표로 합니다.

---

## 담당 항목 : 또박또박 말하기

CLAP-D 검사는 마비말장애를 위한 5가지 항목으로 구성되며, 그 중 **또박또박 말하기** 항목을 담당하였습니다.

| 항목 | 내용 |
|------|------|
| 또박또박 말하기 | 평가자가 제시한 단어·문장을 또박또박 말하는 검사, **25문항** |

- 검사자(환자)의 **발화 음성** + 평가자의 **제시 텍스트**, 두 입력을 동시에 처리
- 각 문항의 점수를 예측하여 최종 채점 점수(정수)를 산출
- 최종 평가 지표: **Accuracy** 및 **Pearson 상관계수(r)**

---

## 파일 구조

```
CLAP_D/
├── scr/
│   ├── data/
│   │   ├── preprocess.py          # 음성→멜스펙, 텍스트→자모 인덱스 전처리
│   │   └── split.py               # 점수별 계층 분리 후 train/valid/test 분할
│   ├── models/
│   │   └──               # Cross-Attention 모델
│   ├── utils/
│   │   ├── augment_utils.py       # 피치 이동 + 속도 변환 증강
│   │   ├── data_utils_class.py    # 데이터 로드, 증강 통합, reliable 선택
│   │   ├── train_utils.py         # 학습 루프, 손실 가중치 적용
│   │   ├── wav_utils.py           # WAV → 멜스펙트로그램 변환
│   │   └── jamo_utils.py          # 한국어 자모 분리 및 정수 인코딩
│   ├── train/
│   │   └── train.py               # 전체 실험 조합 그리드 서치 실행
│   └── evaluate/
│       └── test.py                # 모델 로드 → 예측 → accuracy/corr 산출 → CSV 저장
├── data/
│   ├── csv/                       # 모델별 score_df, compare_df 결과
│   ├── npy/                       # 전처리된 입력 데이터 (※ 미포함)
│   └── wav_flie/                  # 원본 음성 데이터 (※ 미포함)
└── checkpoints/                   # 학습된 모델 가중치 (※ 미포함)
```

> ※ **data/npy**, **data/wav_flie**: 저작권이 있는 데이터를 제공받아 사용하였으므로 저장소에 포함하지 않습니다.  
> ※ **checkpoints**: 모델 파일(.keras)의 용량이 커 저장소에 포함하지 않습니다.

---

## 코드 흐름

```mermaid
flowchart TD
    W["음성 파일 .wav"]
    C["채점 정보 .csv\n문항번호 · 제시텍스트 · 정답점수 · 만점"]

    W -->|"wav_utils.py\x_data_preprocess()"| NP["멜 스펙트로그램 .npy\x1_data / x1_data_length"]
    C -->|"jamo_utils.py\text_to_ctc_indices()"| NP2["텍스트 인코딩 .npy\x2_data / x2_data_length / y_data"]

    NP & NP2 --> LOAD["data_utils_class.py\data_load() 필요한 데이터 로드를 위한 클래스"]

    LOAD --> D1{데이터 구성}
    D1 -->|"no1 ~ no25"| SPL["data_utils.py\data_load()\make_list() 문항별 계층 분리"]
    D1 -->|"total"| CON["data_utils.py\data_load()\total_concat() 전체 문항 병합"]
    D1 -->|"reliable"| REL["data_utils.py\data_load()\select_reliable_data() Target 분산 상위 3문항 선택"]
    

    SPL, CON, REL --> D2{데이터 증강}
    D2 -->|"aug"| AUG["data_utils.py\augment()\augment_utils.py\speed_aug() + pitch_aug()"]
    D2 -->|"no aug"| READY
    AUG --> READY["학습 / 검증 / 테스트 배열"]

    model.py\make_talk_clean_model() 
    READY --> D3{CNN 차원}
    D3 -->|"1D" or "2D"| CNN["Conv1D or Conv2D × 2"]
    CNN --> RNN["Bidirectional GRU→ Sinusoidal PE→ Cross Attention × 6"]

    RNN --> D4{출력 활성화}
    D4 -->|"linear"| OUT["Dense(1, linear)"]
    D4 -->|"relu"| OUT2["Dense(1, relu)"]

    OUT & OUT2 --> D5{손실 가중치}
    D5 -->|"lossO"| WGT["train_utils.py\model_train() 역수 가중치 부여"]
    D5 -->|"lossX"| FIT

    WGT, FIT --> CKPT[".keras 저장"]

    CKPT --> PRED["test.py\model.predict() 예측값 0~1"]
    PRED --> POST["× Score_Alloc → round() 정수 점수 변환"]
    POST --> RESULT["Accuracy / Pearson r 결과 CSV 저장"]
```

---

## 기술 스택

`Python` `TensorFlow/Keras` `NumPy` `Pandas` `Mel-Spectrogram`  `Multi-Head Attention` `CNN` `Bidirectional GRU`

---

## 데이터 구성

| 항목 | 내용 |
|------|------|
| 전체 샘플 수 | 1,500개 (25문항 × 60개) |
| 음성 입력 (x1) | 멜 스펙트로그램, shape `(128, 312)`, 패딩값 `-80.0` |
| 텍스트 입력 (x2) | 제시 단어 자모 분리 후 정수 인코딩, shape `(12,)` |
| 정답 레이블 | `Score(Refer)` — 평가자가 부여한 실제 채점 점수 (정수) |
| 학습 타겟 | `Target` = **득점 / 만점** — 0~1 사이의 연속 비율값 |

모델은 득점 비율(0~1)을 예측하고, 예측값에 해당 문항의 만점(`Score(Alloc)`)을 곱한 뒤 반올림하여 최종 정수 점수로 변환합니다.

```
예측 흐름: 모델 출력(0~1) × Score(Alloc) → 반올림 → 최종 점수
```

### 핵심 문제: 심각한 점수 분포 편향

```
Target = 1 (만점)     :  ~74.46%  (대다수)
Target ≠ 1 (감점/0점)  :  ~25.54%  (소수)
```

전체 25문항 모두 만점(Target=1) 비율이 압도적으로 높고 점수 분산이 낮습니다.  
이는 **단순히 모든 샘플을 만점으로 예측해도 74.46%의 accuracy가 나오는 구조**를 만들어,  
모델이 점수 분포 편향에 의해 항상 만점을 출력하는 방향으로 수렴할 위험이 있습니다.

---

## [모델 아키텍처 (Baseline: 1D_linear_lossO)](https://github.com/gmrmsy/CLAP_D/blob/main/scr/models/model_1D.py#L57)

발화 음성(Query)과 제시 텍스트(Key) 간의 유사성을 측정하기 위해  
**Cross Multi-Head Attention** 구조를 중심으로 설계하였습니다.

```
[음성 입력 (128×312)]
        ↓
  SequenceMask
        ↓
  Conv1D × 2 + BatchNorm + HardTanh
        ↓
  LayerNormalization
        ↓
  Bidirectional GRU (64×2)            [텍스트 입력 (12,)]
        ↓                                     ↓
  Sinusoidal Positional Encoding       Embedding (55→16)
        ↓                                     ↓
  ┌───────────────────────────────────────────────────┐
  │  Multi-Head Attention × 6 (Q=음성, K/V=텍스트)     │
  │  + Add & Norm + FFN (Dense 512→128) + Add & Norm  │
  └───────────────────────────────────────────────────┘
        ↓
  GlobalAveragePooling1D + GlobalMaxPooling1D (concat)
        ↓
  Dense (512, relu) → Dense (1, linear)
        ↓
  예측값 × Score(Alloc) → round → 최종 점수
```

| 구성 요소 | 세부 내용 |
|-----------|-----------|
| 패딩 마스크 | 유효 음성 길이 추출, 패딩 제외 |
| CNN | Conv1D(512, kernel=11, stride=2) or Conv1D(512, kernel=11, stride=1) |
| 활성화 | HardTanh (clip [-20, 20]) |
| RNN | Bidirectional GRU(64), dropout=0.1 |
| 위치 인코딩 | Sinusoidal Positional Encoding |
| Attention | MultiHeadAttention(heads=16, key_dim=32, dropout=0.1) × 6 |
| 출력층 | Dense(1, activation=`out`) — relu 또는 linear |
| 학습 | Adam(lr=0.001), loss=MSE, EarlyStopping(patience=10) |
| 배치 크기 | 64, 최대 100 epoch |

---

## 실험 조합

총 **8가지 모델** × **25문항 + total + reliable + aug 버전** = 대규모 그리드 서치  
(`itertools.product`로 모든 조합 자동 생성)

| 변수 | 선택지 | 비고 |
|------|--------|--------|
| 학습 데이터 구성 | no1~no25, total, reliable | 단일 항목의 데이터로 학습된 모델이 전체 항목을 예측할 수 있을지 확인하기 위해 각 항목별로도 학습을 진행 |
| CNN 차원 | 1D / 2D | 312의 시간축에 128의 특성을 지닌 1D, 가로 세로 흑백의 이미지를 가진 2D 두 가지 관점의 차이가 있는지 확인하기 위해 진행 |
| 출력 활성화 | relu / linear | 학습 타겟(득점 비율)의 범위가 0~1 임을 고려 예측값이 음수가 되지 않도록 ReLU를 시도 |
| 손실 가중치 | lossO (있음) / lossX (없음) | 데이터 불균형으로 인해 생기는 예측편향이 손실 가중치로 해결되는지 확인하기 위해 진행 |
| 데이터 증강 | aug 적용 / 미적용 | 미만점 항목의 데이터를 증강을 통해 모델의 정확도를 높일 수 있을지 확인을 위해 진행 |

---

## 성능 향상을 위한 시도

점수 분포 편향·일반화 성능 부족 문제를 해결하기 위해 아래 5가지 축으로 실험을 진행하였습니다.

---

### [1. 학습 데이터 구성 변경](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/data_utils.py#L8)

25개 문항을 어떻게 조합해 학습할지도 변수로 실험하였습니다.

| 구성 | 내용 |
|------|------|
| `no1` ~ `no25` | 각 문항별 독립 모델 학습 |
| `total` | 25문항 전체를 하나로 합쳐 학습 |
| `reliable` | 점수 분산이 가장 높은 상위 3개 문항만 선택하여 학습 |

```python
# reliable: Target 표준편차 기준 상위 3문항 선택
temp_list = d_2_csv.groupby('QUESTION_NO')['Target'].describe()[['std']] \
              .sort_values('std', ascending=False).index[:3]
```

**의도**: 분산이 낮은 문항(대부분 만점)은 모델이 "항상 만점 출력"을 학습하도록 유도하므로 제외하고,  
상대적으로 점수 다양성이 높은 문항만 선별하여 학습의 질을 높이고자 함  
**결과**: reliable도 total과 마찬가지로 기준선 수준을 벗어나지 못함. 문항 수 자체가 줄어 데이터 부족 심화.

---

### 2. CNN 입력 차원 변경: 1D → 2D

| 구분 | 방식 |
|------|------|
| **1D** (baseline) | 멜 스펙트로그램을 `(시간, 주파수)` 형태로 변환 후 Conv1D 처리 |
| **2D** | `(128, 312, 1)` 원본 형태 그대로 Conv2D 처리 후 Reshape |

```python
# 1D: (B, 128, 312, 1) → Permute → (B, 312, 128) → Conv1D
# 2D: (B, 128, 312, 1) → Conv2D(32, (41,11), stride=(2,2)) → Conv2D(32, (21,11), stride=(2,1))
```

**의도**: 2D CNN이 주파수-시간 축의 2차원 패턴(포만트 구조 등)을 더 잘 포착할 것이라 기대  
**결과**: 동일 조건(linear, lossX)에서 total accuracy 동등(0.5793), aug 데이터 corr에서 2D가 소폭 우세(0.5160 vs 0.4646)

---

### 3. 출력 활성화 함수 변경: ReLU → Linear

학습 타겟(득점 비율)은 0~1 범위이므로, 예측값이 음수가 되지 않도록 ReLU를 시도하였습니다.

| 구분 | 출력층 활성화 | 의도 |
|------|-------------|------|
| **relu** | `Dense(1, activation='relu')` | 예측값 ≥ 0 보장 |
| **linear** (baseline) | `Dense(1, activation='linear')` | 제약 없이 학습 |

**결과**: ReLU 모델은 학습 중 출력이 0으로 완전히 수렴하는 현상이 발생하였습니다.

출력층에 ReLU를 적용하면, 뉴런의 입력값이 음수가 될 경우 출력이 0으로 고정되고 역전파 기울기도 0이 됩니다 (Dying ReLU 현상). MSE 손실 환경에서 이 상태가 되면 가중치가 더 이상 업데이트되지 않아 모델이 **모든 샘플에 대해 0점(예측값=0)을 출력하는 상태로 고착**됩니다. 그 결과 실제 정답이 0점인 샘플만 맞히게 되어 accuracy ≈ 14.13%에 머물게 됩니다.

반면 linear 활성화는 이런 수렴 문제 없이 정상적으로 학습이 이루어졌습니다.

---

### [4. 점수 분포 편향 보정: 손실 가중치 (lossO / lossX)](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/train_utils.py#L4)

[`weight_return`](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/train_utils.py#L15) 함수에서 샘플 가중치를 계산하여 학습에 적용합니다.

```python
# lossO: 점수별 샘플 빈도의 역수(제곱근)를 가중치로 부여
weight[score] = sqrt(total / (num_unique_scores × count[score]))

# lossX: 모든 샘플 가중치 = 1 (가중치 없음)
```

| 구분 | 내용 |
|------|------|
| **lossO** (baseline) | 만점(Target=1) 외 소수 샘플에 더 높은 손실 가중치 부여 |
| **lossX** | 가중치 없이 균등 학습 |

**의도**: 만점(Target=1) 외 소수 샘플에 높은 가중치를 줘 모델이 만점만 예측하지 않도록 유도  
**결과**: 역설적으로 lossX(가중치 없음)의 total accuracy가 더 높음(0.5793 vs 0.2593).  
lossO는 소수 샘플에 과한 가중치가 부여되어 오히려 전체 예측이 불안정해짐. 일부 문항(no14)에서는 corr이 소폭 높아짐(r=0.6398).

---

### [5. 데이터 증강: 피치 이동 + 속도 변환](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/augment_utils.py#L116)

만점(Target=1) 외 소수 샘플의 절대량이 부족하여 음성 증강을 적용하였습니다.

```python
# 속도 변환: 원본 발화의 재생 속도를 무작위 변경 (감속/가속)
speed = random(slow_range) or random(fast_range)
# → 멜 스펙트로그램의 시간축을 선형 보간으로 리샘플링

# 피치 이동: 멜 빈(bin)을 위아래로 무작위 이동
shift_bins = random(-max_pitch_bins, +max_pitch_bins)
# min_pitch_bins=5, max_pitch_bins=10 (5~10 bin 범위 강제)
```

| 증강 기법 | 구현 |
|-----------|------|
| 속도 변환 | [augment_utils.py / speed_aug()](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/augment_utils.py#L3) — 멜 스펙트로그램 시간축 선형 보간 리샘플링 |
| 피치 이동 | [augment_utils.py / pitch_aug()](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/augment_utils.py#L27) — 멜 빈(bin) 단위 주파수 축 이동 |
| 데이터 병합 | [augment_utils.py / make_aug_dataset_pitch_speed()](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/augment_utils.py#L116) — 속도/피치 함수를 사용해 데이터 증강 |
| 소수 샘플 선택 | [data_utils.py / augment()](https://github.com/gmrmsy/CLAP_D/blob/main/scr/utils/data_utils.py#L105) — idx_0/idx_1 분리 — Target≠1 샘플만 선택하여 증강 |

**의도**: 소수 샘플을 늘려 점수 분포 편향 완화  
**결과**: aug 적용 시 일부 문항에서 개선. 특히 1D_relu_lossX의 `no4_aug`가 corr=0.7001로 전체 최고 상관계수 달성. 단, 이는 **특정 문항 + aug 조합에서의 이례적 현상**으로, total/reliable 예측에서는 개선되지 않아 신뢰하기 어려움.

---

## 실험 결과 요약

### 핵심 지표 (total 예측 기준)

| 모델 | total acc | total corr | 유효 문항 수 (acc>0.30) |
|------|-----------|------------|------------------------|
| **1D_linear_lossX** | **0.5793** | 0.4748 | 16 / 27 |
| **2D_linear_lossX** | **0.5793** | 0.4748 | 17 / 27 |
| 1D_linear_lossO | 0.2593 | 0.5129 | 4 / 27 |
| 2D_linear_lossO | 0.2593 | 0.5129 | 2 / 27 |
| 1D_relu_lossO | 0.1413 | N/A | 4 / 27 |
| 2D_relu_lossO | 0.1413 | N/A | 1 / 27 |
| 1D_relu_lossX | 0.1413 | N/A | 0 / 27 |
| 2D_relu_lossX | 0.1413 | N/A | 0 / 27 |

---

## 핵심 발견: Accuracy의 함정

가장 좋아 보이는 문항(no1)의 accuracy 0.7447은 실제 학습의 결과가 아니었습니다.

### 예측 분포 분석 (no1, 1D_linear_lossX)

```
              실제=0  실제=1  실제=2  실제=3
  예측=1 (300)   65     235      0       0
  예측=2 (780)  102      99    579       0
  예측=3 (420)   45      17     55     303

  점수 0 정답률:  0 / 212 =  0.0%   ← 한 번도 맞히지 못함
  점수 1 정답률: 235 / 351 = 66.9%
  점수 2 정답률: 579 / 634 = 91.3%
  점수 3 정답률: 303 / 303 = 100%

  전체 accuracy: 1117 / 1500 = 74.47%
```

**문제점**: 0점 예측을 전혀 하지 못하고, 특정 점수값으로 편향된 예측을 함.
이는 **74.46%가 데이터의 지배적 점수(만점=Target=1)의 비율 그 자체**이기 때문입니다.

모든 실험에서 높은 accuracy를 보인 모델들은 공통적으로  
**지배적인 점수값만 예측하거나 특정 점수 조합에서 우연히 높은 정확도를 기록**하는  
점수 분포 편향된 학습의 결과였으며, 실제로 언어장애 진단에 활용할 수 있는 수준의 일반화 성능을 달성하지 못하였습니다.

### Accuracy vs 상관계수

| 지표 | 의미 | 구현 | 한계 |
|------|------|------|------|
| Accuracy | 예측값 == 정답인 비율 | `np.mean(process_score == correct_y)` | 점수 분포 편향 시 지배적 점수 예측만으로도 높게 나옴 |
| Pearson r | 예측값과 실제값의 선형 상관 | `np.corrcoef(...)` | 실제 임상적 유용성을 더 잘 반영 |

안정적인 조건(linear + lossX)에서의 corr은 약 0.47~0.51 수준에 머물렀습니다.

---

## 프로젝트 한계 및 회고

| 문제 | 내용 |
|------|------|
| **점수 분포 편향** | 전체 25문항 모두 만점 비율 ~74%. 손실 가중치·증강을 시도했으나 근본적 해결 불가 |
| **데이터 부족** | 문항당 약 60개 샘플으로 딥러닝 모델의 일반화에 충분하지 않음 |
| **평가 지표 오류** | Accuracy가 점수 분포 편향 상황을 반영하지 못해, 학습 실패를 성공으로 오인할 뻔함 |
| **ReLU 수렴 문제** | MSE 손실 + ReLU 출력층 조합에서 Dying ReLU 현상으로 예측값이 0으로 고착 |
| **집계(total) 역효과** | 개별 문항보다 전체 앙상블(total)이 오히려 약함 — 특정 점수 예측 능력이 평균화 과정에서 손실 |

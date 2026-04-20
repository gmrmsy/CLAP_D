"""
한국어 자소 토큰화 및 음절 복원 유틸리티
"""

# 자모 리스트 정의 
CHOSUNG_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ',
                'ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSUNG_LIST = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ',
                 'ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSUNG_LIST = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ',
                 'ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ',
                 'ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
RESERVED = ["<pad>", "<sos>", "<eos>", " "]   # <pad> == <blank> (CTC)

# 어휘 인덱스 테이블
ALL_JAMOS = sorted(set(CHOSUNG_LIST + JUNGSUNG_LIST + JONGSUNG_LIST))
ALL_JAMOS.remove('')   # 종성의 빈 문자열 제거
ALL_JAMOS = RESERVED + ALL_JAMOS

jamo_to_index = {j: i for i, j in enumerate(ALL_JAMOS)}
index_to_jamo = {i: j for j, i in jamo_to_index.items()}

vocab_size = len(jamo_to_index)   # 55


# 한글 문자열 → 자소 분해
def decompose_hangul(text):
    """
    한글 문자열을 자소 시퀀스로 분해.
    <sos>, <eos> 토큰을 앞뒤에 추가.

    예: "안녕" → ["<sos>", "ㅇ","ㅏ","ㄴ", "ㄴ","ㅕ","ㅇ", "<eos>"]
    """
    result = ["<sos>"]
    for char in text:
        if '가' <= char <= '힣':
            code = ord(char) - ord('가')
            cho  = CHOSUNG_LIST[code // (21 * 28)]
            jung = JUNGSUNG_LIST[(code % (21 * 28)) // 28]
            jong = JONGSUNG_LIST[code % 28]
            result.extend([cho, jung])
            if jong != '':
                result.append(jong)
        else:
            result.append(char)   # 공백 등 비한글 문자 그대로 추가
    result.append("<eos>")
    return result


# 자소단위 문자열 -> 인덱스 시퀀스 변환
def char_to_index(text):
    """input: 자모 시퀀스(list)
        output: 인덱스 시퀀스(list)"""
    text = text[1:] if text[0] == " " else text
    text = text[:-1] if text[-1] == " " else text
    idx_seq = [jamo_to_index[j] for j in text if j in jamo_to_index]
    return idx_seq, len(idx_seq)


# 인덱스 시퀀스 padding처리
def seq_padding(seq, seq_max_len=12):
    """input: 인덱스 시퀀스(list)
        output: 패딩된 인덱스 시퀀스(list)"""
    
    pad_width = seq_max_len - len(seq)  # 얼마나 채워야 하는지
    if pad_width > 0:
        # 오른쪽(열 끝)에 0을 채움: ((행 시작, 행 끝), (열 시작, 열 끝))
        for _ in range(pad_width):
            seq.append(0)
    return seq


# 한글 문자열 전처리
def text_to_ctc_indices(text):
    """한글 문자열 → (인덱스 시퀀스, 시퀀스 길이)"""
    import re
    text = re.sub(r'[^가-힣 ]', '', text)

    jamo_seq = decompose_hangul(text)
    indices, temp_y_data_length = char_to_index(jamo_seq)
    temp_y_data = seq_padding(indices)
    
    return temp_y_data, temp_y_data_length
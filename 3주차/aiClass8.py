import numpy as np
import tensorflow as tf
import os, re, requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. 데이터셋 로드 (안전한 경로 활용)
# ==========================================
url = "https://raw.githubusercontent.com/devm-m/Seq2Seq-Chatbot/master/fra.txt"

print("데이터셋 로딩 시도 중...")
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        print(f"온라인 데이터 로드 성공! (총 {len(lines)}개 문장)")
    else:
        raise Exception
except:
    print("온라인 연결 실패. 실습용 내부 데이터를 생성합니다.")
    # 샘플 데이터를 충분히 늘려 에러를 방지합니다.
    sample_raw = ["I am a student.\tJe suis étudiant.", "He is a doctor.\tIl est médecin.", 
                  "It is cold.\tIl fait froid.", "I love you.\tJe t'aime."]
    lines = sample_raw * 500 

# ==========================================
# 2. 데이터 전처리 및 샘플링 (에러 방지 핵심 로직)
# ==========================================
def preprocess(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
    return s.strip()

# 데이터가 충분하면 10,000번째부터, 부족하면 처음부터 가져옵니다.
start_idx = 10000 if len(lines) > 11000 else 0
num_samples = 1000 # 학습할 문장 개수
selected_lines = lines[start_idx : start_idx + num_samples]

input_texts, target_texts = [], []
for line in selected_lines:
    if '\t' in line:
        parts = line.split('\t')
        input_texts.append(preprocess(parts[0]))
        target_texts.append('\t ' + preprocess(parts[1]) + ' \n')

if not input_texts:
    print("오류: 학습할 데이터가 추출되지 않았습니다. 인덱스를 확인하세요.")
else:
    print(f"{len(input_texts)}개의 문장으로 학습을 준비합니다.")

# ==========================================
# 3. 토큰화 및 시퀀스 변환
# ==========================================
def tokenize(texts):
    tokenizer = Tokenizer(filters='', lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer, tokenizer.texts_to_sequences(texts)

input_tok, input_seq = tokenize(input_texts)
target_tok, target_seq = tokenize(target_texts)

# 모든 문장의 길이를 최대 길이에 맞춰 늘려줍니다.
max_in, max_out = max(len(s) for s in input_seq), max(len(s) for s in target_seq)
encoder_input = pad_sequences(input_seq, maxlen=max_in, padding='post')
decoder_input = pad_sequences(target_seq, maxlen=max_out, padding='post')

# 정답지 생성 (Sparse 방식: 메모리 절약형)
decoder_target = np.zeros_like(decoder_input, dtype='float32')
for i, seq in enumerate(target_seq):
    for t, word_id in enumerate(seq):
        if t > 0: decoder_target[i, t-1] = word_id

# ==========================================
# 4. Seq2Seq 모델 설계 (LSTM 구조)
# ==========================================

h_dim = 256
v_in, v_out = len(input_tok.word_index) + 1, len(target_tok.word_index) + 1

# 인코더 (입력 문장을 요약함)
enc_in = Input(shape=(None,))
enc_emb = Embedding(v_in, h_dim)(enc_in)
_, h, c = LSTM(h_dim, return_state=True)(enc_emb)

# 디코더 (요약을 보고 번역함)
dec_in = Input(shape=(None,))
dec_emb = Embedding(v_out, h_dim)(dec_in)
dec_lstm = LSTM(h_dim, return_sequences=True, return_state=True)
dec_out, _, _ = dec_lstm(dec_emb, initial_state=[h, c])
dec_dense = Dense(v_out, activation='softmax')
dec_out = dec_dense(dec_out)

model = Model([enc_in, dec_in], dec_out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print("모델 학습 중... (잠시만 기다려주세요)")
# 데이터 부족 경고를 막기 위해 에폭과 배치를 조절합니다.
model.fit([encoder_input, decoder_input], decoder_target, 
          batch_size=32, epochs=50, verbose=0) 

# ==========================================
# 5. 번역 실행 (추론 함수)
# ==========================================
enc_model = Model(enc_in, [h, c])
dec_h_in, dec_c_in = Input(shape=(h_dim,)), Input(shape=(h_dim,))
d_out, d_h, d_c = dec_lstm(dec_emb, initial_state=[dec_h_in, dec_c_in])
d_out = dec_dense(d_out)
dec_model = Model([dec_in, dec_h_in, dec_c_in], [d_out, d_h, d_c])

def translate(input_data):
    # 인코더에 영어 문장을 넣어 '생각(상태)'을 추출합니다.
    states = enc_model.predict(input_data, verbose=0)
    # 시작 기호를 준비합니다.
    target = np.zeros((1, 1))
    target[0, 0] = target_tok.word_index['\t']
    
    result = ""
    for _ in range(max_out):
        # 현재 상태를 바탕으로 다음 단어를 예측합니다.
        out, h, c = dec_model.predict([target, states[0], states[1]], verbose=0)
        idx = np.argmax(out[0, -1, :]) # 확률이 가장 높은 단어 선택
        word = target_tok.index_word.get(idx, '')
        
        if word == '\n' or word == '': break # 끝 기호를 만나면 종료
        result += " " + word
        # 예측한 단어를 다음 단계의 입력으로 사용합니다.
        target[0, 0], states = idx, [h, c]
    return result.strip()

# ==========================================
# 6. 최종 번역 결과 출력
# ==========================================
print("\n" + "="*40)
print("번역 테스트 결과")
print("="*40)
for i in range(5):
    # 데이터가 부족할 경우 인덱스 에러 방지
    if i >= len(input_texts): break
    
    test_seq = encoder_input[i:i+1]
    prediction = translate(test_seq)
    print(f"영어(입력): {input_texts[i]}")
    print(f"불어(예측): {prediction}")
    print("-" * 20)

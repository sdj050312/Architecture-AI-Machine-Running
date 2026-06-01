# 1. 필요한 도구들을 가져옵니다.
import urllib.request # 인터넷에서 데이터를 다운로드하는 도구입니다.
import pandas as pd # 표 형태의 데이터를 다루기 쉽게 해주는 도구입니다.
import torch # 딥러닝을 위한 핵심 파이토치 도구입니다.
import torch.nn as nn # 신경망 블록(부품)들이 들어있는 도구함입니다.
import math # 위치 정보를 계산할 때 사용할 수학 도구입니다.

print("1. 데이터를 준비하고 나만의 단어 사전을 만듭니다...")

# 2. 송영숙님의 한국어 챗봇 데이터를 다운로드합니다.
urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", "ChatbotData.csv")

# 3. 다운로드한 데이터를 읽어옵니다. (빠른 실습을 위해 100개만 사용합니다)
data = pd.read_csv('ChatbotData.csv')[:100]

# 4. 외부 사전 없이 '글자(Character)' 단위로 나만의 단어 사전을 만듭니다.
# 챗봇이 이해할 수 있도록 모든 문장의 글자를 분해해서 번호를 매기는 과정입니다.
text_data = "".join(data['Q'].tolist() + data['A'].tolist()) # 질문과 답변을 모두 하나의 긴 글로 합칩니다.
chars = sorted(list(set(text_data))) # 중복된 글자를 없애고 순서대로 정렬합니다.

# 5. 컴퓨터가 알아볼 수 있도록 특수 기호 3개를 사전에 추가합니다.
# <PAD>: 문장 길이를 맞추기 위한 빈칸, <SOS>: 대화 시작, <EOS>: 대화 끝
vocab = ['<PAD>', '<SOS>', '<EOS>'] + chars 
vocab_size = len(vocab) # 단어 사전의 총 크기(글자 종류의 수)를 기억해둡니다.

# 6. 글자를 숫자(인덱스)로, 숫자를 다시 글자로 바꿔주는 딕셔너리를 만듭니
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

# 7. 문장을 숫자로 바꿔주는 함수를 만듭니다.
def encode(text):
    # 문장 맨 앞에 시작(<SOS>) 기호를 넣고, 글자들을 숫자로 바꾼 뒤, 끝에 종료(<EOS>) 기호를 넣습니다.
    return [char_to_idx['<SOS>']] + [char_to_idx[c] for c in text if c in char_to_idx] + [char_to_idx['<EOS>']]

# ==========================================
# [트랜스포머 핵심 부품 만들기]
# ==========================================

print("2. 트랜스포머 모델의 부품을 조립합니다...")

# 8. 단어의 '순서(위치)'를 기억하게 해주는 포지셔널 인코딩(Positional Encoding) 부품입니다.
# 트랜스포머는 문장을 한꺼번에 읽기 때문에 단어의 위치를 알려주어야 합니다.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__() # 부모 클래스(nn.Module)의 설정을 가져옵니다.
        # 위치 정보를 저장할 빈 도화지를 만듭니다.
        pe = torch.zeros(max_len, d_model)
        # 단어의 위치(0, 1, 2...)를 나타내는 숫자를 만듭니다.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 사인(sin)과 코사인(cos) 함수를 이용해 겹치지 않는 위치 패턴을 계산합니다.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 짝수 자리에는 사인파를 칠해줍니다.
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 자리에는 코사인파를 칠해줍니다.
        pe[:, 1::2] = torch.cos(position * div_term)
        # 계산된 도화지의 차원을 모델에 맞게 조절합니다.
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 학습되는 값이 아니라 고정된 규칙이므로 모델에 그대로 등록해둡니다.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 들어온 단어들(x)에 미리 만들어둔 위치 정보(pe)를 더해줍니다.
        return x + self.pe[:x.size(0), :]

# 9. 전체 트랜스포머 챗봇 모델을 조립합니다.
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__() # 부모 클래스의 설정을 가져옵니다.
        # 숫자로 된 단어를 풍부한 의미를 가진 벡터(d_model 크기)로 바꿔줍니다.
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 위에서 만든 위치 정보 부여 부품을 가져옵니다.
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 파이토치에 내장된 트랜스포머 엔진(심장)을 가져옵니다.
        # nhead: 한 번에 여러 측면을 동시에 집중해서 보는(Multi-Head Attention) 개수입니다.
        # num_layers: 트랜스포머 블록을 몇 층으로 쌓을지 결정합니다.
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers)
        
        # 트랜스포머가 생각한 결과를 다시 우리 단어 사전 크기로 바꿔서 어떤 글자인지 맞추게 합니다.
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model # 모델의 크기를 기억해둡니다.

    # 챗봇이 미래의 정답을 미리 컨닝하지 못하게 가려주는(Mask) 함수입니다.
    def generate_square_subsequent_mask(self, sz):
        # 1로 채워진 정사각형 행렬을 만들고, 대각선 아래쪽만 남깁니다.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # 가려야 할 곳(미래)은 마이너스 무한대(-inf)로, 볼 수 있는 곳(과거)은 0으로 바꿉니다.
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # 모델이 실제로 작동하는(데이터가 흘러가는) 과정을 정의합니다.
    def forward(self, src, tgt):
        # 입력(질문)과 정답(답변)을 임베딩하고 위치 정보를 더해줍니다. 
        # (math.sqrt를 곱하는 것은 임베딩 값을 안정화하기 위한 트랜스포머의 공식입니다.)
        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        # 답변(tgt)이 미래의 단어를 보지 못하도록 컨닝 방지 마스크를 만듭니다.
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        
        # 트랜스포머 엔진에 질문, 답변, 마스크를 모두 넣고 돌립니다.
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # 트랜스포머의 최종 생각을 우리가 아는 글자로 변환하여 내보냅니다.
        return self.fc_out(output)

# ==========================================
# [모델 학습시키기]
# ==========================================

print("3. 데이터를 준비하고 학습을 시작합니다! (수학 문제를 푸는 중...)")

# GPU가 있으면 GPU를, 없으면 CPU를 사용하도록 설정합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 트랜스포머 챗봇 객체를 생성하고 선택한 장치(GPU/CPU)로 보냅니다.
model = TransformerChatbot(vocab_size=vocab_size).to(device)

# 모델이 정답과 얼마나 틀렸는지 채점하는 도구(손실 함수)입니다. 
# <PAD>(빈칸)는 학습하지 않도록 설정합니다.
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])

# 모델이 틀린 만큼 스스로 고쳐나가도록 돕는 최적화 도구(Adam)입니다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습을 100번(에포크) 반복합니다.
epochs = 100 

model.train() # 모델을 '학습 모드'로 바꿉니다.

for epoch in range(epochs):
    total_loss = 0 # 이번 에포크의 총 오차(틀린 정도)를 저장할 변수입니다.
    
    # 100개의 데이터를 하나씩 꺼내서 학습합니다.
    for _, row in data.iterrows():
        # 질문(Q)을 숫자로 바꾸고 파이토치 형태(Tensor)로 만든 뒤 세로로 세웁니다(unsqueeze).
        src = torch.tensor(encode(row['Q'])).unsqueeze(1).to(device)
        
        # 답변(A)도 숫자로 바꾸고 세로로 세웁니다.
        tgt_full = torch.tensor(encode(row['A'])).unsqueeze(1).to(device)
        
        # 모델에게 입력할 정답 데이터(마지막 <EOS> 기호 제외)
        tgt_input = tgt_full[:-1, :]
        # 모델이 맞춰야 할 실제 정답 데이터(맨 앞 <SOS> 기호 제외)
        tgt_expected = tgt_full[1:, :]
        
        # 이전에 계산했던 모델의 찌꺼기(기울기)를 깨끗이 지웁니다.
        optimizer.zero_grad()
        
        # 질문과 정답 일부를 모델에 넣고 다음 글자를 예측하게 합니다.
        output = model(src, tgt_input)
        
        # 예측한 결과물(output)과 실제 정답(tgt_expected)을 비교해 오차를 채점합니다.
        # 비교하기 쉽게 형태를 한 줄로 쭉 폅니다(view).
        loss = criterion(output.view(-1, vocab_size), tgt_expected.view(-1))
        
        # 채점된 오차를 바탕으로 모델 안의 톱니바퀴들을 어떻게 돌릴지 계산합니다(역전파).
        loss.backward()
        
        # 계산된 방향대로 톱니바퀴를 살짝 움직여서 모델을 성장시킵니다.
        optimizer.step()
        
        total_loss += loss.item() # 계산된 오차를 누적합니다.
        
    # 10번 반복할 때마다 현재 학습 상태(오차율)를 화면에 보여줍니다.
    if (epoch + 1) % 10 == 0:
        print(f"에포크 {epoch+1}/{epochs} 완료 - 평균 오차(Loss): {total_loss/len(data):.4f}")

# ==========================================
# [챗봇과 대화하기]
# ==========================================

print("\n4. 학습 완료! 챗봇과 대화를 시작합니다. (종료하려면 '종료' 입력)")

model.eval() # 모델을 더 이상 학습하지 않는 '평가(실전) 모드'로 바꿉니다.

while True:
    user_input = input("나: ") # 사용자의 말을 입력받습니다.
    if user_input == '종료': # '종료'라고 쓰면 루프를 빠져나갑니다.
        print("챗봇: 대화를 종료합니다. 안녕!")
        break
        
    # 사용자의 말을 우리가 만든 사전으로 번역(숫자로 변환)합니다.
    src = torch.tensor(encode(user_input)).unsqueeze(1).to(device)
    
    # 챗봇이 첫 마디를 뗄 수 있도록 <SOS> 기호를 줍니다.
    tgt = torch.tensor([[char_to_idx['<SOS>']]]).to(device)
    
    # 챗봇이 대답을 만들어낼 공간입니다.
    answer = ""
    
    # 최대 30글자까지만 대답하도록 설정합니다.
    for _ in range(30):
        # 학습할 때처럼 컨닝 방지 마스크를 씌워 모델에 넣습니다.
        tgt_mask = model.generate_square_subsequent_mask(len(tgt)).to(device)
        
        with torch.no_grad(): # 실전에서는 기울기를 계산하지 않으므로 메모리를 절약합니다.
            output = model(src, tgt)
            
        # 모델이 예측한 가장 마지막 글자에서 가장 확률이 높은 글자의 번호를 뽑습니다.
        next_word_idx = output[-1, 0, :].argmax().item()
        
        # 만약 모델이 예측한 다음 글자가 끝(<EOS>) 기호라면 말을 멈춥니다.
        if next_word_idx == char_to_idx['<EOS>']:
            break
            
        # 숫자를 다시 우리가 아는 글자로 번역합니다.
        next_char = idx_to_char[next_word_idx]
        answer += next_char # 대답 문장에 글자를 하나씩 이어 붙입니다.
        
        # 방금 예측한 글자를 모델에게 다시 넣어서 그 다음 글자를 예측하게 만듭니다.
        next_word_tensor = torch.tensor([[next_word_idx]]).to(device)
        tgt = torch.cat([tgt, next_word_tensor], dim=0)

    # 완성된 챗봇의 대답을 화면에 보여줍니다.
    print(f"챗봇: {answer}")

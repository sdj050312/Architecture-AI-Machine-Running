# 필요한 라이브러리들을 불러옵니다.
import numpy as np # 수학적 계산과 배열 처리를 위한 라이브러리입니다.
import pandas as pd # 데이터를 표 형태로 다루기 위한 라이브러리입니다.
import matplotlib.pyplot as plt # 데이터와 예측 결과를 그래프로 그리기 위한 라이브러리입니다.
from sklearn.preprocessing import MinMaxScaler # 데이터의 크기를 0과 1 사이로 맞추기 위한 도구입니다.
from tensorflow.keras.models import Sequential # 딥러닝 모델을 층(layer) 단위로 쌓기 위한 도구입니다.
from tensorflow.keras.layers import LSTM, Dense # LSTM 층과 일반 신경망 층을 사용하기 위해 불러옵니다.

# ==========================================
# 1. 가상의 건설 시계열 데이터 생성
# ==========================================
# 실제 CSV 파일을 불러오려면 pd.read_csv('파일이름.csv')를 사용하면 됩니다.
# 여기서는 학생들이 바로 실행해 볼 수 있게 가상의 10년(120개월)치 건설 지출 데이터를 만듭니다.

# 1부터 120까지의 숫자를 만듭니다. (120개월의 시간 흐름을 의미합니다)
time_steps = np.arange(1, 121) 

# 시간에 따른 상승 추세(0.5 * 시간)와 계절성(sin 함수)을 결합하여 데이터를 만듭니다.
# 건설업 특성상 특정 계절에 지출이 늘어나고 줄어드는 현상을 흉내 낸 것입니다.
construction_spending = 0.5 * time_steps + 10 * np.sin(time_steps * 0.5) + np.random.normal(0, 2, 120)

# 만들어진 데이터를 보기 편하게 판다스(Pandas) 데이터프레임으로 변환합니다.
df = pd.DataFrame({'Month': time_steps, 'Spending': construction_spending})

# 학습에 사용할 '지출(Spending)' 열의 데이터만 추출하여 2차원 배열로 만듭니다. (LSTM 입력 형태를 맞추기 위함)
data = df['Spending'].values.reshape(-1, 1)

# ==========================================
# 2. 데이터 전처리 (스케일링)
# ==========================================
# 딥러닝 모델(특히 LSTM)은 데이터의 숫자가 너무 크면 학습이 잘 안 되는 성질이 있습니다.
# 그래서 데이터의 가장 작은 값은 0, 가장 큰 값은 1이 되도록 비율을 축소해 줍니다.
scaler = MinMaxScaler(feature_range=(0, 1))

# 데이터를 0과 1 사이의 값으로 변환(스케일링)합니다.
scaled_data = scaler.fit_transform(data)

# ==========================================
# 3. 데이터셋 분할 (학습용 vs 테스트용)
# ==========================================
# 전체 데이터의 80%는 과거 데이터로 모델을 학습하는 데 사용하고, 
# 나머지 20%는 미래 데이터로 가정하여 모델이 잘 예측하는지 시험하는 데 사용합니다.
train_size = int(len(scaled_data) * 0.8) # 80%에 해당하는 데이터 개수를 계산합니다.

# 처음부터 80% 지점까지를 학습 데이터(train_data)로 자릅니다.
train_data = scaled_data[:train_size]

# 80% 지점부터 끝까지를 시험용 데이터(test_data)로 자릅니다.
test_data = scaled_data[train_size:]

# ==========================================
# 4. 시퀀스 데이터 생성 (Windowing)
# ==========================================
# LSTM은 '과거의 연속된 데이터'를 보고 '다음 데이터'를 예측합니다.
# 이 함수는 데이터를 '과거 N개월치'와 '정답(다음 1개월치)'의 짝으로 묶어주는 역할을 합니다.
def create_dataset(dataset, look_back=1):
    X, Y = [], [] # X는 과거 데이터(문제), Y는 다음 데이터(정답)를 담을 빈 상자입니다.
    # 전체 데이터 길이에서 look_back을 뺀 만큼 반복합니다.
    for i in range(len(dataset) - look_back):
        # i부터 i+look_back 까지의 데이터를 잘라서 X에 넣습니다.
        X.append(dataset[i:(i + look_back), 0])
        # 바로 그 다음 데이터를 Y(정답)로 넣습니다.
        Y.append(dataset[i + look_back, 0])
    # 만들어진 리스트를 넘파이 배열로 변환하여 반환합니다.
    return np.array(X), np.array(Y)

# '과거 3개월'의 데이터를 보고 '다음 달'을 예측하도록 설정합니다. (학생들이 값을 바꿔볼 수 있습니다)
look_back = 3 

# 학습용 데이터와 시험용 데이터를 시퀀스 형태로 변환합니다.
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# LSTM 모델에 넣기 위해서는 데이터의 형태를 [데이터 개수, 과거 개월 수, 특징 개수] 인 3차원으로 만들어야 합니다.
# 특징은 '건설 지출액' 1개이므로 제일 끝에 1을 적어 형태를 변경합니다.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ==========================================
# 5. LSTM 모델 만들기
# ==========================================
# 딥러닝 모델이라는 빈 도화지를 꺼냅니다.
model = Sequential()

# 빈 도화지에 기억력이 좋은 LSTM 층을 올립니다.
# units=50은 두뇌의 크기(뉴런 수)를 의미하고, input_shape는 들어올 데이터의 형태를 알려주는 것입니다.
model.add(LSTM(units=50, input_shape=(look_back, 1)))

# 마지막으로 예측된 값을 하나로 모아줄 일반 신경망 층(Dense)을 추가합니다. (예측값이 1개이므로 1을 적습니다)
model.add(Dense(1))

# 모델이 학습할 때 어떤 방식으로 오차를 줄일지 설정합니다.
# optimizer='adam'은 오차를 줄여나가는 똑똑한 방식 중 하나이고, loss='mean_squared_error'는 오차를 계산하는 방법입니다.
model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 6. 모델 학습시키기
# ==========================================
# 준비된 문제(X_train)와 정답(y_train)을 주며 모델을 공부시킵니다.
# epochs=50 은 전체 데이터를 50번 반복해서 공부하라는 뜻입니다.
# batch_size=8 은 한 번에 8개씩 묶어서 문제를 풀라는 뜻입니다.
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# ==========================================
# 7. 예측 및 결과 시각화
# ==========================================
# 시험용 데이터(X_test)를 주고 미래를 예측해 보라고 합니다.
test_predict = model.predict(X_test)

# 앞서 데이터를 0과 1 사이로 압축했었기 때문에, 우리가 알아볼 수 있는 원래 숫자로 다시 되돌려줍니다.
test_predict = scaler.inverse_transform(test_predict)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# 결과를 눈으로 확인하기 위해 그래프를 그립니다.
plt.figure(figsize=(12, 6)) # 그래프의 가로, 세로 크기를 정합니다.

# 실제 정답(파란 점선)을 그래프에 그립니다.
plt.plot(y_test_original, label='Actual Construction Spending', color='blue', linestyle='--')

# 모델이 예측한 값(빨간 실선)을 그래프에 그립니다.
plt.plot(test_predict, label='Predicted Spending (LSTM)', color='red')

# 그래프의 제목과 X축, Y축 이름, 범례를 달아줍니다.
plt.title('Construction Spending Forecast using LSTM')
plt.xlabel('Time (Months)')
plt.ylabel('Spending')
plt.legend()

# 화면에 그래프를 띄웁니다.
plt.show()

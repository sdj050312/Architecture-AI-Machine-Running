from sklearn.linear_model import LinearRegression
import numpy as np

# 센서 데이터 (온도, 습도, 인원수)
X = np.array([
    [25, 50, 5],
    [27, 55, 10],   
    [28, 60, 15],
    [30, 65, 20],
])

Y = np.array([0, 0, 1, 1])  # 0: 에너지 절약 모드, 1: 일반 모드
# 모델 생성
model = LinearRegression()

# 모델학습
model.fit(X, Y)

#새로운 센서 데이터 
temperature = 29
humidity = 58   
people = 9 

ac_power = model.predict([[temperature, humidity, people]])
print("추천 에어컨 출력:", ac_power[0])
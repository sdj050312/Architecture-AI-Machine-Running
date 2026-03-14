from sklearn.linear_model import LinearRegression
import numpy as np

# 입력 데이터 (온도, 습도, 풍량)
X= np.array([
[25, 50, 200],
    [27, 55, 220],
    [28, 60, 250]
])

# 결과 데이터 (냉방 부하)

Y = np.array([5.2, 6.1, 7.0]) 
# 모델 생성
model = LinearRegression()
# 모델 학습
model.fit(X, Y)

# 예측하기 온도 29도, 습도 65%, 풍량 260으로 냉방부하 
predicted_load = model.predict([[29, 65, 260]])

print("내가 예측한 냉방 부하:", predicted_load[0])

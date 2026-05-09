#Linear Regression 이란? 선형회귀는 연속형 값을 예측하기 위한 지도학습 알고리즘이다. 입력변수와 출력변수간의 선형
#관계를 수학적 모형으로 표현하여 예측, 추정, 트랜드 분석등에 널리 사용된다. 단순하면서도 통계적해석이 용이해 머신러닝의 기초모델로 중요하다 
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터를 배열로 정리하고 (X는 공부시간, Y는 시험점수임!)
X = np.array([[1], [2], [3], [4], [5]])

#y는 결과 시험점수 
Y = np.array([55, 65, 75, 84, 95])

def classification (x, y): 
    if (x < 4):
        return ;    
    else: 
        return "높음"

print(classification(2, 65))


# 모델 생성 
model = LinearRegression()

# 모델 학습
model.fit(X, Y)

# 예측하기
predicted_score = model.predict([[6]])
print("예측점수:", predicted_score[0])


# 터미널 노드 데이터를 저장하는 최소단위 
# 터미널이라는 끝이라는 뜻이고, none 터미널 노드, 어떠한 노드보다 위에 있다, 그러면 
# parant node 
# child node 
# terminal node
# split node 데이터를 구분하기 위한 함수 if (x1 < 1, y= 1, y= 2)

import numpy as np  
import sklearn.tree as plot_tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt



# 1. 콘크리트 데이터 
# [cement, water, fine_agg, coarse_agg, curing_days]

x = np.array([[540, 162, 1040, 1130, 28],
              [540, 162, 1055, 1130, 28], ])

# y는 콘크리트의 압축강도 
y = np.array([28, 27])

# 모델생성 

model = DecisionTreeRegressor(max_depth = 4)

model.fit(x, y)

new_mix = np.array([[540, 162, 1050, 1130, 28]])

predicted_strength = model.predict(new_mix)

print(predicted_strength)

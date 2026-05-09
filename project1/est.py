import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 한글 깨짐 방지 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드 (사용자가 입력한 진짜 데이터)
df = pd.read_csv('construction_data.csv')

# 2. 시각화 1: 면적(Area)과 공사비(Cost)의 상관관계 그래프
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='area', y='cost', scatter_kws={'s':100}, line_kws={'color':'red'})
plt.title('면적 대비 공사비 추세 분석')
plt.xlabel('연면적 (㎡)')
plt.ylabel('공사비 (만 원)')
plt.grid(True)
plt.show()

# 3. 모델 학습을 위한 인코딩
df_ml = pd.get_dummies(df, columns=['structure', 'location'])
X = df_ml.drop('cost', axis=1)
y = df_ml['cost']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. 시각화 2: 어떤 요소가 공사비에 가장 큰 영향을 주는가? (Feature Importance)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('공사비 결정 요인 중요도 (AI 분석)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

print("✅ 분석 및 시각화가 완료되었습니다.")
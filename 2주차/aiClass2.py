import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# 1. 파일 설정 (내 컴퓨터 경로)
file_name = 'Concrete_Data.xls' 

if os.path.exists(file_name):
    # 탭(\t) 구분자 데이터 로드 및 결측치 제거
    df = pd.read_csv(file_name, sep='\t')
    df = df.dropna()
    print("✅ 데이터 로드 완료!")
else:
    print(f"❌ '{file_name}' 파일을 찾을 수 없습니다.")
    exit()

# 2. 수치형 데이터 추출 및 정규화
X = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 최적의 k 찾기 분석 (2~10까지)
k_range = range(2, 11)
scores = []
best_k = 4  # 스크린샷에 맞춰 4로 설정 (또는 자동 계산 가능)
best_score = -1

print("\n--- 군집 분석 시작 ---")
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # 실루엣 점수 계산
    current_score = silhouette_score(X_scaled, labels)
    scores.append(current_score)
    print(f"k={k}일 때 실루엣 점수: {current_score:.4f}")

# 4. 시각화 구성 (스크린샷과 동일한 레이아웃)
plt.figure(figsize=(14, 6))

# [왼쪽] Silhouette Method 그래프
plt.subplot(1, 2, 1)
plt.plot(k_range, scores, 'go-', markersize=8)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
plt.title('Silhouette Method for Optimal k', fontsize=14)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()

# [오른쪽] PCA 결과 그래프 (4개 군집)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 최적 k로 최종 학습
final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
final_labels = final_kmeans.fit_predict(X_scaled)

plt.subplot(1, 2, 2)
for i in range(best_k):
    plt.scatter(X_pca[final_labels == i, 0], X_pca[final_labels == i, 1], 
                label=f'Cluster {i}', s=40, edgecolors='white', alpha=0.8)

plt.title(f'Final Clustering Result (k={best_k})', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.tight_layout()
plt.show()

# 5. 중심점(Centroid) 정보 출력
centroids_scaled = final_kmeans.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_orig, columns=X.columns)
print("\n--- 각 군집별 특징(평균값) ---")
print(centroids_df)
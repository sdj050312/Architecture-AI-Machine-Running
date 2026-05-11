import pandas as pd  # 데이터프레임 처리를 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프 시각화를 위한 라이브러리
from sklearn.cluster import KMeans  # K-Means 군집화 알고리즘
from sklearn.metrics import silhouette_score  # 군집 품질 평가를 위한 실루엣 계수
from sklearn.preprocessing import StandardScaler  # 데이터 표준화(평균 0, 분산 1)를 위한 도구
from sklearn.decomposition import PCA  # 고차원 데이터를 2차원으로 축소하기 위한 도구
import os  # 파일 경로 및 시스템 조작을 위한 라이브러리

# 1. 구글 드라이브 연결 (코랩 환경에서 드라이브 파일에 접근하기 위함)
from google.colab import drive
drive.mount('/content/drive')

# 2. 데이터 불러오기 설정
file_path = '/content/drive/MyDrive/AI기반 건축공학/2주차_Concrete_Data.xls'

# 파일이 해당 경로에 존재하는지 확인
if os.path.exists(file_path):
    # 엑셀 파일을 판다스 데이터프레임으로 읽어오기
    df = pd.read_excel(file_path)
    print("데이터 로드 완료. 데이터의 첫 5개 행을 확인합니다:")
    print(df.head())

    # 분석에 사용할 수치형 데이터(숫자 데이터)만 따로 추출
    X = df.select_dtypes(include=[np.number])

    # 3. 데이터 정규화 (K-Means는 거리를 기반으로 하므로 정규화가 필수입니다)
    # 각 변수(특성)들의 단위를 맞추기 위해 StandardScaler를 사용하여 표준화합니다.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # 데이터를 학습시키고 동시에 변환합니다.

    # 4. 최적의 k(군집 수) 찾기 (실루엣 계수 활용)
    k_range = range(2, 11)  # 군집 수를 2개부터 10개까지 테스트해봅니다.
    best_k = -1  # 가장 좋은 k값을 저장할 변수 초기화
    best_score = -1  # 가장 높은 실루엣 점수를 저장할 변수 초기화
    scores = []  # 그래프를 그리기 위해 각 k별 점수를 저장할 리스트

    for k in k_range:
        # k개의 군집으로 모델 설정 (init='k-means++'는 효율적인 초기 중심점 설정을 의미)
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        # 데이터를 학습시키고 각 데이터가 어느 군집에 속하는지 라벨을 예측합니다.
        labels = kmeans.fit_predict(X_scaled)

        # 실루엣 점수 계산 (1에 가까울수록 군집화가 잘 된 것임)
        score = silhouette_score(X_scaled, labels)
        scores.append(score) # 점수 저장
        print(f"군집 수 k={k}일 때, 실루엣 점수: {score:.4f}")

        # 만약 현재 점수가 이전까지의 최고점보다 높다면 최적의 k로 갱신
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✅ 최적의 군집 수: {best_k} (최고 점수: {best_score:.4f})")

    # 5. 최적의 k로 최종 군집화 수행
    # 위에서 찾은 'best_k'를 사용하여 최종 모델을 만듭니다.
    final_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    # 최종적으로 데이터의 군집 라벨(0, 1, 2...)을 생성합니다.
    final_labels = final_kmeans.fit_predict(X_scaled)

    # 원본 데이터프레임에 'Cluster'라는 열을 새로 만들어 군집 결과를 추가합니다.
    df['Cluster'] = final_labels

    # 6. 결과 시각화 (PCA 차원 축소 활용)
    # 데이터는 변수가 많아 한 눈에 볼 수 없으므로, PCA를 통해 2차원으로 축소하여 점을 찍습니다.
    pca = PCA(n_components=2) # 2차원으로 설정
    X_pca = pca.fit_transform(X_scaled) # 정규화된 데이터를 PCA로 변환

    plt.figure(figsize=(12, 5)) # 그래프 크기 설정

    # [왼쪽] k값에 따른 실루엣 점수 변화 그래프
    plt.subplot(1, 2, 1)
    plt.plot(k_range, scores, 'go-') # 초록색 점과 선으로 점수 표시
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}') # 최적 k 지점에 빨간 점선
    plt.title('Silhouette Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.legend()

    # [오른쪽] 2차원으로 축소된 실제 군집 결과 분포도
    plt.subplot(1, 2, 2)
    for i in range(best_k):
        # 각 군집별로 데이터를 필터링하여 산점도(Scatter plot)를 그립니다.
        plt.scatter(X_pca[final_labels == i, 0], X_pca[final_labels == i, 1], label=f'Cluster {i}')

    plt.title(f'Final Clustering Result (k={best_k})')
    plt.xlabel('PCA Component 1') # 첫 번째 주성분
    plt.ylabel('PCA Component 2') # 두 번째 주성분
    plt.legend()

    plt.tight_layout() # 그래프 간격 자동 조정
    plt.show() # 화면에 출력

    # 7. 각 군집의 중심점(Centroid) 정보 추출 및 역변환
    # 학습된 모델에서 정규화된 상태의 중심점 위치를 가져옵니다.
    centroids_scaled = final_kmeans.cluster_centers_
    # StandardScaler로 변형된 값을 다시 우리가 이해할 수 있는 원본 수치(inverse)로 되돌립니다.
    centroids_orig = scaler.inverse_transform(centroids_scaled)

    # 추출한 중심점 데이터를 보기 좋게 데이터프레임으로 만듭니다.
    centroids_df = pd.DataFrame(centroids_orig, columns=X.columns)
    # 맨 앞에 군집 ID(0, 1, 2...)를 추가하여 구분하기 쉽게 만듭니다.
    centroids_df.insert(0, 'Cluster_ID', range(best_k))

    # 8. 분석 결과 저장
    # 파일 이름을 k값에 맞춰 생성합니다.
    centroid_filename = f'Concrete_Centroids_k{best_k}.csv'
    # 원본 데이터가 있던 폴더 경로를 가져와 저장 경로를 생성합니다.
    centroid_save_path = os.path.join(os.path.dirname(file_path), centroid_filename)

    # 결과를 CSV 파일로 저장 (한글 깨짐 방지를 위해 utf-8-sig 사용)
    centroids_df.to_csv(centroid_save_path, index=False, encoding='utf-8-sig')

    print("\n--- 군집별 특징(중심점) 정보 (원본 스케일) ---")
    print(centroids_df)
    print(f"\n🎉 분석 결과 파일이 저장되었습니다: {centroid_save_path}")

else:
    # 파일 경로가 틀렸을 경우 출력되는 메시지
    print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다. 구글 드라이브 경로를 다시 확인해주세요.")









#3주차입니다. 
#데이터가 앞선 예제처럼 3차원이면 어떡함? 군집이 몇개인지 알 수 있음? 모른다...
#데이터가 5,6 차원이면 데이터가 몇개가 적합한지 알 수 가 없음... 몇이 적합하다!
#군집 내 데이터들이 얼마나 해당 군집접과 가까이 모여 있는지를 나타내는 지표
# 실루엣계수 다른군집과는 얼마나 멀리떨어져있는지 분리도를 나타냄, SSE의 평균값 
# 군집화는 데이터간 거리에 기반하여 유사한데이터를 묶는 알고리즘 
# 만약, 아래와 같이 "정규화를 해야합니다."
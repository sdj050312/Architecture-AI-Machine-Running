import numpy as np
import matplotlib.pyplot as plt

#함수 만들기 
def kmeans_with_random_init(X, k, max_iters=100):
    # 데이터의 각 차원별 최소/최대값을 구해 범위를 설정합니다.
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # 1. 초기 중심점(centroids) 설정: 데이터 범위 내에서 무작위 좌표 생성
    # [min, max] 범위 내에서 k개의 (x, y) 좌표를 생성합니다.
    centroids = np.zeros((k, 2))
    centroids[:, 0] = np.random.uniform(x_min, x_max, k)
    centroids[:, 1] = np.random.uniform(y_min, y_max, k)

    centroid_history = [centroids.copy()]

    for i in range(max_iters):
        # 2. Step 1: 거리 계산 및 군집 할당 (Assignment)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Step 2: 중심점 업데이트 (Update)
        new_centroids = []
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # 군집에 데이터가 없으면 기존 위치 유지 또는 다시 무작위 생성
                new_centroids.append(centroids[j])

        new_centroids = np.array(new_centroids)
        centroid_history.append(new_centroids.copy())

        # 수렴 확인 (변화량이 매우 적을 때 종료)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            print(f"Convergence reached at iteration {i}")
            break

        centroids = new_centroids

    return centroids, labels, np.array(centroid_history)

# --- 실행 및 시각화 ---
X_train = np.array([[1, 2], [2, 1], [4, 5], [5, 4], [2, 2], [4, 4], [1, 1], [5, 5]])
k = 2

final_centroids, labels, history = kmeans_with_random_init(X_train, k)

print(final_centroids)
print(labels)
print(history)

plt.figure(figsize=(8, 6))

# 데이터 포인트
colors = ['lightcoral', 'lightskyblue']
for j in range(k):
    plt.scatter(X_train[labels == j][:, 0], X_train[labels == j][:, 1],
                c=colors[j], label=f'Cluster {j}', s=80, edgecolors='black', alpha=0.7)

# 궤적 플롯
for j in range(k):
    traj = history[:, j, :]
    plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=5, label=f'Trajectory $\mathbf{{c}}_{j+1}$')
    # 시작점 표시
    plt.scatter(traj[0, 0], traj[0, 1], c='red', marker='x', s=100, zorder=5)
    # 종료점 표시
    plt.scatter(traj[-1, 0], traj[-1, 1], c='yellow', marker='*', s=200, edgecolors='black', zorder=5)

plt.title('K-means: Random Coordinate Initialization', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
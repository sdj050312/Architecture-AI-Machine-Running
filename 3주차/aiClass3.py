import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =====================================================================
# 1. 의사결정나무(CustomRegressionTree) 클래스
#    - 랜덤 포레스트의 구성 요소가 되는 개별 나무입니다.
# =====================================================================
class CustomRegressionTree:
    def __init__(self, max_depth=3, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features  # 랜덤 포레스트를 위한 변수 무작위 선택 개수
        self.tree = {}
        self.importances = []
        self.total_samples = 0

    def fit(self, X, y):
        """ 입력 데이터를 받아 나무를 학습시킵니다. """
        self.total_samples = len(y)
        self.importances = np.zeros(X.shape[1])  
        self.tree = self._build_tree(X, y, depth=0)
        
        # 중요도를 100% 기준으로 정규화합니다.
        if np.sum(self.importances) > 0:
            self.importances = (self.importances / np.sum(self.importances)) * 100

    def _build_tree(self, X, y, depth):
        """ 데이터를 최적으로 분할하여 재귀적으로 나무를 구축합니다. """
        n_samples = len(y)
        variance = np.var(y)

        # [종료 조건] 깊이 초과, 데이터 부족, 혹은 오차가 0일 때
        if depth >= self.max_depth or n_samples <= 1 or variance == 0:
            return {'is_leaf': True, 'value': np.mean(y), 'samples': n_samples}

        best_split = None
        max_var_reduction = 0

        # [Randomness] 전체 변수 중 일부(max_features)만 무작위로 골라 탐색합니다.
        n_features = X.shape[1]
        features_to_try = range(n_features)
        if self.max_features is not None:
            features_to_try = np.random.choice(n_features, self.max_features, replace=False)

        for feature_idx in features_to_try:
            unique_vals = np.unique(X[:, feature_idx])
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for thresh in thresholds:
                left_mask = X[:, feature_idx] <= thresh
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0: continue

                # 분산 감소량(오차 개선 정도) 계산
                w_left, w_right = len(y_left) / n_samples, len(y_right) / n_samples
                weighted_var = (w_left * np.var(y_left)) + (w_right * np.var(y_right))
                var_reduction = variance - weighted_var

                if var_reduction > max_var_reduction:
                    max_var_reduction = var_reduction
                    best_split = {
                        'feature_idx': feature_idx, 'threshold': thresh,
                        'left_mask': left_mask, 'right_mask': right_mask
                    }

        # 나눌 기준이 없으면 잎으로 확정
        if best_split is None or max_var_reduction == 0:
            return {'is_leaf': True, 'value': np.mean(y), 'samples': n_samples}

        # 변수 중요도 누적
        weight = n_samples / self.total_samples
        self.importances[best_split['feature_idx']] += weight * max_var_reduction

        # 왼쪽/오른쪽 자식 노드 재귀 생성
        left_child = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_child = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'is_leaf': False, 'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'], 'var_reduction': max_var_reduction,
            'samples': n_samples, 'left': left_child, 'right': right_child
        }

    def predict(self, X):
        """ 학습된 나무를 기반으로 새로운 데이터를 예측합니다. """
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_row(self, row, node):
        if node['is_leaf']: return node['value']
        if row[node['feature_idx']] <= node['threshold']:
            return self._predict_row(row, node['left'])
        else:
            return self._predict_row(row, node['right'])


# =====================================================================
# 2. 랜덤 포레스트(CustomRandomForest) 클래스
#    - 여러 개의 나무를 모아 평균을 내는 앙상블 모델입니다.
# =====================================================================
class CustomRandomForest:
    def __init__(self, n_trees=50, max_depth=5, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """ 여러 개의 나무를 각각 다른 데이터(Bootstrap)로 학습시킵니다. """
        self.trees = []
        n_samples, n_features = X.shape
        
        # 회귀 관례에 따라 변수 개수의 1/3을 무작위 선택수로 설정
        if self.max_features is None:
            self.max_features = max(1, int(n_features / 3))

        for _ in range(self.n_trees):
            # [Bootstrap] 복원 추출로 중복을 허용한 샘플링 수행
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_b, y_b = X[indices], y[indices]

            tree = CustomRegressionTree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_b, y_b)
            self.trees.append(tree)

    def predict(self, X):
        """ 모든 나무의 예측값을 평균내어 최종 결과를 반환합니다. """
        all_tree_preds = np.array([tree.predict(X) for tree in self.trees])
        rf_preds = np.mean(all_tree_preds, axis=0)
        return rf_preds, all_tree_preds


# =====================================================================
# 3. 데이터 로드 및 모델 실행
# =====================================================================

file_path = 'Concrete_Data.xls'  # 데이터 파일 경로

if os.path.exists(file_path):
    print("✅ 데이터 로딩 중...")
    df = pd.read_csv(file_path, sep='\t')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    # 시각화를 위한 100개 테스트 샘플 추출
    np.random.seed(42)
    test_idx = np.random.choice(len(y), 100, replace=False)
    X_test, y_test = X[test_idx], y[test_idx]

    # 모델 학습 (트리 50개)
    rf_model = CustomRandomForest(n_trees=50, max_depth=5)
    print("⚙️ 랜덤 포레스트 학습 시작 (50개 트리 생성 중)...")
    rf_model.fit(X, y)
    print("✅ 학습 완료!\n")

    # 예측 수행
    rf_preds, tree_preds = rf_model.predict(X_test)

    # --- 시각화 ---
    print("📊 결과 그래프를 생성합니다...")
    sorted_idx = np.argsort(y_test) # 정답 순서대로 정렬 (그래프 확인용)
    
    plt.figure(figsize=(14, 7))
    for i in range(tree_preds.shape[0]):
        label = 'Individual Trees' if i == 0 else ""
        plt.plot(tree_preds[i, sorted_idx], color='dodgerblue', alpha=0.1, label=label)

    plt.plot(rf_preds[sorted_idx], color='red', linewidth=3, label='Random Forest (Ensemble)')
    plt.plot(y_test[sorted_idx], color='black', linewidth=2, linestyle='--', label='Actual Strength')

    plt.title('Final Result: Random Forest vs Actual Data', fontsize=16)
    plt.ylabel('Compressive Strength (MPa)')
    plt.xlabel('Sorted Sample Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

else:
    print(f"❌ 파일 없는데요 ㅠㅠ: {file_path}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# =====================================================================
# 1. 의사결정나무(회귀 트리) 모델 클래스 정의
# =====================================================================
class CustomRegressionTree:
    def __init__(self, max_depth=3):
        """ 나무의 기본 설정을 초기화하는 생성자입니다. """
        self.max_depth = max_depth       # 나무가 아래로 몇 단계까지 자랄지 결정 (과적합 방지용)
        self.tree = {}                   # 학습된 트리 구조(가지와 잎)를 저장할 딕셔너리
        self.importances = []            # 각 변수가 예측에 기여한 정도를 기록할 리스트
        self.total_samples = 0           # 전체 학습 데이터 개수 (중요도 계산용)

    def fit(self, X, y):
        """ 데이터를 입력받아 학습을 시작하는 메인 함수입니다. """
        self.total_samples = len(y)              # 데이터 총 개수 저장
        self.importances = np.zeros(X.shape[1])  # 변수 개수만큼 중요도 저장소를 0으로 초기화
        self.tree = self._build_tree(X, y, depth=0) # 깊이 0부터 나무 만들기(재귀) 시작
        
        # 학습 종료 후 중요도 합계를 100% 기준으로 정규화
        if np.sum(self.importances) > 0:
            self.importances = (self.importances / np.sum(self.importances)) * 100

    def _build_tree(self, X, y, depth):
        """ 데이터를 최적으로 분할하며 가지를 뻗는 핵심 재귀 함수입니다. """
        n_samples = len(y)               # 현재 노드의 데이터 개수
        variance = np.var(y)             # 현재 노드의 타겟값(강도) 분산(오차) 계산

        # [종료 조건] 최대 깊이 도달, 데이터 1개 이하, 혹은 오차가 0이면 '잎(Leaf)' 반환
        if depth >= self.max_depth or n_samples <= 1 or variance == 0:
            return {'is_leaf': True, 'value': np.mean(y), 'samples': n_samples}

        best_split = None                # 최적의 분할 기준 저장용
        max_var_reduction = 0            # 분산이 얼마나 줄었는지 최대치 기록용

        # 모든 변수(Feature)를 하나씩 돌아가며 최적의 칼질 지점을 찾습니다.
        for feature_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, feature_idx]) # 중복 제거 후 값 추출
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 # 인접 값의 중간을 기준점 후보로 설정

            for thresh in thresholds:
                # 기준점보다 작거나 같으면 왼쪽(Left), 크면 오른쪽(Right)
                left_mask = X[:, feature_idx] <= thresh
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0: continue # 한쪽이 비면 무시

                # 나뉜 두 그룹의 오차(가중 평균 분산) 계산
                w_left, w_right = len(y_left) / n_samples, len(y_right) / n_samples
                weighted_var = (w_left * np.var(y_left)) + (w_right * np.var(y_right))
                
                # 분할 전후의 오차 감소량(이득) 계산
                var_reduction = variance - weighted_var

                # 지금까지 중 가장 성능이 좋은(오차를 많이 줄인) 기준이면 저장
                if var_reduction > max_var_reduction:
                    max_var_reduction = var_reduction
                    best_split = {
                        'feature_idx': feature_idx, 'threshold': thresh,
                        'left_mask': left_mask, 'right_mask': right_mask
                    }

        # 만약 더 이상 오차를 줄일 수 없다면 잎으로 확정
        if best_split is None or max_var_reduction == 0:
            return {'is_leaf': True, 'value': np.mean(y), 'samples': n_samples}

        # [변수 중요도 기록] 해당 변수가 기여한 만큼 점수 누적
        weight = n_samples / self.total_samples
        self.importances[best_split['feature_idx']] += weight * max_var_reduction

        # 최적 기준으로 데이터를 쪼개서 자식 노드들을 다시 만듦 (재귀)
        left_child = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_child = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'is_leaf': False, 'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'], 'var_reduction': max_var_reduction,
            'samples': n_samples, 'left': left_child, 'right': right_child
        }

    def plot_tree(self, feature_names):
        """ 학습된 나무를 그림으로 보여주는 함수입니다. """
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.axis('off') # 그래프 축 숨김
        self._draw_node(self.tree, ax, x=0.5, y=1.0, dx=0.25, dy=0.15, feature_names=feature_names)
        plt.show()

    def _draw_node(self, node, ax, x, y, dx, dy, feature_names):
        """ 트리 시각화를 위해 박스와 선을 그리는 내부 함수입니다. """
        if node['is_leaf']:
            text = f"Leaf\nValue: {node['value']:.1f}\n(n={node['samples']})"
            bbox_props = dict(boxstyle="round,pad=0.3", fc="#e9ecef", ec="#adb5bd", lw=2)
        else:
            text = f"{feature_names[node['feature_idx']]} <= {node['threshold']:.1f}\nVar Red: {node['var_reduction']:.2f}\n(n={node['samples']})"
            bbox_props = dict(boxstyle="round,pad=0.3", fc="#cfe2ff", ec="#0d6efd", lw=2)

        ax.text(x, y, text, ha='center', va='center', bbox=bbox_props, fontsize=9)

        if not node['is_leaf']:
            # 왼쪽/오른쪽 자식과 연결 선 그리기
            ax.plot([x, x - dx], [y - 0.03, y - dy + 0.03], color='gray', zorder=0)
            ax.plot([x, x + dx], [y - 0.03, y - dy + 0.03], color='gray', zorder=0)
            self._draw_node(node['left'], ax, x - dx, y - dy, dx / 2, dy, feature_names)
            self._draw_node(node['right'], ax, x + dx, y - dy, dx / 2, dy, feature_names)


# =====================================================================
# 2. 데이터 불러오기 및 실행
# ======================================

file_path = 'Concrete_Data.xls'

if os.path.exists(file_path):
    print("✅ 데이터를 불러오는 중입니다...")
    df = pd.read_csv(file_path, sep='\t')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    print("✅ 데이터 로드 완료!\n")
    # 마지막 열(강도)을 타겟(y)으로, 나머지를 입력 변수(X)로 설정
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 모델 생성 및 학습
    tree_model = CustomRegressionTree(max_depth=3)
    print("⚙️ 모델 학습 중...")
    tree_model.fit(X, y)
    print("✅ 학습 완료!\n")

    # 변수 중요도 출력
    print("--- ⭐ 변수 중요도 ---")
    importances_dict = {name: imp for name, imp in zip(feature_names, tree_model.importances)}
    for name, importance in sorted(importances_dict.items(), key=lambda x: x[1], reverse=True):
        if importance > 0: print(f"  - {name}: {importance:.2f}%")

    # 트리 다이어그램 출력
    print("\n📊 트리 시각화:")
    tree_model.plot_tree(feature_names)

else:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")

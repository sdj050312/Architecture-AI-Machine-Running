import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# =====================================================================
# 1. 인공신경망 클래스 (이미지 속 역전파 원리 구현)
# =====================================================================
class ConcreteNeuralNet:
    def __init__(self, layers, learning_rate=0.001, alpha=0.01):
        self.learning_rate = learning_rate
        self.alpha = alpha  
        self.params = {}
        self.L = len(layers) - 1
        
        # 가중치 초기화 (He Initialization)
        for i in range(1, len(layers)):
            self.params[f'W{i}'] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2. / layers[i-1])
            self.params[f'b{i}'] = np.zeros((1, layers[i]))

    def leaky_relu(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

    def forward(self, X):
        cache = {'A0': X}
        for i in range(1, self.L + 1):
            cache[f'Z{i}'] = np.dot(cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            if i < self.L:
                cache[f'A{i}'] = self.leaky_relu(cache[f'Z{i}'])
            else:
                cache[f'A{i}'] = cache[f'Z{i}'] 
        return cache[f'A{self.L}'], cache

    def backward(self, cache, y):
        grads = {}
        m = y.shape[0]
        y_hat = cache[f'A{self.L}']
        dZ = (y_hat - y) / m 

        for i in reversed(range(1, self.L + 1)):
            grads[f'dW{i}'] = np.dot(cache[f'A{i-1}'].T, dZ)
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                dA_prev = np.dot(dZ, self.params[f'W{i}'].T)
                dZ = dA_prev * self.leaky_relu_derivative(cache[f'Z{i-1}'])
        return grads

    def train(self, X, y, epochs=2000):
        for epoch in range(epochs):
            y_hat, cache = self.forward(X)
            grads = self.backward(cache, y)
            for i in range(1, self.L + 1):
                self.params[f'W{i}'] -= self.learning_rate * grads[f'dW{i}']
                self.params[f'b{i}'] -= self.learning_rate * grads[f'db{i}']
            if epoch % 500 == 0:
                mse = np.mean((y_hat - y)**2)
                print(f"Epoch {epoch}, MSE: {mse:.4f}")

# =====================================================================
# 2. 데이터 실행부 (결측치 제거 및 수치 안정화 적용)
# =====================================================================
file_path = "Concrete_Data.xls"  # 구글 드라이브에 업로드한 엑셀 파일 경로

if os.path.exists(file_path):
    print("✅ 데이터를 불러옵니다...")
    df = pd.read_csv(file_path, sep='\t')

    # [추가] 결측치(NaN) 행 제거
    df = df.dropna()
    print(f"🧹 결측치 제거 완료. 남은 데이터 수: {len(df)}")

    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # [중요] X 데이터 정규화 (분모가 0이 되는 것 방지)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1.0 # 편차가 0인 컬럼 처리
    X_scaled = (X - X_mean) / (X_std + 1e-8)
    
    # [중요] y 데이터 정규화 (회귀 예측값 발산 방지)
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / (y_std + 1e-8)

    # 샘플링 (테스트용 100개)
    np.random.seed(42)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    test_indices = indices[:100]
    
    X_test_scaled = X_scaled[test_indices]
    y_test_original = y[test_indices] # 시각화 비교용 원본값

    # 모델 생성 (학습률을 0.005로 살짝 낮춰 더 안정적으로 학습)
    nn = ConcreteNeuralNet(layers=[X.shape[1], 16, 8, 1], learning_rate=0.005)

    print("🚀 신경망 학습을 시작합니다...")
    nn.train(X_scaled, y_scaled, epochs=3000)

    # 예측 및 원복 (Inverse Scaling)
    y_pred_scaled, _ = nn.forward(X_test_scaled)
    y_pred = (y_pred_scaled * y_std) + y_mean # 정규화된 값을 다시 MPa 단위로

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred, alpha=0.6, color='darkgreen', label='Neural Net Prediction')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Match')
    plt.xlabel("Actual Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title("Concrete Strength Prediction - Neural Network (Stable)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

else:
    print(f"⚠️ 파일을 찾을 수 없습니다: {file_path}")
